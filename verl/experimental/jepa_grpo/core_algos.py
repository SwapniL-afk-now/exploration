# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JEPA-GRPO algorithm primitives.

All pure differentiable math lives here:
  - Dr.GRPO policy loss (group-centered, no std normalization)
  - Epps-Pulley CF test statistic (vectorized)
  - SIGReg (Sketched Isotropic Gaussian Regularization, LeJEPA)
  - LeJEPA loss (squared-Euclidean alignment + SIGReg)
  - LLM-JEPA loss (cosine-distance prediction alignment + SIGReg)

References:
  - Balestriero et al. 2025 "LeJEPA" arXiv:2511.08544
  - Huang, LeCun, Balestriero 2025 "LLM-JEPA" arXiv:2509.14252
  - Equation doc: JEPA-GRPO-Equation.md
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# Re-export for trainer convenience
from verl.experimental.fepo.core_algos import compute_group_advantages  # noqa: F401


# ---------------------------------------------------------------------------
# Dr.GRPO policy loss
# ---------------------------------------------------------------------------

def dr_grpo_loss(
    current_logps: torch.Tensor,    # (T_total,) flat token log-probs, current policy
    old_logps: torch.Tensor,        # (T_total,) flat, detached, from rollout time
    advantages: torch.Tensor,       # (T_total,) flat, per-token broadcast of group advantage
    loss_mask: torch.Tensor,        # (T_total,) bool — True = valid completion token
    clip_eps: float = 0.2,
    token_log_ratio_clip: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dr.GRPO clipped policy loss.

    Group-centered advantages, NO std normalization (Dr.GRPO variant).
    Advantages are pre-computed by the caller via compute_group_advantages.
    """
    if current_logps.numel() == 0 or loss_mask.sum() == 0:
        zero = current_logps.sum() * 0.0
        return zero, {
            "grpo_pg_loss": 0.0,
            "grpo_clip_fraction": 0.0,
            "grpo_token_ratio_mean": 1.0,
            "grpo_log_ratio_abs_mean": 0.0,
            "grpo_loss_token_count": 0.0,
        }

    old_logps = old_logps.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    advantages = advantages.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    mask = loss_mask.to(device=current_logps.device).bool().detach()

    cur = current_logps[mask]
    old = old_logps[mask]
    adv = advantages[mask]

    log_ratio = (cur - old).clamp(-token_log_ratio_clip, token_log_ratio_clip)
    ratio = log_ratio.exp()

    unclipped = ratio * adv
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * adv
    pg_loss = -torch.min(unclipped, clipped).mean()

    clip_frac = ((ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)).float().mean()

    return pg_loss, {
        "grpo_pg_loss": float(pg_loss.detach().cpu()),
        "grpo_clip_fraction": float(clip_frac.detach().cpu()),
        "grpo_token_ratio_mean": float(ratio.detach().mean().cpu()),
        "grpo_log_ratio_abs_mean": float(log_ratio.detach().abs().mean().cpu()),
        "grpo_loss_token_count": float(mask.sum().cpu()),
    }


# ---------------------------------------------------------------------------
# Epps-Pulley CF test statistic (vectorized, single direction)
# ---------------------------------------------------------------------------

def epps_pulley_statistic(
    z: torch.Tensor,    # (N,) 1-D projected sample
    t: torch.Tensor,    # (K,) frequency grid
    s: float = 1.0,     # Gaussian-tapered kernel bandwidth (LeJEPA default)
) -> torch.Tensor:
    """Epps-Pulley test statistic comparing empirical CF of z to CF of N(0,1).

    Uses a Gaussian-tapered kernel w(t) = exp(-t²/s²) as in the LeJEPA paper
    (NOT the classical 1/t² weighting).
    """
    angles = z.unsqueeze(-1) * t          # (N, K)
    emp_cf_real = torch.cos(angles).mean(dim=0)   # (K,)
    emp_cf_imag = torch.sin(angles).mean(dim=0)   # (K,)

    gaussian_cf = torch.exp(-0.5 * t ** 2)        # (K,)  CF of N(0,1) is real-valued
    weights = torch.exp(-t ** 2 / s ** 2)          # (K,)

    diff_real = emp_cf_real - gaussian_cf
    return (weights * (diff_real ** 2 + emp_cf_imag ** 2)).sum()


# ---------------------------------------------------------------------------
# SIGReg — Sketched Isotropic Gaussian Regularization (LeJEPA)
# ---------------------------------------------------------------------------

def sigreg_loss(
    embeddings: torch.Tensor,   # (N, d) L2-normalized embeddings on unit sphere
    M: int = 1024,              # Random projection directions (LeJEPA default)
    n_freq: int = 17,           # Epps-Pulley quadrature points (LeJEPA default)
    t_min: float = -5.0,        # Frequency range min (LeJEPA default)
    t_max: float = 5.0,         # Frequency range max (LeJEPA default)
    s: float = 1.0,             # Gaussian kernel bandwidth (LeJEPA default)
) -> torch.Tensor:
    """SIGReg: fully vectorized over all M directions simultaneously.

    Memory: (N, M, K) tensor. With N=512, M=1024, K=17 in fp32 ≈ 35 MB.

    LeJEPA defaults (arXiv 2511.08544 §6.1):
        M=1024, n_freq=17, t∈[−5,+5], s=1.0
    """
    N, d = embeddings.shape
    device = embeddings.device
    dtype = embeddings.dtype

    # M random unit projection directions on S^{d-1}
    v = F.normalize(torch.randn(M, d, device=device, dtype=dtype), dim=-1)  # (M, d)

    # All 1-D projections: z[n, m] = dot(embeddings[n], v[m])
    z = embeddings @ v.T   # (N, M)

    # Frequency grid
    t = torch.linspace(t_min, t_max, n_freq, device=device, dtype=dtype)  # (K,)

    # Batch CF across all M directions: angles[n, m, k] = z[n, m] * t[k]
    angles = z.unsqueeze(-1) * t   # (N, M, K)

    emp_cf_real = torch.cos(angles).mean(dim=0)   # (M, K)
    emp_cf_imag = torch.sin(angles).mean(dim=0)   # (M, K)

    gaussian_cf = torch.exp(-0.5 * t ** 2)        # (K,)
    weights = torch.exp(-t ** 2 / s ** 2)          # (K,)

    diff_real = emp_cf_real - gaussian_cf          # (M, K)
    # Epps-Pulley statistic per direction (M,), then average
    stats = (weights * (diff_real ** 2 + emp_cf_imag ** 2)).sum(dim=-1)   # (M,)
    return stats.mean()


# ---------------------------------------------------------------------------
# LeJEPA loss: squared-Euclidean alignment + SIGReg
# ---------------------------------------------------------------------------

def lejepa_loss(
    enc_q_cot: torch.Tensor,        # (B_joint, d) L2-normalized CoT question embeddings
    enc_a_code: torch.Tensor,       # (B_joint, d) L2-normalized correct code response embeddings
    all_embeddings: torch.Tensor,   # (N_pool, d)  L2-normalized pool for SIGReg
    lambda_: float = 0.05,          # SIGReg vs align mixing (LeJEPA default)
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """LeJEPA loss = (1-λ)·L_align + λ·L_SIGReg.

    L_align: squared-Euclidean distance on the unit sphere between paired views.
    L_SIGReg: Epps-Pulley test that joint embedding distribution matches N(0, I_d).

    Per LeJEPA Theorem 1: the isotropic Gaussian is the unique minimizer of the
    integrated square bias — no other regularizer (InfoNCE, VICReg) has this guarantee.
    """
    # L_align: 0.5 * ||enc_q_cot - enc_a_code||² averaged over pairs
    diff = enc_q_cot - enc_a_code       # (B_joint, d)
    align = 0.5 * (diff ** 2).sum(dim=-1).mean()

    # L_SIGReg: on the full pool (both views, all correct)
    sig = sigreg_loss(all_embeddings, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * align + lambda_ * sig

    return loss, {
        "jepa/align_loss": float(align.detach().cpu()),
        "jepa/sigreg_loss": float(sig.detach().cpu()),
        "jepa/lejepa_loss": float(loss.detach().cpu()),
        "jepa/lambda": float(lambda_),
        "jepa/n_pairs": int(enc_q_cot.shape[0]),
        "jepa/pool_size": int(all_embeddings.shape[0]),
    }


# ---------------------------------------------------------------------------
# LLM-JEPA loss: cosine-distance prediction alignment + SIGReg
# ---------------------------------------------------------------------------

def llm_jepa_loss(
    pred_text: torch.Tensor,        # (B_joint, d) L2-normalized Pred(Enc(Text)) embeddings
    enc_code: torch.Tensor,         # (B_joint, d) L2-normalized Enc(Code) embeddings
    all_embeddings: torch.Tensor,   # (N_pool, d)  L2-normalized pool for SIGReg
    lambda_: float = 0.05,          # SIGReg vs align mixing
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """LLM-JEPA prediction loss (Huang, LeCun, Balestriero — arXiv:2509.14252, eq. 2),
    combined with SIGReg for anti-collapse.

    L = (1-λ)·d(Pred(Enc(Text)), Enc(Code)) + λ·L_SIGReg

    d is cosine distance (1 - cosine similarity), per the paper's main result
    (§3.1 "The metric" — confirmed best vs. ℓ2-norm/MSE in the ablation, Table 3).
    Both inputs are already L2-normalized so cosine similarity reduces to a dot
    product. ``pred_text`` is Pred(Enc(Text)): with k=0 tied-weight predictor
    tokens this is just Enc(Text) (Pred(x) = x per the paper); with k>0 it is the
    embedding of the last appended predictor token (see worker._extract_embeddings).

    The paper relies on a joint cross-entropy/NTP term to prevent embedding
    collapse. Since that term is intentionally not added here (the existing GRPO
    objective already covers generative capability), SIGReg (LeJEPA,
    arXiv:2511.08544) is reused as the anti-collapse regularizer, exactly as in
    `lejepa_loss`.
    """
    # L_align: 1 - cosine_similarity(pred_text, enc_code), averaged over pairs
    cos_sim = (pred_text * enc_code).sum(dim=-1)        # (B_joint,)
    align = (1.0 - cos_sim).mean()

    # L_SIGReg: on the full pool (both views, all correct)
    sig = sigreg_loss(all_embeddings, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * align + lambda_ * sig

    return loss, {
        "jepa/llm_jepa_align_loss": float(align.detach().cpu()),
        "jepa/llm_jepa_cos_sim_mean": float(cos_sim.detach().mean().cpu()),
        "jepa/llm_jepa_sigreg_loss": float(sig.detach().cpu()),
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/llm_jepa_lambda": float(lambda_),
        "jepa/n_pairs": int(pred_text.shape[0]),
        "jepa/pool_size": int(all_embeddings.shape[0]),
    }


# ---------------------------------------------------------------------------
# LLM-JEPA hard-negative triplet loss: LLM-JEPA align + triplet hinge + SIGReg
# ---------------------------------------------------------------------------

def llm_jepa_triplet_loss(
    pred_text: torch.Tensor,          # (B, d) p^c = Pred(Enc(CoT)), L2-normalized
    enc_code_correct: torch.Tensor,   # (B, d) e^c = Enc(Code_correct), L2-normalized
    enc_code_wrong: torch.Tensor,     # (T, d) e^w = Enc(Code_wrong), L2-normalized, T <= B
    all_pool: torch.Tensor,           # (2B, d) [p^c, e^c] pool for SIGReg; e^w excluded
    margin: float = 0.1,
    w_tri: float = 0.3,
    lambda_: float = 0.05,
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """LLM-JEPA prediction loss + hard-negative triplet term + SIGReg.

    L = (1-λ)·(L_align + w_tri·L_tri) + λ·L_SIGReg

    L_align (over all B pairs): cosine distance, identical to ``llm_jepa_loss``.

    L_tri (over the T <= B triplet-eligible prompts):
        (1/T) * Σ max(0, margin - <p^c_i, e^c_i> + <p^c_i, e^w_i>)
    ``enc_code_wrong``'s rows are assumed to be a prefix-aligned subset: row i
    of ``enc_code_wrong`` corresponds to row i of ``pred_text``/
    ``enc_code_correct`` (the caller/data-pipeline is responsible for this
    ordering invariant). When T == 0 (no triplet-eligible prompts in the
    batch), L_tri is set to exactly 0 — never 0/0.

    ``enc_code_wrong`` is always stop-gradiented (detached) before use: this
    kills gradient into the wrong-code encoder (the "trash-pole" failure
    mode) while still letting gradient flow into ``p^c`` (repulsion, via the
    +<p^c, e^w> term) and ``e^c`` (unaffected, via L_align). This is a mode
    invariant, not a configurable knob.

    SIGReg is computed on ``all_pool`` = [p^c, e^c] only; e^w is deliberately
    excluded since it is displaced by the triplet term and would change what
    "isotropic" means for the target distribution.
    """
    B = pred_text.shape[0]
    T = enc_code_wrong.shape[0]
    device, dtype = pred_text.device, pred_text.dtype

    cos_sim = (pred_text * enc_code_correct).sum(dim=-1)   # (B,)
    align = (1.0 - cos_sim).mean()

    if T > 0:
        pos_scores_tri = cos_sim[:T]
        neg_scores_tri = (pred_text[:T] * enc_code_wrong.detach()).sum(dim=-1)
        tri = F.relu(margin - pos_scores_tri + neg_scores_tri)
        align_tri = tri.mean()
    else:
        align_tri = torch.zeros((), device=device, dtype=dtype)

    sig = sigreg_loss(all_pool, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * (align + w_tri * align_tri) + lambda_ * sig

    metrics = {
        "jepa/llm_jepa_align_loss": float(align.detach().cpu()),
        "jepa/llm_jepa_triplet_loss": float(align_tri.detach().cpu()) if T > 0 else 0.0,
        "jepa/llm_jepa_sigreg_loss": float(sig.detach().cpu()),
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),
        "jepa/llm_jepa_lambda": float(lambda_),
        "jepa/n_pairs": int(B),
        "jepa/n_triplets": int(T),
        "jepa/triplet_frac": float(T / B) if B > 0 else 0.0,
        "jepa/pool_size": int(all_pool.shape[0]),
    }
    if T > 0:
        with torch.no_grad():
            violated = (tri > 0).float().mean()
        metrics["jepa/triplet_pos_score_mean"] = float(pos_scores_tri.detach().mean().cpu())
        metrics["jepa/triplet_neg_score_mean"] = float(neg_scores_tri.detach().mean().cpu())
        metrics["jepa/llm_jepa_triplet_violated_frac"] = float(violated.cpu())
        metrics["jepa/llm_jepa_triplet_margin"] = float(
            (pos_scores_tri - neg_scores_tri).detach().mean().cpu()
        )
        # unbiased=False: population variance, well-defined even for T == 1
        # (the unbiased N-1 estimator is NaN there).
        metrics["jepa/hard_neg_ew_variance"] = float(
            enc_code_wrong.detach().var(dim=0, unbiased=False).mean().cpu()
        )
    else:
        metrics["jepa/triplet_pos_score_mean"] = 0.0
        metrics["jepa/triplet_neg_score_mean"] = 0.0
        metrics["jepa/llm_jepa_triplet_violated_frac"] = 0.0
        metrics["jepa/llm_jepa_triplet_margin"] = 0.0
        metrics["jepa/hard_neg_ew_variance"] = 0.0

    return loss, metrics
