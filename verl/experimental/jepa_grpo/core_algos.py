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

    # SIGReg (Epps-Pulley) tests the embedding distribution against N(0, I_d) via
    # random 1-D projections, and that test only has dynamic range when the
    # projected coordinates are O(1)-variance. Callers pass L2-normalized
    # (unit-sphere) embeddings — required for the cosine alignment term — but the
    # projection of a unit-norm vector onto a random unit direction has variance
    # ~1/d (std ~0.026 at d=1536). The empirical characteristic function then sits
    # pinned at ~1 across t∈[t_min,t_max] for *every* arrangement on the sphere, so
    # the statistic degenerates into a near-constant scale mismatch that is blind to
    # the actual anisotropy/collapse it is supposed to penalize.
    #
    # Fix: rescale by the global radius only — divide by rms_norm/sqrt(d), where
    # rms_norm = sqrt(mean_i ||x_i||²). For isotropic unit-sphere data this maps the
    # mean projected variance to 1 (E_v[(x·v)²] = ||x||²/d), so projections look like
    # N(0,1) and the loss is low; a collapsed cone keeps its near-constant projection
    # along most directions (variance ≪ 1, nonzero mean), so the CF stays pinned at 1
    # and mismatches the Gaussian — high loss, real gradient. Crucially we do NOT
    # center and do NOT per-dim standardize: both would whiten away the rank/anisotropy
    # signal (subtracting the mean of a tight cone leaves only its ~isotropic jitter,
    # which then looks exactly like a well-spread set). Scaling is differentiable.
    rms_norm = embeddings.pow(2).sum(dim=-1).mean().sqrt()
    embeddings = embeddings * (d ** 0.5) / (rms_norm + 1e-6)

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

def llm_jepa_separation_loss(
    pred_text: torch.Tensor,          # (B, d) p^c = Pred(Enc(CoT)), L2-normalized
    enc_code_correct: torch.Tensor,   # (B, d) e^c = Enc(correct code rollout), L2-normalized
    enc_code_wrong: torch.Tensor,     # (T, d) e^w = Enc(clean-wrong code rollout: wrong CoT + wrong code), L2-normalized, T <= B
    all_pool: torch.Tensor,           # (2B+T, d) [p^c, e^c, e^w] pool for SIGReg; e^w INCLUDED
    sep_margin: float = 0.1,
    sep_w: float = 1.0,
    lambda_: float = 0.05,
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """LLM-JEPA prediction loss + correctness-separation term + SIGReg.

    L = (1-λ)·(L_align + sep_w·L_sep) + λ·L_SIGReg

    This is a predictor-space triplet: the SAME predictor output
    p^c = Pred(Enc(correct CoT)) is pulled toward the frozen correct-code
    target e^c (align) and pushed away from the frozen wrong-code target e^w
    (separation). Both code poles are stop-gradiented, so the predictor/CoT
    side is the only mover in both terms.

    L_align = (1/B) Σ (1 - <p^c_i, sg(e^c_i)>)
    L_sep   = (1/T) Σ relu(sep_margin - (1 - <p^c_i, sg(e^w_i)>)).

    L_sep is active only while p^c sits within sep_margin (cosine distance) of
    the wrong-code target; it is a bounded hinge (no softmax), so once p^c is
    pushed past the margin the gradient is zero — no runaway repulsion.

    Design notes:
      * The negative signal is the unweighted L_sep (``sep_w`` defaults to 1.0),
        not a down-weighted triplet term.
      * Stop-gradient on BOTH code TARGETS: e^c is detached in align and e^w is
        detached in the separation hinge. The predictor/CoT side p^c is the
        only mover in both terms (standard JEPA predictor stopgrad on both
        poles).
      * e^w is INCLUDED in ``all_pool`` for SIGReg, so it cannot collapse to a
        low-variance pole; each wrong code keeps its own location and the
        separation is per-prompt rather than a global shift.
    """
    B = pred_text.shape[0]
    T = enc_code_wrong.shape[0]
    device, dtype = pred_text.device, pred_text.dtype

    cos_sim = (pred_text * enc_code_correct.detach()).sum(dim=-1)   # (B,)
    align = (1.0 - cos_sim).mean()

    if T > 0:
        # <p^c, e^w>: same predictor output as align (p^c = Pred(Enc(correct CoT)))
        # is pushed AWAY from the frozen wrong-code target e^w. Both code poles
        # (e^c in align, e^w here) are stop-gradiented, so the predictor/CoT side
        # is the only mover in BOTH terms -- a predictor-space triplet.
        cw_sim = (pred_text[:T] * enc_code_wrong.detach()).sum(dim=-1)   # (T,)
        sep = F.relu(sep_margin - (1.0 - cw_sim))   # active when (1 - <p^c,e^w>) < sep_margin
        sep_loss = sep.mean()
    else:
        sep_loss = torch.zeros((), device=device, dtype=dtype)

    sig = sigreg_loss(all_pool, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * (align + sep_w * sep_loss) + lambda_ * sig

    metrics = {
        "jepa/llm_jepa_align_loss": float(align.detach().cpu()),
        "jepa/separation_loss": float(sep_loss.detach().cpu()) if T > 0 else 0.0,
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
            cw = cw_sim.detach()   # <p^c, e^w>
            neg_scores = cw
        # sep_gap = 1 - <p^c,e^w>; should RISE toward sep_margin as p^c is pushed off e^w.
        metrics["jepa/sep_gap_mean"] = float((1.0 - cw).mean().cpu())
        metrics["jepa/sep_cw_sim_mean"] = float(cw.mean().cpu())
        metrics["jepa/sep_active_frac"] = float((sep > 0).float().mean().cpu())
        # comparability with triplet runs:
        metrics["jepa/triplet_pos_score_mean"] = float(cos_sim[:T].detach().mean().cpu())
        metrics["jepa/triplet_neg_score_mean"] = float(neg_scores.mean().cpu())
        metrics["jepa/hard_neg_ew_variance"] = float(
            enc_code_wrong.detach().var(dim=0, unbiased=False).mean().cpu()
        )
    else:
        metrics["jepa/sep_gap_mean"] = 0.0
        metrics["jepa/sep_cw_sim_mean"] = 0.0
        metrics["jepa/sep_active_frac"] = 0.0
        metrics["jepa/triplet_pos_score_mean"] = 0.0
        metrics["jepa/triplet_neg_score_mean"] = 0.0
        metrics["jepa/hard_neg_ew_variance"] = 0.0

    return loss, metrics


# ---------------------------------------------------------------------------
# CLReg (v3) loss: matched-pair contrastive separation + SIGReg
# ---------------------------------------------------------------------------

def llm_jepa_clreg_loss(
    pred_text: torch.Tensor,          # (A, d) p_i = Pred(Enc(correct CoT_i)), L2-normalized, A anchors
    enc_code_correct: torch.Tensor,   # (A, d) e^c_i = Enc(correct Code) paired with anchor i, L2-normalized
    enc_code_wrong: torch.Tensor,     # (W, d) e^w_i = Enc(wrong CoT⊕Code joint) MATCHED to the W anchors with a wrong
    wrong_mask: torch.Tensor,         # (A,) bool — True for anchors that have a matched wrong (sum == W)
    all_pool: torch.Tensor,           # (2A+W, d) [p^c, e^c, e^w] pool for SIGReg (option a)
    tau: float = 0.5,
    mode: str = "dpo",
    sep_w: float = 1.0,
    lambda_: float = 0.5,
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CLReg separation loss (jepa_separation_loss.md), LLM-JEPA style.

    L = (1-λ)·(L_align + sep_w·L_s) + λ·L_SIGReg

    L_align = (1/A) Σ_i (1 - <p_i, e^c_i>)

    L_s uses MATCHED 1:1 PAIRS (not a per-group cross product): each correct
    anchor p_i is paired with a single wrong joint embedding e^w_i drawn (cycled)
    from the same GRPO group at batch-build time. ``enc_code_wrong`` holds those
    W matched negatives, aligned to the W anchors selected by ``wrong_mask`` (in
    order). With s(·,·) = cosine similarity on L2-normalized vectors, over the W
    anchors that have a matched wrong:

      mode="dpo":
        L_s = mean_i  -2·τ·logσ( (s(p_i, e^c_i) - s(p_i, e^w_i)) / τ )
      mode="info":   (single negative -> binary softmax)
        L_s = mean_i  -log[ exp(s/τ_pos) / (exp(s/τ_pos) + exp(s/τ_neg)) ]
            = mean_i  softplus( (s(p_i, e^w_i) - s(p_i, e^c_i)) / τ )

    Anchors with no matched wrong contribute 0 (when W == 0, L_s == 0).

    NO stop-gradient (LLM-JEPA, arXiv:2509.14252): Pred, Enc(Text) and Enc(Code)
    are the same LLM encoder in a single forward pass; there is no frozen target
    network, so e^c and e^w receive gradient just like p_i does (the paper never
    detaches). SIGReg pool is code-only/decoupled (option a in the doc): it reuses
    the already-computed [p^c, e^c, e^w] embeddings, not a second representation.
    """
    A = pred_text.shape[0]
    W = enc_code_wrong.shape[0]
    device, dtype = pred_text.device, pred_text.dtype

    # L_align: cosine distance between each anchor and its positive code. No
    # stop-gradient — gradient flows through e^c too (LLM-JEPA shared encoder).
    pos_sim = (pred_text * enc_code_correct).sum(dim=-1)   # (A,)
    align = (1.0 - pos_sim).mean()

    if W > 0:
        wrong_mask = wrong_mask.bool()
        pos_w = pos_sim[wrong_mask]                                  # (W,) s(p_i, e^c_i)
        # s(p_i, e^w_i) for each matched pair. No stop-gradient — e^w receives
        # gradient too (same shared encoder; the negative is pushed away as p is).
        neg_sim = (pred_text[wrong_mask] * enc_code_wrong).sum(dim=-1)   # (W,)
        margin = (pos_w - neg_sim) / tau                            # (W,)
        if mode == "info":
            # Binary softmax with a single negative == softplus(-margin).
            per_pair = F.softplus(-margin)
        else:
            # DPO: log-sigmoid of the (pos - neg) margin.
            per_pair = -2.0 * tau * F.logsigmoid(margin)
        sep_loss = per_pair.mean()
        active = margin < 0                                          # positive not yet dominant
    else:
        sep_loss = torch.zeros((), device=device, dtype=dtype)
        active = torch.zeros((0,), dtype=torch.bool, device=device)

    sig = sigreg_loss(all_pool, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * (align + sep_w * sep_loss) + lambda_ * sig

    metrics = {
        "jepa/clreg_align_loss": float(align.detach().cpu()),
        "jepa/separation_loss": float(sep_loss.detach().cpu()) if W > 0 else 0.0,
        "jepa/clreg_sigreg_loss": float(sig.detach().cpu()),
        "jepa/clreg_loss": float(loss.detach().cpu()),
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),   # alias read back by worker
        "jepa/clreg_lambda": float(lambda_),
        "jepa/clreg_tau": float(tau),
        "jepa/n_pairs": int(A),
        "jepa/n_wrong": int(W),
        "jepa/pool_size": int(all_pool.shape[0]),
    }
    metrics["jepa/triplet_pos_score_mean"] = float(pos_sim.detach().mean().cpu()) if A > 0 else 0.0
    if W > 0:
        metrics["jepa/sep_active_frac"] = float(active.float().mean().cpu())
        metrics["jepa/triplet_neg_score_mean"] = float(neg_sim.detach().mean().cpu())
        metrics["jepa/sep_gap_mean"] = float((pos_w.detach() - neg_sim.detach()).mean().cpu())
        metrics["jepa/hard_neg_ew_variance"] = float(
            enc_code_wrong.detach().var(dim=0, unbiased=False).mean().cpu()
        )
    else:
        metrics["jepa/sep_active_frac"] = 0.0
        metrics["jepa/triplet_neg_score_mean"] = 0.0
        metrics["jepa/sep_gap_mean"] = 0.0
        metrics["jepa/hard_neg_ew_variance"] = 0.0

    return loss, metrics


# ---------------------------------------------------------------------------
# TCR (Teacher-Correct Representation) loss: align-to-teacher + SIGReg
# ---------------------------------------------------------------------------

def llm_jepa_tcr_loss(
    pred_text: torch.Tensor,          # (A, d) p_i = Pred(Enc(student CoT_i)), L2-normalized
    teacher_target: torch.Tensor,     # (A, d) z_T+ paired to anchor i (precomputed, constant), L2-normalized
    all_pool: torch.Tensor,           # (A, d) SIGReg pool — STUDENT preds only (the only thing that moves)
    group_id: torch.Tensor | None = None,    # (A,) long: per-prompt id (0..G-1) for each anchor
    is_correct: torch.Tensor | None = None,  # (A,) bool: True for rew>0 anchors
    lambda_: float = 0.5,
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Teacher-Correct Representation (TCR) alignment loss.

    L = (1-λ)·L_align + λ·L_SIGReg,  ℓ_i = 1 - <p_i, sg(z_T+_i)>

    Each anchor p_i = Pred(Enc([x, y_S,i, [PRED]xk])) is pulled toward a precomputed
    teacher-correct target z_T+_i in the SAME 1536-d student space (teacher text encoded
    offline by a frozen student-size reference — no projector, no cross-dim cosine).
    Correct AND wrong rollouts may be anchors (jepa.jepa_anchor_set): for wrong anchors the
    [PRED] token learns to predict the teacher-correct latent from a failed trajectory's
    context (latent correction), without forcing the wrong response's own hidden state to
    look correct.

    ANCHOR AGGREGATION (when group_id/is_correct are given) is reward-stratified and
    PROMPT-averaged, so raw anchor counts never implicitly weight the loss:

        L(x) = ½·mean_{i∈C_x} ℓ_i + ½·mean_{i∈W_x} ℓ_i   if both correct & wrong present
             = mean over whichever conditional set is non-empty otherwise
             = 0                                          if the prompt has no anchors
        L_align = (1/G) Σ_x L(x)                          # mean over prompts, not anchors

    The fixed ½/½ split defines a reward-stratified anchor distribution; it is NOT a tunable
    per-class loss weight. With group_id/is_correct=None this falls back to the flat
    (1/A) Σ ℓ_i used before the wrong-anchor extension.

    NOTE on jepa_anchor_set="correct": this mode now uses the prompt-averaged correct-anchor
    loss (mean over prompts of each prompt's mean correct ℓ), which can differ NUMERICALLY
    from the old flat (1/A) Σ ℓ_i whenever prompts have unequal numbers of correct anchors.
    This is intentional: prompt-level averaging stops prompts that happen to sample many
    correct rollouts from dominating the representation objective. The flat mean survives only
    as the group_id/is_correct=None back-compat path (never hit in normal training, since the
    batch builder always supplies these tensors).

    STOP-GRADIENT on the teacher target (cached constant; ``.detach()`` is explicit). No
    separation term — SIGReg over the student preds alone supplies anti-collapse, keeping the
    global-radius rescale that is load-bearing on unit-sphere inputs (see sigreg_loss).
    """
    A = pred_text.shape[0]
    device, dtype = pred_text.device, pred_text.dtype

    # Defensive empty-batch guard: SIGReg over an empty pool is undefined (NaN), so
    # short-circuit to a finite zero loss with zeroed metrics. Normal training never
    # reaches here (the batch builder + worker both skip below min_valid_pairs).
    if A == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        metrics = {
            "jepa/tcr_align_loss": 0.0, "jepa/tcr_sigreg_loss": 0.0, "jepa/tcr_loss": 0.0,
            "jepa/llm_jepa_loss": 0.0, "jepa/tcr_lambda": float(lambda_), "jepa/n_anchors": 0,
            "jepa/tcr_pos_score_mean": 0.0, "jepa/pool_size": int(all_pool.shape[0]),
            "jepa/num_anchors_total": 0, "jepa/num_anchors_correct": 0, "jepa/num_anchors_wrong": 0,
            "jepa/loss_align": 0.0, "jepa/loss_correct_monitor": 0.0, "jepa/loss_wrong_monitor": 0.0,
            "jepa/cos_total": 0.0, "jepa/cos_correct": 0.0, "jepa/cos_wrong": 0.0,
        }
        return zero, metrics

    # ℓ_i: cosine distance to the stop-gradiented teacher target.
    pos_sim = (pred_text * teacher_target.detach()).sum(dim=-1)   # (A,)
    ell = 1.0 - pos_sim                                           # (A,)

    if group_id is None or is_correct is None:
        # Back-compat flat mean over all anchors.
        align = ell.mean()
    else:
        # Reward-stratified, prompt-averaged aggregation.
        gid = group_id.to(device=device, dtype=torch.long)
        corr = is_correct.to(device=device, dtype=torch.bool)
        G = int(gid.max().item()) + 1 if A > 0 else 0
        ones = torch.ones(A, device=device, dtype=dtype)

        def _group_mean(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # Per-group mean of ℓ over the masked subset; returns (mean_g, present_g).
            m = mask.to(dtype)
            ssum = torch.zeros(G, device=device, dtype=dtype).index_add_(0, gid, ell * m)
            cnt = torch.zeros(G, device=device, dtype=dtype).index_add_(0, gid, ones * m)
            present = cnt > 0
            mean_g = ssum / cnt.clamp(min=1.0)
            return mean_g, present

        cmean, cpresent = _group_mean(corr)
        wmean, wpresent = _group_mean(~corr)
        both = cpresent & wpresent
        # ½/½ where both classes exist, else the single available conditional mean.
        per_prompt = torch.where(both, 0.5 * cmean + 0.5 * wmean,
                                 torch.where(cpresent, cmean, wmean))
        any_anchor = cpresent | wpresent
        n_prompts = any_anchor.sum().clamp(min=1)
        align = (per_prompt * any_anchor.to(dtype)).sum() / n_prompts

    sig = sigreg_loss(all_pool, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)

    loss = (1.0 - lambda_) * align + lambda_ * sig

    # ---- monitor-only stats (do NOT affect the optimized loss) ----
    pos_sim_d = pos_sim.detach()
    ell_d = ell.detach()
    if is_correct is not None and A > 0:
        corr = is_correct.to(device=device, dtype=torch.bool)
        n_correct = int(corr.sum().item())
        n_wrong = A - n_correct
        cos_correct = float(pos_sim_d[corr].mean().cpu()) if n_correct > 0 else 0.0
        cos_wrong = float(pos_sim_d[~corr].mean().cpu()) if n_wrong > 0 else 0.0
        loss_correct_monitor = float(ell_d[corr].mean().cpu()) if n_correct > 0 else 0.0
        loss_wrong_monitor = float(ell_d[~corr].mean().cpu()) if n_wrong > 0 else 0.0
    else:
        n_correct, n_wrong = A, 0
        cos_correct = float(pos_sim_d.mean().cpu()) if A > 0 else 0.0
        cos_wrong = 0.0
        loss_correct_monitor = float(ell_d.mean().cpu()) if A > 0 else 0.0
        loss_wrong_monitor = 0.0

    # Single host transfer for the repeated full-pool means (avoid one .cpu() per use).
    align_v = float(align.detach().cpu())
    pos_mean_v = float(pos_sim_d.mean().cpu()) if A > 0 else 0.0
    metrics = {
        "jepa/tcr_align_loss": align_v,
        "jepa/tcr_sigreg_loss": float(sig.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),       # full optimized JEPA-side scalar
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),  # alias read back by worker
        "jepa/tcr_lambda": float(lambda_),
        "jepa/n_anchors": int(A),
        "jepa/tcr_pos_score_mean": pos_mean_v,
        "jepa/pool_size": int(all_pool.shape[0]),
        # reward-stratified monitoring (monitor-only — NOT separately weighted in the loss)
        "jepa/num_anchors_total": int(A),
        "jepa/num_anchors_correct": int(n_correct),
        "jepa/num_anchors_wrong": int(n_wrong),
        "jepa/loss_align": align_v,            # the (prompt-averaged) alignment term only
        "jepa/loss_correct_monitor": loss_correct_monitor,
        "jepa/loss_wrong_monitor": loss_wrong_monitor,
        "jepa/cos_total": pos_mean_v,
        "jepa/cos_correct": cos_correct,
        "jepa/cos_wrong": cos_wrong,
    }
    return loss, metrics
