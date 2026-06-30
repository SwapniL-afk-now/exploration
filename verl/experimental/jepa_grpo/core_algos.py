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

def _prompt_stratified_align(
    ell: torch.Tensor,
    group_id: torch.Tensor | None,
    is_correct: torch.Tensor | None,
) -> torch.Tensor:
    """Reward-stratified, PROMPT-averaged mean of per-anchor losses ``ell``.

    Mirrors the aggregation inside ``llm_jepa_tcr_loss``: per prompt,
    ``½·mean(correct ℓ) + ½·mean(wrong ℓ)`` when both classes are present, else the
    single available conditional mean; then mean over PROMPTS (not raw anchors) so
    anchor counts never implicitly weight the loss. Falls back to a flat mean when
    ``group_id``/``is_correct`` are None. Pure function of its tensors (unit-testable).
    """
    A = ell.shape[0]
    device, dtype = ell.device, ell.dtype
    if A == 0:
        return torch.zeros((), device=device, dtype=dtype)
    if group_id is None or is_correct is None:
        return ell.mean()
    gid = group_id.to(device=device, dtype=torch.long)
    corr = is_correct.to(device=device, dtype=torch.bool)
    G = int(gid.max().item()) + 1
    ones = torch.ones(A, device=device, dtype=dtype)

    def _group_mean(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = mask.to(dtype)
        ssum = torch.zeros(G, device=device, dtype=dtype).index_add_(0, gid, ell * m)
        cnt = torch.zeros(G, device=device, dtype=dtype).index_add_(0, gid, ones * m)
        present = cnt > 0
        return ssum / cnt.clamp(min=1.0), present

    cmean, cpresent = _group_mean(corr)
    wmean, wpresent = _group_mean(~corr)
    both = cpresent & wpresent
    per_prompt = torch.where(both, 0.5 * cmean + 0.5 * wmean,
                             torch.where(cpresent, cmean, wmean))
    any_anchor = cpresent | wpresent
    n_prompts = any_anchor.sum().clamp(min=1)
    return (per_prompt * any_anchor.to(dtype)).sum() / n_prompts


def llm_jepa_tcr_dual_loss(
    pred_cot: torch.Tensor,             # (A_c, d) CoT student [PRED] reads, L2-normalized
    teacher_target_cot: torch.Tensor,  # (A_c, d) cached CoT teacher targets (constant)
    pred_code: torch.Tensor,           # (A_k, d) Code student [PRED] reads, L2-normalized
    teacher_target_code: torch.Tensor, # (A_k, d) cached Code teacher targets (constant)
    self_target: torch.Tensor,         # (A_c, d) Code_S BOUNDARY reads paired to CoT rows (raw, will be detached)
    self_mask: torch.Tensor,           # (A_c,) bool: True where a CoT row has a Code partner
    cot_group_id: torch.Tensor | None = None,
    cot_is_correct: torch.Tensor | None = None,
    code_group_id: torch.Tensor | None = None,
    code_is_correct: torch.Tensor | None = None,
    self_consist_w: float = 1.0,
    align_cot_on: bool = True,
    align_code_on: bool = True,
    self_on: bool = True,
    lambda_: float = 0.5,
    M: int = 1024,
    n_freq: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    s: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Dual-target, self-consistent TCR loss (loss_type='jepa-tcr-dual').

    L = (1-λ)·( align_cot + align_code + w·L_self ) + λ·SIGReg([pred_cot, pred_code])

      align_cot  = stratified mean of 1 - <pred_cot_i,  sg(z_cot_i)>      (CoT teacher pull)
      align_code = stratified mean of 1 - <pred_code_j, sg(z_code_j)>     (Code teacher pull)
      L_self     = mean over partnered CoT rows of 1 - <pred_cot_i, sg(b_i)>

    where b_i is the *boundary read* of the paired Code_S block (the raw last-real-token
    embedding BEFORE the [PRED] token) — a free second read of a block already in the
    forward. self_target is stop-gradiented here (only pred_cot moves toward it), the
    direction specified for the self-consistency arm; this differs from the symmetric
    crossview_partner term in llm_jepa_tcr_loss (which moves both [PRED] reads). L_self
    self-gates to 0 when no CoT row has a code partner (self_mask all-False).

    Both teacher targets are stop-gradiented constants (cached). SIGReg over the student
    preds of BOTH views supplies anti-collapse, preserving the load-bearing global-radius
    rescale on unit-sphere inputs (see sigreg_loss / project_jepa_sigreg_collapse).
    """
    A_c = pred_cot.shape[0]
    A_k = pred_code.shape[0]
    device = pred_cot.device
    dtype = pred_cot.dtype

    pool = torch.cat([pred_cot, pred_code], dim=0)
    if pool.shape[0] == 0:
        zero = torch.zeros((), device=device, dtype=dtype)
        return zero, {
            "jepa/tcr_loss": 0.0, "jepa/llm_jepa_loss": 0.0, "jepa/tcr_lambda": float(lambda_),
            "jepa/align_cot": 0.0, "jepa/align_code": 0.0, "jepa/self_consist_loss": 0.0,
            "jepa/tcr_sigreg_loss": 0.0, "jepa/n_anchors_cot": 0, "jepa/n_anchors_code": 0,
            "jepa/n_self_pairs": 0, "jepa/cos_cot": 0.0, "jepa/cos_code": 0.0, "jepa/cos_self": 0.0,
            "jepa/pool_size": 0,
        }

    # Teacher pulls (each stratified + prompt-averaged independently).
    cot_pos = (pred_cot * teacher_target_cot.detach()).sum(dim=-1) if A_c > 0 else pred_cot.new_zeros((0,))
    code_pos = (pred_code * teacher_target_code.detach()).sum(dim=-1) if A_k > 0 else pred_code.new_zeros((0,))
    align_cot = _prompt_stratified_align(1.0 - cot_pos, cot_group_id, cot_is_correct)
    align_code = _prompt_stratified_align(1.0 - code_pos, code_group_id, code_is_correct)

    # Self-consistency: pred_CoT pulled toward the stop-grad Code_S boundary read.
    self_loss = torch.zeros((), device=device, dtype=dtype)
    n_self = 0
    self_pos_mean = 0.0
    if A_c > 0 and self_mask is not None:
        m = self_mask.to(device=device, dtype=torch.bool)
        if m.any():
            self_pos = (pred_cot[m] * self_target[m].detach()).sum(dim=-1)
            self_loss = (1.0 - self_pos).mean()
            n_self = int(m.sum().item())
            self_pos_mean = float(self_pos.detach().mean().cpu())

    sig = sigreg_loss(pool, M=M, n_freq=n_freq, t_min=t_min, t_max=t_max, s=s)
    # Per-arm plateau latches: a disabled arm contributes 0 gradient (coefficient 0),
    # but its preds STILL feed the SIGReg pool above (anti-collapse pressure preserved).
    c_cot = 1.0 if align_cot_on else 0.0
    c_code = 1.0 if align_code_on else 0.0
    c_self = 1.0 if self_on else 0.0
    loss = (1.0 - lambda_) * (
        c_cot * align_cot + c_code * align_code + c_self * self_consist_w * self_loss
    ) + lambda_ * sig

    metrics = {
        "jepa/align_cot": float(align_cot.detach().cpu()),
        "jepa/align_code": float(align_code.detach().cpu()),
        "jepa/self_consist_loss": float(self_loss.detach().cpu()),
        "jepa/tcr_sigreg_loss": float(sig.detach().cpu()),
        "jepa/tcr_loss": float(loss.detach().cpu()),
        "jepa/llm_jepa_loss": float(loss.detach().cpu()),  # alias read back by worker
        "jepa/tcr_lambda": float(lambda_),
        "jepa/n_anchors_cot": int(A_c),
        "jepa/n_anchors_code": int(A_k),
        "jepa/n_self_pairs": int(n_self),
        "jepa/cos_cot": float(cot_pos.detach().mean().cpu()) if A_c > 0 else 0.0,
        "jepa/cos_code": float(code_pos.detach().mean().cpu()) if A_k > 0 else 0.0,
        "jepa/cos_self": self_pos_mean,
        "jepa/pool_size": int(pool.shape[0]),
        "jepa/arm_cot_on": float(align_cot_on),
        "jepa/arm_code_on": float(align_code_on),
        "jepa/arm_self_on": float(self_on),
    }
    return loss, metrics
