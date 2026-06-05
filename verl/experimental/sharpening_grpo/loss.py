# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SharpeningLossOutput:
    """Container for the three terms of the Sequence Sharpening loss.

    All tensors are scalar loss terms ready to be added into ``policy_loss``.
    ``metrics`` carries the diagnostic values to be logged alongside the loss.
    """

    grpo_term: torch.Tensor
    seq_term: torch.Tensor
    kl_term: torch.Tensor
    metrics: dict[str, float]


def _masked_token_mean(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid response tokens. Empty masks are guarded with a +0 nudge."""
    mask = response_mask.to(dtype=values.dtype)
    denom = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denom


def compute_sharpening_grpo_loss(
    *,
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_prob: torch.Tensor,
    group_size: int,
    clip_ratio: float,
    use_grpo_reward: bool,
) -> SharpeningLossOutput:
    """Compute the full Sequence Sharpening actor loss.

    Args:
        old_log_prob: ``(B, T)`` log-probabilities from the rollout policy
            (``pi_old``).
        log_prob: ``(B, T)`` log-probabilities from the current actor.
        advantages: ``(B, T)`` per-token advantages (GRPO is flat per sequence
            so this is replicated across the time axis). The first axis of
            ``B`` is ``num_prompts * group_size``.
        response_mask: ``(B, T)`` boolean / 0-1 mask. ``1`` for valid response
            tokens, ``0`` for prompt / padding.
        ref_log_prob: ``(B, T)`` frozen reference log-probabilities used for
            the K3 KL penalty and the entropy-rate suffix sum.
        group_size: ``G``, the number of rollouts per prompt. ``B`` must be
            divisible by ``G``.
        clip_ratio: PPO clip ``epsilon`` reused by the sequence surrogate so
            that both terms share the same trust region.
        use_grpo_reward: if ``False``, the standard GRPO term is zeroed out
            entirely. This is the verifier-free ablation toggle.

    Returns:
        ``SharpeningLossOutput`` with the three loss terms (already correctly
        signed) and a flat metrics dict. The caller is responsible for the
        final combination:

        .. code-block:: python

            total = grpo_term * ppo_loss_coef \
                  + gamma * alpha * seq_term \
                  + beta * kl_term
    """

    # All inputs share shape (B, T). Normalize mask to bool.
    if response_mask.dtype != torch.bool:
        response_mask = response_mask.to(torch.bool)
    if old_log_prob.shape != log_prob.shape or log_prob.shape != ref_log_prob.shape:
        raise ValueError(
            "old_log_prob, log_prob and ref_log_prob must share shape. "
            f"Got {tuple(old_log_prob.shape)}, {tuple(log_prob.shape)}, {tuple(ref_log_prob.shape)}."
        )

    B_full, T = log_prob.shape
    G = int(group_size)

    # ---------------------------------------------------------------- GRPO term
    # PPO-clipped surrogate on the rollout advantage. Identical to the verl
    # vanilla policy loss; kept inline so the toggle can zero it cleanly.
    log_ratio_grpo = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio_grpo = torch.exp(log_ratio_grpo)
    surr1 = -advantages * ratio_grpo
    surr2 = -advantages * torch.clamp(ratio_grpo, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss = torch.maximum(surr1, surr2)
    grpo_term = _masked_token_mean(pg_loss, response_mask)
    if not bool(use_grpo_reward):
        grpo_term = grpo_term.detach() * 0.0  # zero with grad-disconnected for clean autograd

    # ----------------------------------------------------- Entropy-rate weight
    # w_t = (sum_{s >= t} ref_log_prob_s) / (sum_{s >= t} mask_s)
    # Computed via reverse cumulative sum so that the weight represents the
    # average reference log-probability of the **future** path. Padding tokens
    # are excluded via the masked cumsum trick (0 * log_prob = 0 -> counts as
    # zero contribution but the denominator tracks the true length).
    masked_ref = ref_log_prob * response_mask.to(dtype=ref_log_prob.dtype)
    rev_cumsum_ref = torch.flip(torch.cumsum(torch.flip(masked_ref, dims=[-1]), dim=-1), dims=[-1])
    rev_cumsum_mask = torch.flip(
        torch.cumsum(torch.flip(response_mask.to(dtype=ref_log_prob.dtype), dims=[-1]), dim=-1), dims=[-1]
    )
    w_t = rev_cumsum_ref / (rev_cumsum_mask + 1e-8)

    # ---------------------------------------------------------- GRSA normalize
    # Reshape (B, T) -> (P, G, T) and normalize across the G axis to get a
    # per-prompt, per-timestep z-score.  When B is not divisible by G (dynamic
    # batching), the trailing sequences use the raw entropy-rate weight.
    B = B_full
    P_complete = B // G
    B_grsa = P_complete * G

    if B_grsa > 0:
        w_grsa = w_t[:B_grsa].view(P_complete, G, T)
        mu_group = w_grsa.mean(dim=1, keepdim=True)
        sigma_group = w_grsa.std(dim=1, keepdim=True)
        w_bar_grsa = ((w_grsa - mu_group) / (sigma_group + 1e-8)).view(B_grsa, T)
    else:
        # Fewer sequences than group_size — no complete groups for GRSA.
        # Use the raw entropy-rate weight (no z-scoring).
        w_bar_grsa = torch.empty(0, T, dtype=w_t.dtype, device=w_t.device)
        mu_group = torch.empty(0, 1, dtype=w_t.dtype, device=w_t.device)
        sigma_group = torch.empty(0, 1, dtype=w_t.dtype, device=w_t.device)

    # Remainder sequences that don't form a complete group get raw w_t.
    if B > B_grsa:
        w_bar_remainder = w_t[B_grsa:]
    else:
        w_bar_remainder = torch.empty(0, T, dtype=w_t.dtype, device=w_t.device)

    # Concatenate GRSA-normalized and raw remainder weights.
    w_bar = torch.cat([w_bar_grsa, w_bar_remainder], dim=0) * response_mask.to(dtype=w_t.dtype)

    # ------------------------------------------------ Sequence sharpening term
    # Same PPO-style clip as the GRPO term, but multiplied by the
    # GRSA-normalized entropy-rate weight w_bar. Sign convention follows the
    # algorithm spec: ``L_seq = min(ratio*w_bar, clip(ratio)*w_bar)`` and we
    # minimize ``-L_seq``.
    log_ratio_seq = (log_prob - old_log_prob).clamp(min=-20.0, max=20.0)
    ratio_seq = torch.exp(log_ratio_seq)
    surr1_seq = ratio_seq * w_bar
    surr2_seq = torch.clamp(ratio_seq, 1.0 - clip_ratio, 1.0 + clip_ratio) * w_bar
    L_seq = torch.minimum(surr1_seq, surr2_seq)
    seq_term = -_masked_token_mean(L_seq, response_mask)

    # -------------------------------------------------------------- K3 KL term
    # Unbiased K3 (a.k.a. K1) estimator: kl_k3 = exp(r) - 1 - r, where
    # r = pi_theta(y) / pi_ref(y). Always non-negative when log_ratio is
    # defined; clamping protects against fp16 overflow.
    log_ratio_kl = (log_prob - ref_log_prob).clamp(min=-20.0, max=20.0)
    r_kl = torch.exp(log_ratio_kl)
    kl_k3 = r_kl - 1.0 - log_ratio_kl
    kl_term = _masked_token_mean(kl_k3, response_mask)

    # ----------------------------------------------------------------- Metrics
    mu_scalar = float(mu_group.detach().mean().cpu()) if mu_group.numel() > 0 else 0.0
    sigma_scalar = float(sigma_group.detach().mean().cpu()) if sigma_group.numel() > 0 else 0.0
    metrics: dict[str, float] = {
        "sharpen/grpo_term": float(grpo_term.detach().cpu()),
        "sharpen/seq_term": float(seq_term.detach().cpu()),
        "sharpen/kl_term": float(kl_term.detach().cpu()),
        "sharpen/mu_group_mean": mu_scalar,
        "sharpen/sigma_group_mean": sigma_scalar,
        "sharpen/mu_group_abs_mean": float(mu_group.detach().abs().mean().cpu()) if mu_group.numel() > 0 else 0.0,
        "sharpen/w_bar_abs_mean": float(w_bar.detach().abs().mean().cpu()),
        "sharpen/entropy_rate_mean": float(w_t.detach().mean().cpu()),
        "sharpen/use_grpo_reward": float(bool(use_grpo_reward)),
        "sharpen/group_size": float(G),
        "sharpen/num_prompts": float(P_complete),
    }

    return SharpeningLossOutput(grpo_term=grpo_term, seq_term=seq_term, kl_term=kl_term, metrics=metrics)
