# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class TAFRLossOutput:
    loss: torch.Tensor
    metrics: dict[str, float]


def response_length_normalized_mean(values: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    mask = response_mask.to(dtype=values.dtype)
    lengths = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / lengths


def replay_gate(group_reward_mean: torch.Tensor) -> torch.Tensor:
    return 1.0 - group_reward_mean


def _group_mean(values: torch.Tensor, group_ids: Optional[Iterable[object]]) -> torch.Tensor:
    if values.numel() == 0:
        return values.sum()
    if group_ids is None:
        return values.mean()
    ids = list(group_ids)
    if len(ids) != values.shape[0]:
        return values.mean()
    grouped = []
    for uid in dict.fromkeys(ids):
        idx = [i for i, item in enumerate(ids) if item == uid]
        if idx:
            grouped.append(values[torch.as_tensor(idx, device=values.device, dtype=torch.long)].mean())
    if not grouped:
        return values.mean()
    return torch.stack(grouped).mean()


def compute_tafr_grpo_auxiliary_loss(
    *,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    group_reward_mean: torch.Tensor,
    beta: float,
    variant: str = "full",
    anchor_log_prob: Optional[torch.Tensor] = None,
    replay_log_prob: Optional[torch.Tensor] = None,
    group_ids: Optional[Iterable[object]] = None,
) -> TAFRLossOutput:
    """Compute TAFR-GRPO KL terms on the pi_old rollout samples.

    All rows are actor (pi_old) rollout samples — no separate replay rows.
    Both anchor and replay log-probs are evaluated on the same y_i ~ pi_old.

    Anchor KL:   D_KL(pi_theta || pi_anchor)  = log pi_theta(y_i) - log pi_anchor(y_i)
    Replay KL:   D_KL(pi_replay || pi_theta)  = log pi_replay(y_i) - log pi_theta(y_i)

    Actor loss adds:
        + beta * anchor_kl   (stay close to stable anchor)
        - beta * (1 - r_bar_x) * replay_kl   (move away from failure modes)

    Only log_prob carries actor gradients; anchor_log_prob and replay_log_prob
    must be detached (frozen models).
    """

    if variant not in {"full", "anchor_only", "replay_only"}:
        raise ValueError("variant must be 'full', 'anchor_only', or 'replay_only'.")

    group_reward_mean = group_reward_mean.to(device=log_prob.device, dtype=log_prob.dtype).clamp(0.0, 1.0)
    gate = replay_gate(group_reward_mean)

    zero = log_prob.sum() * 0.0
    anchor_loss = zero
    replay_loss = zero

    anchor_kl_metric = 0.0
    replay_kl_metric = 0.0
    actor_lp_metric = 0.0
    anchor_lp_metric = 0.0
    replay_lp_metric = 0.0
    actor_lp_on_replay_metric = 0.0

    if variant in {"full", "anchor_only"} and anchor_log_prob is not None:
        frozen_anchor = anchor_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach()
        # k3 estimator: (r-1) - log(r), lower variance than k1 log(r)
        log_ratio_anchor = (log_prob - frozen_anchor).clamp(min=-20, max=20)
        r_anchor = torch.exp(log_ratio_anchor)
        anchor_seq_kl = response_length_normalized_mean(((r_anchor - 1) - log_ratio_anchor).clamp(min=-10, max=10), response_mask)
        anchor_loss = _group_mean(anchor_seq_kl, group_ids)
        anchor_kl_metric = float(anchor_seq_kl.detach().mean().cpu())
        actor_lp_metric = float(response_length_normalized_mean(log_prob.detach(), response_mask).mean().cpu())
        anchor_lp_metric = float(response_length_normalized_mean(frozen_anchor, response_mask).mean().cpu())

    if variant in {"full", "replay_only"} and replay_log_prob is not None:
        frozen_replay = replay_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach()
        # k3 estimator for D_KL(pi_replay || pi_theta): log ratio is frozen_replay - log_prob
        log_ratio_replay = (frozen_replay - log_prob).clamp(min=-20, max=20)
        r_replay = torch.exp(log_ratio_replay)
        replay_seq_kl = response_length_normalized_mean(((r_replay - 1) - log_ratio_replay).clamp(min=-10, max=10), response_mask)
        gated_replay_seq_kl = gate * replay_seq_kl
        replay_loss = _group_mean(gated_replay_seq_kl, group_ids)
        replay_kl_metric = float(replay_seq_kl.detach().mean().cpu())
        replay_lp_metric = float(response_length_normalized_mean(frozen_replay, response_mask).mean().cpu())
        actor_lp_on_replay_metric = float(response_length_normalized_mean(log_prob.detach(), response_mask).mean().cpu())

    loss = float(beta) * anchor_loss - float(beta) * replay_loss
    group_reward_mean_d = group_reward_mean.detach()

    metrics = {
        "tafr_grpo/loss_total": float(loss.detach().cpu()),
        "tafr_grpo/kl_anchor": anchor_kl_metric,
        "tafr_grpo/kl_replay": replay_kl_metric,
        "tafr_grpo/replay_gate_mean": float(gate.detach().mean().cpu()),
        "tafr_grpo/mean_group_reward": float(group_reward_mean_d.mean().cpu()),
        "tafr_grpo/fraction_all_wrong_groups": float((group_reward_mean_d == 0).float().mean().cpu()),
        "tafr_grpo/fraction_all_correct_groups": float((group_reward_mean_d == 1).float().mean().cpu()),
        "tafr_grpo/fraction_mixed_groups": float(
            ((group_reward_mean_d > 0) & (group_reward_mean_d < 1)).float().mean().cpu()
        ),
        "tafr_grpo/beta": float(beta),
    }
    if variant in {"full", "anchor_only"}:
        metrics["tafr_grpo/actor_logprob"] = actor_lp_metric
        metrics["tafr_grpo/anchor_logprob"] = anchor_lp_metric
    if variant in {"full", "replay_only"}:
        metrics["tafr_grpo/replay_logprob"] = replay_lp_metric
        metrics["tafr_grpo/actor_logprob_on_replay_samples"] = actor_lp_on_replay_metric
    return TAFRLossOutput(loss=loss, metrics=metrics)
