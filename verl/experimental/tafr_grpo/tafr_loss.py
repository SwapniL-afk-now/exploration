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
    is_replay: torch.Tensor,
    group_reward_mean: torch.Tensor,
    beta: float,
    variant: str = "full",
    anchor_log_prob: Optional[torch.Tensor] = None,
    replay_log_prob: Optional[torch.Tensor] = None,
    group_ids: Optional[Iterable[object]] = None,
) -> TAFRLossOutput:
    """Compute TAFR-GRPO KL terms on sampled response tokens.

    Anchor KL uses actor-generated samples:
        D_KL(pi_theta || pi_anchor)

    Replay KL uses replay-generated samples:
        D_KL(pi_replay || pi_theta)

    The actor loss adds + beta * anchor_kl and - beta * (1-r_bar_x) * replay_kl.
    Frozen log-probs must be detached before this function is called, and only
    ``log_prob`` should carry actor gradients.
    """

    if variant not in {"full", "anchor_only", "replay_only"}:
        raise ValueError("variant must be 'full', 'anchor_only', or 'replay_only'.")

    is_replay = is_replay.to(device=log_prob.device).bool()
    is_actor = ~is_replay
    group_reward_mean = group_reward_mean.to(device=log_prob.device, dtype=log_prob.dtype).clamp(0.0, 1.0)
    gate = replay_gate(group_reward_mean)

    zero = log_prob.sum() * 0.0
    anchor_loss = zero
    replay_loss = zero

    anchor_kl_metric = 0.0
    replay_kl_metric = 0.0
    actor_lp_actor_metric = 0.0
    anchor_lp_actor_metric = 0.0
    replay_lp_replay_metric = 0.0
    actor_lp_replay_metric = 0.0

    if variant in {"full", "anchor_only"} and anchor_log_prob is not None and bool(is_actor.any()):
        actor_idx = is_actor.nonzero(as_tuple=True)[0]
        actor_log_prob = log_prob.index_select(0, actor_idx)
        actor_anchor_log_prob = anchor_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach().index_select(
            0, actor_idx
        )
        actor_mask = response_mask.index_select(0, actor_idx)
        actor_group_ids = [list(group_ids)[i] for i in actor_idx.tolist()] if group_ids is not None else None

        anchor_seq_kl = response_length_normalized_mean(actor_log_prob - actor_anchor_log_prob, actor_mask)
        anchor_loss = _group_mean(anchor_seq_kl, actor_group_ids)

        actor_lp_actor = response_length_normalized_mean(actor_log_prob.detach(), actor_mask)
        anchor_lp_actor = response_length_normalized_mean(actor_anchor_log_prob, actor_mask)
        anchor_kl_metric = float(anchor_seq_kl.detach().mean().cpu())
        actor_lp_actor_metric = float(actor_lp_actor.detach().mean().cpu())
        anchor_lp_actor_metric = float(anchor_lp_actor.detach().mean().cpu())

    if variant in {"full", "replay_only"} and replay_log_prob is not None and bool(is_replay.any()):
        replay_idx = is_replay.nonzero(as_tuple=True)[0]
        actor_on_replay_log_prob = log_prob.index_select(0, replay_idx)
        frozen_replay_log_prob = replay_log_prob.to(device=log_prob.device, dtype=log_prob.dtype).detach().index_select(
            0, replay_idx
        )
        replay_mask = response_mask.index_select(0, replay_idx)
        replay_gate_values = gate.index_select(0, replay_idx)
        replay_group_ids = [list(group_ids)[i] for i in replay_idx.tolist()] if group_ids is not None else None

        replay_seq_kl = response_length_normalized_mean(frozen_replay_log_prob - actor_on_replay_log_prob, replay_mask)
        gated_replay_seq_kl = replay_gate_values * replay_seq_kl
        replay_loss = _group_mean(gated_replay_seq_kl, replay_group_ids)

        replay_lp_replay = response_length_normalized_mean(frozen_replay_log_prob, replay_mask)
        actor_lp_replay = response_length_normalized_mean(actor_on_replay_log_prob.detach(), replay_mask)
        replay_kl_metric = float(replay_seq_kl.detach().mean().cpu())
        replay_lp_replay_metric = float(replay_lp_replay.detach().mean().cpu())
        actor_lp_replay_metric = float(actor_lp_replay.detach().mean().cpu())

    loss = float(beta) * anchor_loss - float(beta) * replay_loss
    group_reward_mean_detached = group_reward_mean.detach()
    gate_detached = gate.detach()
    actor_group_rewards = group_reward_mean_detached[is_actor] if bool(is_actor.any()) else group_reward_mean_detached

    metrics = {
        "tafr_grpo/loss_total": float(loss.detach().cpu()),
        "tafr_grpo/kl_anchor": anchor_kl_metric,
        "tafr_grpo/kl_replay": replay_kl_metric,
        "tafr_grpo/replay_gate_mean": float(gate_detached.mean().cpu()),
        "tafr_grpo/mean_group_reward": float(actor_group_rewards.mean().cpu()),
        "tafr_grpo/fraction_all_wrong_groups": float((actor_group_rewards == 0).float().mean().cpu()),
        "tafr_grpo/fraction_all_correct_groups": float((actor_group_rewards == 1).float().mean().cpu()),
        "tafr_grpo/fraction_mixed_groups": float(
            ((actor_group_rewards > 0) & (actor_group_rewards < 1)).float().mean().cpu()
        ),
        "tafr_grpo/actor_logprob_on_actor_samples": actor_lp_actor_metric,
        "tafr_grpo/anchor_logprob_on_actor_samples": anchor_lp_actor_metric,
        "tafr_grpo/replay_logprob_on_replay_samples": replay_lp_replay_metric,
        "tafr_grpo/actor_logprob_on_replay_samples": actor_lp_replay_metric,
        "tafr_grpo/beta": float(beta),
    }
    return TAFRLossOutput(loss=loss, metrics=metrics)
