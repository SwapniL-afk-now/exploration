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
"""FEPO/Dr.GRPO algorithm primitives.

These functions intentionally mirror the Kaggle notebook's token-level losses:
group-centered Dr.GRPO advantages without std normalization, solver FEPO with a
failure-escape term, and wrong-only SFT for the failure adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(frozen=True)
class GroupAdvantageResult:
    rewards: torch.Tensor
    anti_rewards: torch.Tensor
    solver_advantages: torch.Tensor
    failure_advantages: torch.Tensor
    reward_mean: torch.Tensor
    reward_std: torch.Tensor
    anti_reward_mean: torch.Tensor
    anti_reward_std: torch.Tensor


def compute_group_advantages(rewards: torch.Tensor | list[float], num_generations: int) -> GroupAdvantageResult:
    """Compute Dr.GRPO group-centered solver and sign-flipped failure advantages."""

    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.numel() == 0:
        empty = rewards.reshape(0)
        return GroupAdvantageResult(empty, empty, empty, empty, empty, empty, empty, empty)
    if num_generations <= 0:
        raise ValueError(f"num_generations must be positive, got {num_generations}")
    if rewards.numel() % num_generations != 0:
        raise ValueError(
            f"rewards length ({rewards.numel()}) must be divisible by num_generations ({num_generations})"
        )

    grouped_rewards = rewards.reshape(-1, num_generations)
    anti_rewards = 1.0 - 2.0 * grouped_rewards

    reward_mean = grouped_rewards.mean(dim=1, keepdim=True)
    reward_std = grouped_rewards.std(dim=1, keepdim=True, unbiased=False)
    anti_reward_mean = anti_rewards.mean(dim=1, keepdim=True)
    anti_reward_std = anti_rewards.std(dim=1, keepdim=True, unbiased=False)

    solver_advantages = grouped_rewards - reward_mean
    failure_advantages = anti_rewards - anti_reward_mean

    return GroupAdvantageResult(
        rewards=grouped_rewards.reshape(-1),
        anti_rewards=anti_rewards.reshape(-1),
        solver_advantages=solver_advantages.reshape(-1),
        failure_advantages=failure_advantages.reshape(-1),
        reward_mean=reward_mean.expand_as(grouped_rewards).reshape(-1),
        reward_std=reward_std.expand_as(grouped_rewards).reshape(-1),
        anti_reward_mean=anti_reward_mean.expand_as(grouped_rewards).reshape(-1),
        anti_reward_std=anti_reward_std.expand_as(grouped_rewards).reshape(-1),
    )


def attach_fepo_dr_grpo_group_advantages(items: list[dict[str, Any]]) -> None:
    """Mutate rollout records with response-level and token-level FEPO advantages."""

    if not items:
        return
    result = compute_group_advantages([float(item.get("reward", 0.0)) for item in items], len(items))
    for idx, item in enumerate(items):
        completion_length = int(item.get("completion_length", 0))
        if completion_length <= 0 and item.get("encoded") is not None:
            completion_length = int(item["encoded"].get("completion_length", 0))

        solver_advantage = float(result.solver_advantages[idx].item())
        failure_advantage = float(result.failure_advantages[idx].item())
        reward = float(result.rewards[idx].item())

        item["anti_reward"] = float(result.anti_rewards[idx].item())
        item["grpo_advantage"] = solver_advantage
        item["failure_advantage"] = failure_advantage
        item["group_reward_mean"] = float(result.reward_mean[idx].item())
        item["group_reward_std"] = float(result.reward_std[idx].item())
        item["group_anti_reward_mean"] = float(result.anti_reward_mean[idx].item())
        item["group_anti_reward_std"] = float(result.anti_reward_std[idx].item())

        if completion_length > 0:
            failed = reward == 0.0
            item["token_advantages"] = torch.full((completion_length,), solver_advantage, dtype=torch.float32)
            item["failure_token_advantages"] = torch.full((completion_length,), failure_advantage, dtype=torch.float32)
            item["failed_token_mask"] = torch.full((completion_length,), failed, dtype=torch.bool)
            item["token_loss_mask"] = torch.ones((completion_length,), dtype=torch.bool)
        else:
            item["token_advantages"] = torch.zeros((0,), dtype=torch.float32)
            item["failure_token_advantages"] = torch.zeros((0,), dtype=torch.float32)
            item["failed_token_mask"] = torch.zeros((0,), dtype=torch.bool)
            item["token_loss_mask"] = torch.zeros((0,), dtype=torch.bool)


def _zero_token_fepo_metrics(prefix: str, reference_beta: float = 0.0, escape_alpha: float = 0.0) -> dict[str, float]:
    return {
        f"{prefix}_pg_loss": 0.0,
        f"{prefix}_pg_loss_abs": 0.0,
        f"{prefix}_total_loss": 0.0,
        f"{prefix}_clip_fraction": 0.0,
        "failure_escape_kl_failed": 0.0,
        "reference_kl_all": 0.0,
        "failed_token_count": 0.0,
        "escape_alpha": float(escape_alpha),
        "reference_beta": float(reference_beta),
        "token_ratio_mean": 0.0,
        "token_log_ratio_abs_mean": 0.0,
        "loss_token_count": 0.0,
    }


def _truncate_and_mask(
    tensors: list[torch.Tensor],
    loss_mask: Optional[torch.Tensor],
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor, int]:
    n = min(tensor.numel() for tensor in tensors)
    if loss_mask is not None:
        n = min(n, loss_mask.numel())
    tensors = [tensor[:n] for tensor in tensors]
    if loss_mask is None:
        mask = torch.ones((n,), dtype=torch.bool, device=device)
    else:
        mask = loss_mask[:n].to(device=device).bool().detach()
    valid_count = int(mask.sum().detach().cpu())
    return tensors, mask, valid_count


def token_solver_fepo_loss(
    current_logps: torch.Tensor,
    old_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    failure_logps: torch.Tensor,
    advantages: torch.Tensor,
    failed_token_mask: torch.Tensor,
    clip_eps: float,
    reference_beta: float,
    escape_alpha: float,
    loss_mask: Optional[torch.Tensor] = None,
    token_log_ratio_clip: Optional[float] = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Token-level solver FEPO loss from the notebook."""

    if current_logps.numel() == 0:
        zero = current_logps.sum() * 0.0
        return zero, _zero_token_fepo_metrics("solver", reference_beta, escape_alpha)

    old_logps = old_logps.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    ref_logps = ref_logps.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    failure_logps = failure_logps.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    advantages = advantages.to(device=current_logps.device, dtype=current_logps.dtype).detach()
    failed_token_mask = failed_token_mask.to(device=current_logps.device).bool().detach()

    tensors, mask, valid_count = _truncate_and_mask(
        [current_logps, old_logps, ref_logps, failure_logps, advantages, failed_token_mask],
        loss_mask,
        current_logps.device,
    )
    current_logps, old_logps, ref_logps, failure_logps, advantages, failed_token_mask = tensors
    if valid_count == 0:
        zero = current_logps.sum() * 0.0
        return zero, _zero_token_fepo_metrics("solver", reference_beta, escape_alpha)

    current_logps = current_logps[mask]
    old_logps = old_logps[mask]
    ref_logps = ref_logps[mask]
    failure_logps = failure_logps[mask]
    advantages = advantages[mask]
    failed_token_mask = failed_token_mask[mask].bool()

    log_ratio = current_logps - old_logps
    if token_log_ratio_clip is not None:
        log_ratio = log_ratio.clamp(-float(token_log_ratio_clip), float(token_log_ratio_clip))
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    clipped = clipped_ratio * advantages
    pg_terms = torch.minimum(unclipped, clipped)
    pg_loss = -pg_terms.mean()
    pg_loss_abs = pg_terms.abs().mean()
    clip_fraction = ((ratio < (1.0 - clip_eps)) | (ratio > (1.0 + clip_eps))).float().mean()

    # Reference KL: ALL tokens (not just failed) - keeps solver globally anchored
    ref_minus_current = ref_logps - current_logps
    reference_kl_all = (torch.exp(ref_minus_current) - ref_minus_current - 1.0).mean()

    # Failure escape KL: failed tokens only - encourages divergence from failure
    failed_count = int(failed_token_mask.sum().detach().cpu())
    if failed_count > 0:
        failed_current_logps = current_logps[failed_token_mask]
        failed_failure_logps = failure_logps[failed_token_mask]

        failure_minus_current = failed_failure_logps - failed_current_logps
        failure_escape_kl_failed = (torch.exp(failure_minus_current) - failure_minus_current - 1.0).mean()
    else:
        failure_escape_kl_failed = current_logps.sum() * 0.0

    loss = pg_loss + float(reference_beta) * reference_kl_all - float(escape_alpha) * failure_escape_kl_failed
    metrics = {
        "solver_pg_loss": float(pg_loss.detach().cpu()),
        "solver_pg_loss_abs": float(pg_loss_abs.detach().cpu()),
        "failure_escape_kl_failed": float(failure_escape_kl_failed.detach().cpu()),
        "reference_kl_all": float(reference_kl_all.detach().cpu()),
        "failed_token_count": float(failed_count),
        "escape_alpha": float(escape_alpha),
        "reference_beta": float(reference_beta),
        "solver_total_loss": float(loss.detach().cpu()),
        "solver_clip_fraction": float(clip_fraction.detach().cpu()),
        "token_ratio_mean": float(ratio.detach().mean().cpu()),
        "token_log_ratio_abs_mean": float(log_ratio.detach().abs().mean().cpu()),
        "loss_token_count": float(valid_count),
    }
    return loss, metrics


def _zero_failure_sft_metrics() -> dict[str, float]:
    return {
        "failure_sft_loss": 0.0,
        "failure_total_loss": 0.0,
        "failure_sft_token_count": 0.0,
        "failure_is_sft_active": 0.0,
    }


def token_failure_sft_loss(
    current_logps: torch.Tensor,
    is_failed: bool,
    loss_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Wrong-only failure adapter SFT loss."""

    if current_logps.numel() == 0 or not bool(is_failed):
        zero = current_logps.sum() * 0.0
        return zero, _zero_failure_sft_metrics()

    tensors, mask, valid_count = _truncate_and_mask([current_logps], loss_mask, current_logps.device)
    (current_logps,) = tensors
    if valid_count == 0:
        zero = current_logps.sum() * 0.0
        return zero, _zero_failure_sft_metrics()

    failure_sft_loss = -current_logps[mask].mean()
    metrics = {
        "failure_sft_loss": float(failure_sft_loss.detach().cpu()),
        "failure_total_loss": float(failure_sft_loss.detach().cpu()),
        "failure_sft_token_count": float(valid_count),
        "failure_is_sft_active": 1.0,
    }
    return failure_sft_loss, metrics
