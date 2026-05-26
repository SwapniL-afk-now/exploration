# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from verl.utils import tensordict_utils as tu

SINKHORN_EPSILON = 0.05
SINKHORN_ITERS = 30
EPS = 1e-8
_EMBED_DIM = 64


@dataclass
class WassersteinGuidanceStats:
    loss: torch.Tensor
    active_groups: int
    mixed_group_fraction: float
    mean_wrong_mass: float


def _as_list(value: Any) -> list[Any]:
    value = tu.unwrap_non_tensor_data(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    try:
        return [tu.unwrap_non_tensor_data(item) for item in value]
    except TypeError:
        return [value]


def _sequence_log_probs(log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    mask = response_mask.to(dtype=log_prob.dtype)
    token_prob = log_prob.exp().detach()
    weights = token_prob * mask
    weights = weights / (weights.sum(dim=-1, keepdim=True) + EPS)
    return (weights.detach() * log_prob).sum(dim=-1)


def _response_embeddings(log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Embed each response as a probability-weighted sum of sinusoidal positional features.

    Using log_prob (the model's own probability signal) rather than raw token IDs means the
    cost matrix reflects what the model has actually learned, not an arbitrary token-ID hash.
    """
    mask = response_mask.detach().to(dtype=torch.float32)
    probs = log_prob.detach().float().exp()  # [bsz, seq_len]
    seq_len = log_prob.shape[-1]
    freqs = torch.arange(1, _EMBED_DIM + 1, device=log_prob.device, dtype=torch.float32)
    positions = torch.arange(seq_len, device=log_prob.device, dtype=torch.float32)
    # [seq_len, _EMBED_DIM] sinusoidal positional features
    phases = positions.unsqueeze(-1) * freqs.view(1, -1) * (2.0 * 3.141592653589793 / max(seq_len, 1))
    features = torch.sin(phases) + torch.cos(phases)  # [seq_len, _EMBED_DIM]
    # Weight positional features by token probabilities then mask
    weights = probs * mask  # [bsz, seq_len]
    weights = weights / (weights.sum(dim=-1, keepdim=True) + EPS)
    emb = (weights.unsqueeze(-1) * features.unsqueeze(0)).sum(dim=1)  # [bsz, _EMBED_DIM]
    return F.normalize(emb, p=2, dim=-1, eps=EPS).detach()


def _cost_matrix(group_log_prob: torch.Tensor, group_mask: torch.Tensor) -> torch.Tensor:
    emb = _response_embeddings(group_log_prob, group_mask)
    cost = (1.0 - emb @ emb.transpose(0, 1)).clamp_(0.0, 1.0)
    cost.fill_diagonal_(0.0)
    return cost.detach()


def _sinkhorn_cost(p: torch.Tensor, q: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    log_k = -cost / SINKHORN_EPSILON
    log_u = torch.zeros_like(p)
    log_v = torch.zeros_like(q)
    log_p = (p + EPS).log()
    log_q = (q + EPS).log()
    for _ in range(SINKHORN_ITERS):
        log_u = log_p - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1)
        log_v = log_q - torch.logsumexp(log_k.transpose(0, 1) + log_u.unsqueeze(0), dim=1)
    log_gamma = log_u.unsqueeze(1) + log_k + log_v.unsqueeze(0)
    gamma = log_gamma.exp()
    return (gamma * cost).sum()


def compute_wasserstein_guidance_loss(
    *,
    log_prob: torch.Tensor,
    response_mask: torch.Tensor,
    rewards: torch.Tensor | None,
    group_ids: list[Any],
    alpha_transport: float,
    responses: torch.Tensor | None = None,  # kept for API compatibility, no longer used
) -> WassersteinGuidanceStats:
    zero = log_prob.sum() * 0.0
    if rewards is None or len(group_ids) != log_prob.shape[0]:
        return WassersteinGuidanceStats(zero, 0, 0.0, 0.0)

    rewards = rewards.to(device=log_prob.device, dtype=log_prob.dtype).view(-1)
    binary_rewards = rewards > 0.5
    seq_log_probs = _sequence_log_probs(log_prob, response_mask)
    unique_group_ids = list(dict.fromkeys(group_ids))
    group_losses: list[torch.Tensor] = []
    wrong_masses: list[float] = []

    for group_id in unique_group_ids:
        indices = [idx for idx, uid in enumerate(group_ids) if uid == group_id]
        if len(indices) < 2:
            continue
        idx = torch.tensor(indices, device=log_prob.device, dtype=torch.long)
        group_correct = binary_rewards.index_select(0, idx)
        wrong = ~group_correct
        correct = group_correct
        if not bool(wrong.any() and correct.any()):
            continue

        ell = seq_log_probs.index_select(0, idx)
        p_theta = torch.softmax(ell, dim=0)
        p_sg = p_theta.detach()
        p_wrong = p_sg[wrong].sum()
        p_correct = p_sg[correct].sum()

        q = p_sg.clone()
        q[wrong] = (1.0 - alpha_transport) * p_sg[wrong]
        q[correct] = p_sg[correct] + alpha_transport * p_wrong * p_sg[correct] / (p_correct + EPS)
        q = (q / (q.sum() + EPS)).detach()

        cost = _cost_matrix(log_prob.index_select(0, idx), response_mask.index_select(0, idx))
        transport_cost = _sinkhorn_cost(p_theta, q, cost)
        moved_mass = alpha_transport * p_wrong
        group_losses.append(transport_cost / (moved_mass + EPS))
        wrong_masses.append(float(p_wrong.detach().item()))

    if not group_losses:
        return WassersteinGuidanceStats(zero, 0, 0.0, 0.0)

    loss = torch.stack(group_losses).mean()
    mixed_fraction = len(group_losses) / max(len(unique_group_ids), 1)
    mean_wrong_mass = sum(wrong_masses) / max(len(wrong_masses), 1)
    return WassersteinGuidanceStats(loss, len(group_losses), mixed_fraction, mean_wrong_mass)
