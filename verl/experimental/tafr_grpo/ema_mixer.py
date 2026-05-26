# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch


StateDict = Mapping[str, torch.Tensor]


def mix_state_dicts(base: StateDict, delta: StateDict, eta: float) -> dict[str, torch.Tensor]:
    if not 0 <= eta <= 1:
        raise ValueError("eta must be in [0, 1].")
    mixed = {}
    for key, base_value in base.items():
        if key not in delta:
            mixed[key] = base_value.detach().clone()
            continue
        other = delta[key].to(dtype=base_value.dtype)
        if torch.is_floating_point(base_value):
            mixed[key] = (1.0 - eta) * base_value.detach().clone() + eta * other.detach().clone()
        else:
            mixed[key] = other.detach().clone()
    return mixed


@dataclass
class CheckpointEMA:
    """EMA updated only from saved checkpoints on the GRPO global clock."""

    gamma: float
    ema_state: Optional[dict[str, torch.Tensor]] = None
    last_checkpoint_step: Optional[int] = None
    last_source_id: Optional[str] = None
    updated_steps: list[int] = field(default_factory=list)

    def initialize(self, reference_state: StateDict) -> None:
        self.ema_state = {key: value.detach().clone() for key, value in reference_state.items()}
        self.last_checkpoint_step = 0
        self.last_source_id = "reference"

    def update_from_checkpoint(self, state: StateDict, *, global_step: int, source_id: str) -> bool:
        if self.ema_state is None:
            self.initialize(state)
            self.last_checkpoint_step = global_step
            self.last_source_id = source_id
            return True
        if source_id == self.last_source_id:
            return False
        updated = {}
        for key, ema_value in self.ema_state.items():
            current_value = state[key].to(dtype=ema_value.dtype)
            if torch.is_floating_point(ema_value):
                updated[key] = self.gamma * ema_value + (1.0 - self.gamma) * current_value.detach()
            else:
                updated[key] = current_value.detach().clone()
        self.ema_state = updated
        self.last_checkpoint_step = global_step
        self.last_source_id = source_id
        self.updated_steps.append(global_step)
        return True

    def materialize_mixed(self, reference_state: StateDict, eta: float) -> dict[str, torch.Tensor]:
        if self.ema_state is None:
            self.initialize(reference_state)
        assert self.ema_state is not None
        return mix_state_dicts(reference_state, self.ema_state, eta)


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def assert_frozen(module: torch.nn.Module) -> None:
    trainable = [name for name, param in module.named_parameters() if param.requires_grad]
    if trainable:
        raise AssertionError(f"Expected module to be frozen, but found trainable params: {trainable[:5]}")


def assert_optimizer_excludes_module(optimizer: torch.optim.Optimizer, module: torch.nn.Module) -> None:
    frozen_param_ids = {id(param) for param in module.parameters()}
    optimizer_param_ids = {id(param) for group in optimizer.param_groups for param in group["params"]}
    overlap = frozen_param_ids & optimizer_param_ids
    if overlap:
        raise AssertionError("Optimizer contains frozen anchor/replay/reference model parameters.")
