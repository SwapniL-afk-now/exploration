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

from dataclasses import dataclass
from typing import Optional

import torch

from verl.base_config import BaseConfig


@dataclass
class ExplorationConfig(BaseConfig):
    """Stable Exploration Divergence configuration."""

    enabled: bool = False
    alpha_max: float = 0.3
    delta: float = 3.0
    ema_gamma: float = 0.995
    ema_update_every: int = 5
    entropy_threshold_ratio: float = 0.5
    warmup_steps: int = 100

    def __post_init__(self):
        if self.delta <= 0:
            raise ValueError("exploration.delta must be positive.")
        if not 0 <= self.ema_gamma < 1:
            raise ValueError("exploration.ema_gamma must be in [0, 1).")
        if self.ema_update_every <= 0:
            raise ValueError("exploration.ema_update_every must be positive.")
        if self.alpha_max < 0:
            raise ValueError("exploration.alpha_max must be non-negative.")
        if self.entropy_threshold_ratio < 0:
            raise ValueError("exploration.entropy_threshold_ratio must be non-negative.")
        if self.warmup_steps < 0:
            raise ValueError("exploration.warmup_steps must be non-negative.")


class EMALoRATracker:
    """Detached EMA memory over LoRA parameters."""

    def __init__(self, model: torch.nn.Module, gamma: float = 0.995, update_every: int = 5):
        self.gamma = gamma
        self.update_every = update_every
        self.optimizer_steps = 0
        self.params = self._collect_lora_params(model)
        self.ema_params = {name: param.detach().clone() for name, param in self.params.items()}
        for param in self.ema_params.values():
            param.requires_grad_(False)

    @staticmethod
    def _is_lora_param(name: str, param: torch.nn.Parameter) -> bool:
        lower_name = name.lower()
        return param.requires_grad and ("lora_" in lower_name or ".adapter_" in lower_name)

    def _collect_lora_params(self, model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
        params = {
            name: param
            for name, param in model.named_parameters()
            if self._is_lora_param(name=name, param=param)
        }
        if params:
            return params

        unwrapped = getattr(model, "_fsdp_wrapped_module", model)
        return {
            name: param
            for name, param in unwrapped.named_parameters()
            if self._is_lora_param(name=name, param=param)
        }

    @property
    def has_params(self) -> bool:
        return bool(self.params)

    def load_into_model(self, model: Optional[torch.nn.Module] = None) -> dict[str, torch.Tensor]:
        del model
        saved = {}
        with torch.no_grad():
            for name, param in self.params.items():
                saved[name] = param.detach().clone()
                ema_param = self.ema_params[name].to(device=param.device, dtype=param.dtype)
                param.copy_(ema_param)
        return saved

    def restore_model(self, model: Optional[torch.nn.Module], saved: dict[str, torch.Tensor]) -> None:
        del model
        with torch.no_grad():
            for name, value in saved.items():
                self.params[name].copy_(value.to(device=self.params[name].device, dtype=self.params[name].dtype))

    def update(self, model: Optional[torch.nn.Module] = None) -> bool:
        del model
        self.optimizer_steps += 1
        if not self.has_params or self.optimizer_steps % self.update_every != 0:
            return False

        with torch.no_grad():
            for name, param in self.params.items():
                current = param.detach().to(device=self.ema_params[name].device, dtype=self.ema_params[name].dtype)
                self.ema_params[name].mul_(self.gamma).add_(current, alpha=1.0 - self.gamma)
        return True


class StableExplorationDivergence:
    """Computes the SED auxiliary loss and owns its optimizer-step state."""

    STATUS_DISABLED = 0
    STATUS_WARMUP = 1
    STATUS_ACTIVE = 2
    STATUS_ENTROPY_GATED = 3
    STATUS_CLAMPED_DORMANT = 4

    def __init__(self, config: ExplorationConfig, ema_tracker: Optional[EMALoRATracker] = None):
        self.config = config
        self.ema_tracker = ema_tracker
        self.initial_entropy: Optional[torch.Tensor] = None
        self.optimizer_steps = 0
        self.disabled_reason: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled) and self.disabled_reason is None

    @property
    def has_ema(self) -> bool:
        return self.ema_tracker is not None and self.ema_tracker.has_params

    def disable(self, reason: str) -> None:
        self.disabled_reason = reason

    def should_run_ema_forward(self) -> bool:
        return self.enabled and self.has_ema and self.optimizer_steps >= self.config.warmup_steps

    def after_optimizer_step(self, model: torch.nn.Module) -> None:
        if not self.config.enabled:
            return
        self.optimizer_steps += 1
        if self.ema_tracker is not None:
            self.ema_tracker.update(model)

    def _zero_loss(self, current_log_probs: torch.Tensor) -> torch.Tensor:
        return current_log_probs.sum() * 0.0

    def _metrics(
        self,
        *,
        current_log_probs: torch.Tensor,
        raw_kl: Optional[torch.Tensor] = None,
        clamped_kl: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
        entropy_ratio: Optional[torch.Tensor] = None,
        entropy_gate: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        status: int,
    ) -> dict[str, torch.Tensor]:
        zero = current_log_probs.detach().new_tensor(0.0)
        return {
            "explore/raw_kl": zero if raw_kl is None else raw_kl.detach(),
            "explore/clamped_kl": zero if clamped_kl is None else clamped_kl.detach(),
            "explore/alpha_adaptive": zero if alpha is None else alpha.detach(),
            "explore/entropy_ratio": zero if entropy_ratio is None else entropy_ratio.detach(),
            "explore/entropy_gate": zero if entropy_gate is None else entropy_gate.detach(),
            "explore/loss": zero if loss is None else loss.detach(),
            "explore/status": current_log_probs.detach().new_tensor(float(status)),
        }

    def _entropy_ratio(self, entropy: Optional[torch.Tensor], response_mask: torch.Tensor) -> torch.Tensor:
        if entropy is None:
            return response_mask.detach().new_tensor(1.0, dtype=torch.float32)

        mask = response_mask.to(dtype=entropy.dtype)
        entropy_mean = (entropy * mask).sum() / mask.sum().clamp(min=1)
        if self.initial_entropy is None:
            self.initial_entropy = entropy_mean.detach().clamp(min=1e-8)
        return entropy_mean / self.initial_entropy.to(device=entropy_mean.device, dtype=entropy_mean.dtype)

    def compute_loss(
        self,
        *,
        current_log_probs: torch.Tensor,
        ema_log_probs: Optional[torch.Tensor],
        entropy: Optional[torch.Tensor],
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss = self._zero_loss(current_log_probs)
        entropy_ratio = self._entropy_ratio(entropy=entropy, response_mask=response_mask)

        if not self.enabled or not self.has_ema:
            return loss, self._metrics(
                current_log_probs=current_log_probs,
                entropy_ratio=entropy_ratio,
                loss=loss,
                status=self.STATUS_DISABLED,
            )

        if self.optimizer_steps < self.config.warmup_steps or ema_log_probs is None:
            return loss, self._metrics(
                current_log_probs=current_log_probs,
                entropy_ratio=entropy_ratio,
                entropy_gate=current_log_probs.detach().new_tensor(1.0),
                loss=loss,
                status=self.STATUS_WARMUP,
            )

        ema_log_probs = ema_log_probs.detach().clamp(min=-20.0)
        mask = response_mask.to(dtype=current_log_probs.dtype)
        per_token_kl = torch.exp(ema_log_probs) * (ema_log_probs - current_log_probs)
        per_seq_kl = (per_token_kl * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        raw_kl = per_seq_kl.mean()
        clamped_kl = torch.clamp(raw_kl, max=self.config.delta)

        alpha_scale = 1.0 - clamped_kl.detach().clamp(min=0.0, max=self.config.delta) / self.config.delta
        alpha = current_log_probs.new_tensor(self.config.alpha_max) * alpha_scale
        entropy_gate = (entropy_ratio >= self.config.entropy_threshold_ratio).to(dtype=current_log_probs.dtype)

        loss = -alpha * entropy_gate * clamped_kl
        status = self.STATUS_ACTIVE
        if entropy_gate.detach().item() == 0:
            status = self.STATUS_ENTROPY_GATED
        elif raw_kl.detach().item() >= self.config.delta:
            status = self.STATUS_CLAMPED_DORMANT

        return loss, self._metrics(
            current_log_probs=current_log_probs,
            raw_kl=raw_kl,
            clamped_kl=clamped_kl,
            alpha=alpha,
            entropy_ratio=entropy_ratio,
            entropy_gate=entropy_gate,
            loss=loss,
            status=status,
        )
