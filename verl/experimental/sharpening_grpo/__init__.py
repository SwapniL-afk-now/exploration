# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sequence Sharpening GRPO utilities.

Adds a verifier-free sequence-level loss on top of standard GRPO using
two upgrades:

* Entropy-Rate (Geometric) Sharpening -- reverse-cumulative suffix
  log-likelihood weights that remove the implicit length bias of a
  sequence-level likelihood surrogate.
* Group-Relative Sequence Advantage (GRSA) -- variance reduction by
  z-scoring the entropy-rate weight within each prompt's rollouts.

The total actor loss is

    L = use_grpo_reward * L_grpo
      + gamma * alpha * L_seq
      + beta * L_kl_k3
"""

from verl.experimental.sharpening_grpo.config import (
    SharpeningGRPOConfig,
    sharpening_enabled,
    validate_sharpening_config,
)
from verl.experimental.sharpening_grpo.loss import (
    SharpeningLossOutput,
    compute_sharpening_grpo_loss,
)

__all__ = [
    "SharpeningGRPOConfig",
    "SharpeningLossOutput",
    "compute_sharpening_grpo_loss",
    "sharpening_enabled",
    "validate_sharpening_config",
]
