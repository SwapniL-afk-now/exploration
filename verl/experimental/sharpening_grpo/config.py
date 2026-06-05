# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf


@dataclass
class SharpeningGRPOConfig:
    """Typed config for Sequence Sharpening GRPO.

    Sequence Sharpening adds a verifier-free sequence-level loss on top of
    standard GRPO. The full actor loss is:

        L(theta) = use_grpo_reward * L_grpo(theta)
                 + gamma * alpha * L_seq(theta)
                 + beta * L_kl_k3(theta)

    where
        * L_grpo    -- the standard PPO-clipped policy loss with the supplied
                       advantage (toggleable for verifier-free ablations).
        * L_seq     -- the entropy-rate suffix log-likelihood surrogate, re-weighted
                       by the group-relative sequence advantage (GRSA) for
                       variance reduction and length-bias correction.
        * L_kl_k3   -- the unbiased K3 KL penalty
                       (exp(r) - 1 - r) toward the reference policy.
    """

    enable: bool = False
    use_grpo_reward: bool = True
    gamma: float = 0.1
    alpha: float = 1.0
    beta: float = 0.1
    group_size: int = 8
    clip_ratio: float = 0.2

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "SharpeningGRPOConfig":
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def sharpening_enabled(config: DictConfig | dict) -> bool:
    custom = config.get("custom_sharpening_grpo", {}) if config is not None else {}
    return bool(custom and custom.get("enable", False))


def validate_sharpening_config(config: DictConfig | dict) -> SharpeningGRPOConfig:
    """Validate root verl config and return the typed Sharpening config.

    Hard requirements (errors):
        * adv_estimator must be "grpo" (the algorithm is a GRPO variant).
        * group_size must match ``actor_rollout_ref.rollout.n`` (so that the
          reshape (num_prompts, G, seq_len) is exact).
        * the trainer must compute ref_log_prob (we need a reference policy).
        * gamma, alpha, beta must be non-negative.
        * clip_ratio must be positive.

    Soft warnings (printed, not raised):
        * the on-policy ``data.train_batch_size * rollout.n`` is not divisible
          by ``group_size`` (would indicate a misconfiguration upstream).
    """

    custom = SharpeningGRPOConfig.from_config(
        config.get("custom_sharpening_grpo", {}) if config is not None else {}
    )
    if not custom.enable:
        return custom

    algorithm = config.get("algorithm", {})
    actor = config.get("actor_rollout_ref", {}).get("actor", {})

    raw_adv_estimator = algorithm.get("adv_estimator", "")
    adv_estimator = str(getattr(raw_adv_estimator, "value", raw_adv_estimator)).lower()
    if adv_estimator != "grpo":
        raise ValueError("custom_sharpening_grpo.enable=true requires algorithm.adv_estimator='grpo'.")

    rollout_n = config.get("actor_rollout_ref", {}).get("rollout", {}).get("n", None)
    if rollout_n is not None and int(rollout_n) != int(custom.group_size):
        raise ValueError(
            f"custom_sharpening_grpo.group_size ({custom.group_size}) must equal "
            f"actor_rollout_ref.rollout.n ({rollout_n})."
        )

    if bool(actor.get("use_kl_loss", False)):
        raise ValueError(
            "custom_sharpening_grpo manages its own K3 KL term; set "
            "actor_rollout_ref.actor.use_kl_loss=false to avoid double-counting."
        )

    if custom.gamma < 0:
        raise ValueError("custom_sharpening_grpo.gamma must be non-negative.")
    if custom.alpha < 0:
        raise ValueError("custom_sharpening_grpo.alpha must be non-negative.")
    if custom.beta < 0:
        raise ValueError("custom_sharpening_grpo.beta must be non-negative.")
    if custom.clip_ratio <= 0:
        raise ValueError("custom_sharpening_grpo.clip_ratio must be positive.")
    if custom.group_size <= 0:
        raise ValueError("custom_sharpening_grpo.group_size must be positive.")

    train_batch_size = config.get("data", {}).get("train_batch_size", None)
    if train_batch_size is not None and rollout_n is not None:
        real_bsz = int(train_batch_size) * int(rollout_n)
        if real_bsz % int(custom.group_size) != 0:
            print(
                f"WARNING: data.train_batch_size * rollout.n = {real_bsz} is not "
                f"divisible by custom_sharpening_grpo.group_size = {custom.group_size}. "
                f"GRSA will reshape using the actual batch size and may not align with the group axis."
            )

    return custom
