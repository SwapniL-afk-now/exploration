# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig, OmegaConf


TAFR_VARIANTS = {"full", "anchor_only", "replay_only"}


@dataclass
class TAFRGRPOConfig:
    """Typed config for TAFR-GRPO.

    TAFR-GRPO actor loss:

    L_t(theta) = L_GRPO(theta)
        + beta D_KL(pi_theta || pi_anchor^t)
        - beta(1 - r_bar_x) D_KL(pi_replay^t || pi_theta)

    where pi_anchor and pi_replay are frozen lagged EMA/reference mixtures.
    """

    enable: bool = False
    beta: float = 0.0
    ema_gamma: float = 0.99
    mix_eta: float = 1.0
    sft_update_interval_grpo_steps: int = 5
    checkpoint_interval_grpo_steps: int = 10
    replay_num_samples: int = 1
    disable_builtin_kl: bool = True
    failure_sft_lr: float = 1e-6
    failure_sft_batch_size: int = 8
    failure_sft_max_updates_per_interval: int = 1
    failure_data_max_size: Optional[int] = None
    failure_data_sampling: str = "recent"
    anchor_checkpoint_dir: Optional[str] = None
    replay_checkpoint_dir: Optional[str] = None
    variant: str = "full"

    @classmethod
    def from_config(cls, config: DictConfig | dict | None) -> "TAFRGRPOConfig":
        if not config:
            return cls()
        merged = OmegaConf.merge(OmegaConf.structured(cls), OmegaConf.create(config))
        return OmegaConf.to_object(merged)


def tafr_enabled(config: DictConfig | dict) -> bool:
    custom = config.get("custom_tafr_grpo", {}) if config is not None else {}
    return bool(custom and custom.get("enable", False))


def should_run_failure_sft(global_grpo_step: int, config: TAFRGRPOConfig) -> bool:
    return bool(config.enable and global_grpo_step % config.sft_update_interval_grpo_steps == 0)


def should_checkpoint_and_refresh(global_grpo_step: int, config: TAFRGRPOConfig) -> bool:
    return bool(config.enable and global_grpo_step % config.checkpoint_interval_grpo_steps == 0)


def validate_tafr_config(config: DictConfig | dict) -> TAFRGRPOConfig:
    """Validate root verl config and return the typed TAFR config."""

    custom = TAFRGRPOConfig.from_config(config.get("custom_tafr_grpo", {}) if config is not None else {})
    if not custom.enable:
        return custom

    algorithm = config.get("algorithm", {})
    actor = config.get("actor_rollout_ref", {}).get("actor", {})

    raw_adv_estimator = algorithm.get("adv_estimator", "")
    adv_estimator = str(getattr(raw_adv_estimator, "value", raw_adv_estimator)).lower()
    if adv_estimator != "grpo":
        raise ValueError("custom_tafr_grpo.enable=true requires algorithm.adv_estimator='grpo'.")

    if custom.disable_builtin_kl:
        if bool(algorithm.get("use_kl_in_reward", False)):
            raise ValueError("TAFR-GRPO requires algorithm.use_kl_in_reward=false when disable_builtin_kl=true.")
        if bool(actor.get("use_kl_loss", False)):
            raise ValueError("TAFR-GRPO requires actor_rollout_ref.actor.use_kl_loss=false.")

    if custom.beta < 0:
        raise ValueError("custom_tafr_grpo.beta must be non-negative.")
    if not 0 <= custom.ema_gamma <= 1:
        raise ValueError("custom_tafr_grpo.ema_gamma must be in [0, 1].")
    if not 0 <= custom.mix_eta <= 1:
        raise ValueError("custom_tafr_grpo.mix_eta must be in [0, 1].")
    if custom.sft_update_interval_grpo_steps <= 0:
        raise ValueError("custom_tafr_grpo.sft_update_interval_grpo_steps must be positive.")
    if custom.checkpoint_interval_grpo_steps <= 0:
        raise ValueError("custom_tafr_grpo.checkpoint_interval_grpo_steps must be positive.")
    if custom.replay_num_samples <= 0:
        raise ValueError("custom_tafr_grpo.replay_num_samples must be positive.")
    if custom.failure_sft_batch_size <= 0:
        raise ValueError("custom_tafr_grpo.failure_sft_batch_size must be positive.")
    if custom.failure_sft_max_updates_per_interval <= 0:
        raise ValueError("custom_tafr_grpo.failure_sft_max_updates_per_interval must be positive.")
    if custom.failure_data_max_size is not None and custom.failure_data_max_size <= 0:
        raise ValueError("custom_tafr_grpo.failure_data_max_size must be positive or null.")
    if custom.failure_data_sampling not in {"recent", "uniform"}:
        raise ValueError("custom_tafr_grpo.failure_data_sampling must be 'recent' or 'uniform'.")
    if custom.variant not in TAFR_VARIANTS:
        raise ValueError(f"custom_tafr_grpo.variant must be one of {sorted(TAFR_VARIANTS)}.")

    return custom
