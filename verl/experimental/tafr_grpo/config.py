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
    anchor_beta: float = 0.0
    ema_gamma: float = 0.99
    mix_eta: float = 1.0
    sft_update_interval_grpo_steps: int = 5
    checkpoint_interval_grpo_steps: int = 10
    disable_builtin_kl: bool = True
    failure_sft_lr: float = 1e-6
    failure_sft_batch_size: int = 8
    # Token-budgeted micro-batching for failure-SFT. When > 0, records are packed
    # greedily so each micro-batch's padded token count (n_rows * max_len) stays
    # under this budget, raising GPU utilization vs the fixed failure_sft_batch_size
    # (which is used as a fallback / row cap when this is 0).
    failure_sft_max_token_len_per_gpu: int = 0
    failure_sft_max_updates_per_interval: int = 1
    # Throttle heavy TAFR disk writes independently of the EMA refresh cadence.
    # 0 = write on every refresh (legacy). When > 0, the EMA refresh + vLLM adapter
    # export still run every checkpoint_interval, but the failure model / optimizer /
    # tafr_state.pt are only written when global_step % this == 0. Old dirs are still
    # rotated away, so the latest saved state replaces the previous one.
    save_to_disk_interval_grpo_steps: int = 0
    # Skip persisting the failure-SFT Adam optimizer state (the largest TAFR file).
    # Set False to halve TAFR checkpoint size at the cost of losing failure-SFT
    # optimizer momentum across a resume.
    save_failure_sft_optimizer: bool = True
    failure_data_max_size: Optional[int] = None
    failure_data_sampling: str = "recent"
    anchor_checkpoint_dir: Optional[str] = None
    replay_checkpoint_dir: Optional[str] = None
    variant: str = "full"
    logprob_backend: str = "hf"
    vllm_score_micro_batch_size: int = 16

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
    if custom.failure_sft_batch_size <= 0:
        raise ValueError("custom_tafr_grpo.failure_sft_batch_size must be positive.")
    if custom.failure_sft_max_token_len_per_gpu < 0:
        raise ValueError("custom_tafr_grpo.failure_sft_max_token_len_per_gpu must be >= 0 (0 disables token batching).")
    if custom.save_to_disk_interval_grpo_steps < 0:
        raise ValueError("custom_tafr_grpo.save_to_disk_interval_grpo_steps must be >= 0 (0 saves every refresh).")
    if custom.failure_sft_max_updates_per_interval <= 0:
        raise ValueError("custom_tafr_grpo.failure_sft_max_updates_per_interval must be positive (use 9999 for unlimited).")
    if custom.failure_data_max_size is not None and custom.failure_data_max_size <= 0:
        raise ValueError("custom_tafr_grpo.failure_data_max_size must be positive or null.")
    if custom.failure_data_sampling not in {"recent", "uniform"}:
        raise ValueError("custom_tafr_grpo.failure_data_sampling must be 'recent' or 'uniform'.")
    if custom.variant not in TAFR_VARIANTS:
        raise ValueError(f"custom_tafr_grpo.variant must be one of {sorted(TAFR_VARIANTS)}.")
    if custom.logprob_backend not in {"hf", "vllm"}:
        raise ValueError("custom_tafr_grpo.logprob_backend must be 'hf' or 'vllm'.")
    if custom.vllm_score_micro_batch_size <= 0:
        raise ValueError("custom_tafr_grpo.vllm_score_micro_batch_size must be positive.")

    return custom
