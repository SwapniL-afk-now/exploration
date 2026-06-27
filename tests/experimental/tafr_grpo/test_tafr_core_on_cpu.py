# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.tafr_grpo.config import (
    TAFRGRPOConfig,
    should_checkpoint_and_refresh,
    should_run_failure_sft,
    validate_tafr_config,
)
from verl.experimental.tafr_grpo.ema_mixer import assert_frozen, assert_optimizer_excludes_module, freeze_module
from verl.experimental.tafr_grpo.failure_data_collector import FailureDataCollector
from verl.experimental.tafr_grpo.failure_sft_trainer import failure_sft_loss
from verl.experimental.tafr_grpo.tafr_loss import (
    compute_tafr_grpo_auxiliary_loss,
    replay_gate,
    response_length_normalized_mean,
)
from verl.experimental.tafr_grpo.vllm_scoring import (
    TAFR_VLLM_ADAPTERS,
    extract_response_logprobs_from_prompt_logprobs,
)


def test_validate_rejects_builtin_kl_when_enabled():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo", "use_kl_in_reward": True},
            "actor_rollout_ref": {"actor": {"use_kl_loss": False}},
            "custom_tafr_grpo": {"enable": True, "disable_builtin_kl": True},
        }
    )
    with pytest.raises(ValueError, match="use_kl_in_reward=false"):
        validate_tafr_config(cfg)


def test_failure_collector_accepts_only_reward_zero():
    collector = FailureDataCollector(max_size=10)
    assert collector.add(prompt="p", response="wrong", reward=0)
    assert not collector.add(prompt="p", response="correct", reward=1)
    assert len(collector) == 1
    assert collector.to_records()[0]["wrong_response"] == "wrong"


def test_replay_gate_boundaries():
    rewards = torch.tensor([1.0, 0.0, 0.5])
    gates = replay_gate(rewards)
    assert gates[0].item() == 0.0
    assert gates[1].item() == 1.0
    assert gates[2].item() == 0.5


def test_tafr_replay_loss_zero_for_all_correct_group():
    log_prob = torch.tensor([[-3.0, -4.0]], requires_grad=True)
    replay_log_prob = torch.tensor([[-1.0, -1.0]])
    out = compute_tafr_grpo_auxiliary_loss(
        log_prob=log_prob,
        response_mask=torch.ones_like(log_prob, dtype=torch.bool),
        group_reward_mean=torch.tensor([1.0]),
        beta=0.7,
        variant="replay_only",
        replay_log_prob=replay_log_prob,
    )
    assert out.loss.item() == 0.0


def test_tafr_replay_loss_full_for_all_wrong_group():
    log_prob = torch.tensor([[-3.0, -4.0]], requires_grad=True)
    replay_log_prob = torch.tensor([[-1.0, -1.0]])
    out = compute_tafr_grpo_auxiliary_loss(
        log_prob=log_prob,
        response_mask=torch.ones_like(log_prob, dtype=torch.bool),
        group_reward_mean=torch.tensor([0.0]),
        beta=0.5,
        variant="replay_only",
        replay_log_prob=replay_log_prob,
    )
    # k3 estimator of D_KL(pi_replay || pi_theta): (r - 1) - log_ratio, clamped, then masked-mean.
    log_ratio = (replay_log_prob - log_prob).clamp(min=-20, max=20)
    r = torch.exp(log_ratio)
    expected_kl = (((r - 1) - log_ratio).clamp(min=-10, max=10).mean()).detach()
    assert torch.allclose(out.loss, -0.5 * expected_kl)


def test_length_normalization_uses_response_mask_only():
    values = torch.tensor([[10.0, 2.0, 4.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    seq_mean = response_length_normalized_mean(values, mask)
    assert seq_mean.item() == 3.0


def test_failure_sft_masks_prompt_tokens():
    log_prob = torch.tensor([[-100.0, -2.0, -4.0]])
    mask = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    loss = failure_sft_loss(log_prob, mask)
    assert loss.item() == 3.0


def test_freeze_and_optimizer_exclusion():
    actor = torch.nn.Linear(2, 2)
    frozen = freeze_module(torch.nn.Linear(2, 2))
    opt = torch.optim.AdamW(actor.parameters(), lr=1e-3)
    assert_frozen(frozen)
    assert_optimizer_excludes_module(opt, frozen)


def test_optimizer_exclusion_catches_overlap():
    model = freeze_module(torch.nn.Linear(2, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(AssertionError, match="Optimizer contains"):
        assert_optimizer_excludes_module(opt, model)


def test_global_grpo_clock_controls_sft_and_checkpoint_schedule():
    cfg = TAFRGRPOConfig(enable=True, sft_update_interval_grpo_steps=5, checkpoint_interval_grpo_steps=10)
    assert not should_run_failure_sft(1, cfg)
    assert should_run_failure_sft(5, cfg)
    assert should_run_failure_sft(10, cfg)
    assert not should_checkpoint_and_refresh(5, cfg)
    assert should_checkpoint_and_refresh(10, cfg)


def test_vllm_prompt_logprob_slice_returns_response_only():
    prompt_logprobs = [[-0.1], [-0.2], [-1.0], [-1.5], [-2.0], [0.0]]
    response = extract_response_logprobs_from_prompt_logprobs(
        prompt_logprobs=prompt_logprobs,
        prompt_len=3,
        response_len=3,
    )
    assert response == [-1.0, -1.5, -2.0]


def test_tafr_vllm_adapter_ids_do_not_collide_with_rollout_adapter():
    rollout_lora_id = 123
    ids = {spec.int_id for spec in TAFR_VLLM_ADAPTERS.values()}
    assert rollout_lora_id not in ids
    assert len(ids) == len(TAFR_VLLM_ADAPTERS)
