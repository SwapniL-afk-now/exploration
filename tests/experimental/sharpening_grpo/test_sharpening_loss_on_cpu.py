# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

import pytest
import torch
from omegaconf import OmegaConf

from verl.experimental.sharpening_grpo.config import (
    SharpeningGRPOConfig,
    sharpening_enabled,
    validate_sharpening_config,
)
from verl.experimental.sharpening_grpo.loss import compute_sharpening_grpo_loss


# ---------------------------------------------------------------- config tests


def test_sharpening_disabled_when_flag_false():
    cfg = OmegaConf.create({"custom_sharpening_grpo": {"enable": False}})
    out = validate_sharpening_config(cfg)
    assert isinstance(out, SharpeningGRPOConfig)
    assert out.enable is False


def test_sharpening_enabled_helper():
    cfg = OmegaConf.create({"custom_sharpening_grpo": {"enable": True}})
    assert sharpening_enabled(cfg)
    cfg_off = OmegaConf.create({"custom_sharpening_grpo": {"enable": False}})
    assert not sharpening_enabled(cfg_off)
    assert not sharpening_enabled({})


def test_sharpening_rejects_wrong_adv_estimator():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "ppo"},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {"n": 8},
            },
            "custom_sharpening_grpo": {"enable": True, "group_size": 8},
        }
    )
    with pytest.raises(ValueError, match="adv_estimator='grpo'"):
        validate_sharpening_config(cfg)


def test_sharpening_rejects_builtin_kl_loss():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo"},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": True},
                "rollout": {"n": 8},
            },
            "custom_sharpening_grpo": {"enable": True, "group_size": 8},
        }
    )
    with pytest.raises(ValueError, match="use_kl_loss=false"):
        validate_sharpening_config(cfg)


def test_sharpening_rejects_group_size_mismatch_with_rollout_n():
    cfg = OmegaConf.create(
        {
            "algorithm": {"adv_estimator": "grpo"},
            "actor_rollout_ref": {
                "actor": {"use_kl_loss": False},
                "rollout": {"n": 8},
            },
            "custom_sharpening_grpo": {"enable": True, "group_size": 16},
        }
    )
    with pytest.raises(ValueError, match="group_size"):
        validate_sharpening_config(cfg)


def test_sharpening_rejects_negative_hyperparams():
    base = {
        "algorithm": {"adv_estimator": "grpo"},
        "actor_rollout_ref": {
            "actor": {"use_kl_loss": False},
            "rollout": {"n": 8},
        },
        "custom_sharpening_grpo": {"enable": True, "group_size": 8},
    }
    for field in ("gamma", "alpha", "beta"):
        cfg = OmegaConf.create({**base, "custom_sharpening_grpo": {**base["custom_sharpening_grpo"], field: -0.1}})
        with pytest.raises(ValueError, match=field):
            validate_sharpening_config(cfg)


# ---------------------------------------------------------------- loss tests


def _build_inputs(B=8, T=6, G=4, seed=0):
    """Build deterministic test inputs with shape (B, T)."""
    g = torch.Generator().manual_seed(seed)
    log_prob = torch.randn(B, T, generator=g) * 0.1 - 1.0
    log_prob.requires_grad_(True)
    old_log_prob = (log_prob.detach() + torch.randn(B, T, generator=g) * 0.05).detach()
    advantages = torch.randn(B, T, generator=g)
    ref_log_prob = (log_prob.detach() + torch.randn(B, T, generator=g) * 0.2).detach()
    response_mask = torch.ones(B, T, dtype=torch.bool)
    # Add a trailing padding region to two rows.
    response_mask[0, -2:] = False
    response_mask[3, -1:] = False
    return log_prob, old_log_prob, advantages, ref_log_prob, response_mask


def test_loss_terms_have_correct_shapes_and_finite_values():
    log_prob, old_log_prob, advantages, ref_log_prob, response_mask = _build_inputs()
    out = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        group_size=4,
        clip_ratio=0.2,
        use_grpo_reward=True,
    )
    assert out.grpo_term.ndim == 0
    assert out.seq_term.ndim == 0
    assert out.kl_term.ndim == 0
    assert torch.isfinite(out.grpo_term).item()
    assert torch.isfinite(out.seq_term).item()
    assert torch.isfinite(out.kl_term).item()
    # KL is exp(r) - 1 - r, which is >= 0 for all real r.
    assert out.kl_term.item() >= -1e-6


def test_grpo_toggle_zeros_out_grpo_term():
    log_prob, old_log_prob, advantages, ref_log_prob, response_mask = _build_inputs()
    out_on = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        group_size=4,
        clip_ratio=0.2,
        use_grpo_reward=True,
    )
    out_off = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        group_size=4,
        clip_ratio=0.2,
        use_grpo_reward=False,
    )
    assert out_off.grpo_term.item() == 0.0
    # The other terms are independent of the toggle.
    assert torch.allclose(out_on.seq_term, out_off.seq_term)
    assert torch.allclose(out_on.kl_term, out_off.kl_term)


def test_grsa_z_scores_have_near_zero_group_mean():
    B, T, G = 12, 5, 4
    log_prob, old_log_prob, advantages, ref_log_prob, _ = _build_inputs(B=B, T=T, G=G)
    # Use a fully unmasked response so the per-group mean of w_bar is exactly 0
    # by construction of z-scoring.
    response_mask = torch.ones(B, T, dtype=torch.bool)
    out = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        group_size=G,
        clip_ratio=0.2,
        use_grpo_reward=True,
    )
    # Reconstruct the w_bar tensor from ref_log_prob for the assertion.
    masked_ref = ref_log_prob * response_mask.to(dtype=ref_log_prob.dtype)
    rev_cumsum_ref = torch.flip(torch.cumsum(torch.flip(masked_ref, dims=[-1]), dim=-1), dims=[-1])
    rev_cumsum_mask = torch.flip(
        torch.cumsum(torch.flip(response_mask.to(dtype=ref_log_prob.dtype), dims=[-1]), dim=-1), dims=[-1]
    )
    w_t = rev_cumsum_ref / (rev_cumsum_mask + 1e-8)
    P = B // G
    w_r = w_t.view(P, G, T)
    mu = w_r.mean(dim=1, keepdim=True)
    sigma = w_r.std(dim=1, keepdim=True)
    w_bar = ((w_r - mu) / (sigma + 1e-8)).view(B, T) * response_mask.to(dtype=w_t.dtype)
    # With no padding tokens, the per-group mean of w_bar is exactly 0.
    per_group_mean = w_bar.view(P, G, T).mean(dim=1)
    assert torch.allclose(per_group_mean, torch.zeros_like(per_group_mean), atol=1e-5)
    # And the metric matches the recomputed value.
    assert abs(out.metrics["sharpen/mu_group_mean"] - float(mu.mean().cpu())) < 1e-5
    assert abs(out.metrics["sharpen/sigma_group_mean"] - float(sigma.mean().cpu())) < 1e-5


def test_response_mask_strictly_respected_for_kl():
    B, T, G = 8, 6, 4
    # Use a fully unmasked mask in the helper to avoid pre-existing padding
    # making the expected value trickier to express in one line.
    log_prob, old_log_prob, advantages, ref_log_prob, _ = _build_inputs(B=B, T=T, G=G)
    full_mask = torch.ones(B, T, dtype=torch.bool)
    kl_full = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=full_mask,
        ref_log_prob=ref_log_prob,
        group_size=G,
        clip_ratio=0.2,
        use_grpo_reward=False,
    ).kl_term

    # Drop every token in row 0; the kl term should be the mean over the
    # remaining (B-1)*T tokens.
    masked = full_mask.clone()
    masked[0, :] = False
    kl_dropped_row = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=masked,
        ref_log_prob=ref_log_prob,
        group_size=G,
        clip_ratio=0.2,
        use_grpo_reward=False,
    ).kl_term

    log_ratio = (log_prob - ref_log_prob).clamp(min=-20.0, max=20.0)
    r = torch.exp(log_ratio)
    kl_k3 = r - 1.0 - log_ratio
    # Mean of kl_k3 over all (B*T) tokens.
    expected_unmasked = kl_k3.mean()
    assert torch.allclose(kl_full, expected_unmasked, atol=1e-5)
    # When row 0 is fully masked, the mean is over the remaining (B-1)*T tokens.
    expected_masked = kl_k3[1:].mean()
    assert torch.allclose(kl_dropped_row, expected_masked, atol=1e-5)


def test_batch_size_must_be_divisible_by_group_size():
    log_prob, old_log_prob, advantages, ref_log_prob, response_mask = _build_inputs(B=7, T=4, G=4)
    with pytest.raises(ValueError, match="divisible by group_size"):
        compute_sharpening_grpo_loss(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            ref_log_prob=ref_log_prob,
            group_size=4,
            clip_ratio=0.2,
            use_grpo_reward=True,
        )


def test_gradients_flow_back_through_seq_and_kl_terms():
    log_prob, old_log_prob, advantages, ref_log_prob, response_mask = _build_inputs()
    out = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        ref_log_prob=ref_log_prob,
        group_size=4,
        clip_ratio=0.2,
        use_grpo_reward=True,
    )
    total = out.grpo_term + out.seq_term + out.kl_term
    total.backward()
    assert log_prob.grad is not None
    assert torch.isfinite(log_prob.grad).all()
    assert log_prob.grad.abs().sum().item() > 0.0


def test_entropy_rate_weight_is_finite_with_all_zero_mask():
    log_prob, old_log_prob, advantages, ref_log_prob, _ = _build_inputs()
    zero_mask = torch.zeros_like(log_prob, dtype=torch.bool)
    out = compute_sharpening_grpo_loss(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=zero_mask,
        ref_log_prob=ref_log_prob,
        group_size=4,
        clip_ratio=0.2,
        use_grpo_reward=True,
    )
    # Even with no valid tokens, the loss terms should be finite (the +1e-8
    # guard prevents division-by-zero; the mean is over zero tokens, giving 0).
    assert torch.isfinite(out.grpo_term).item()
    assert torch.isfinite(out.seq_term).item()
    assert torch.isfinite(out.kl_term).item()
