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

import math

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.exploration import EMALoRATracker, ExplorationConfig, StableExplorationDivergence
from verl.utils import tensordict_utils as tu
from verl.workers.config import ActorConfig
from verl.workers.utils.losses import ppo_loss


class TinyLoRAModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(2, 2, bias=False)
        self.lora_A = torch.nn.Linear(2, 2, bias=False)
        self.lora_B = torch.nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)


def make_tracker(gamma=0.5, update_every=2):
    module = TinyLoRAModule()
    with torch.no_grad():
        module.lora_A.weight.fill_(1.0)
        module.lora_B.weight.fill_(2.0)
    return module, EMALoRATracker(module, gamma=gamma, update_every=update_every)


def test_ema_tracker_detaches_and_updates_on_interval():
    module, tracker = make_tracker(gamma=0.5, update_every=2)

    assert tracker.has_params
    assert all(not param.requires_grad for param in tracker.ema_params.values())

    with torch.no_grad():
        module.lora_A.weight.fill_(3.0)
    assert not tracker.update(module)
    torch.testing.assert_close(tracker.ema_params["lora_A.weight"], torch.ones_like(module.lora_A.weight))

    assert tracker.update(module)
    expected = torch.full_like(module.lora_A.weight, 2.0)
    torch.testing.assert_close(tracker.ema_params["lora_A.weight"], expected)


def test_ema_tracker_hot_swap_restores_after_exception():
    module, tracker = make_tracker()
    original = module.lora_A.weight.detach().clone()
    with torch.no_grad():
        tracker.ema_params["lora_A.weight"].fill_(7.0)

    saved = tracker.load_into_model(module)
    try:
        assert torch.equal(module.lora_A.weight, torch.full_like(module.lora_A.weight, 7.0))
        raise RuntimeError("boom")
    except RuntimeError:
        tracker.restore_model(module, saved)

    torch.testing.assert_close(module.lora_A.weight, original)


def make_sed(warmup_steps=0, delta=3.0, alpha_max=0.3):
    module, tracker = make_tracker()
    config = ExplorationConfig(enabled=True, warmup_steps=warmup_steps, delta=delta, alpha_max=alpha_max)
    return StableExplorationDivergence(config=config, ema_tracker=tracker)


def test_sed_disabled_and_warmup_are_zero_loss():
    current = torch.zeros(2, 2, requires_grad=True)
    ema = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)

    disabled = StableExplorationDivergence(config=ExplorationConfig(enabled=False))
    loss, metrics = disabled.compute_loss(
        current_log_probs=current, ema_log_probs=ema, entropy=None, response_mask=mask
    )
    assert loss.item() == 0.0
    assert metrics["explore/status"].item() == StableExplorationDivergence.STATUS_DISABLED

    warmup = make_sed(warmup_steps=10)
    loss, metrics = warmup.compute_loss(current_log_probs=current, ema_log_probs=None, entropy=None, response_mask=mask)
    assert loss.item() == 0.0
    assert metrics["explore/status"].item() == StableExplorationDivergence.STATUS_WARMUP


def test_sed_reversed_kl_formula_and_loss():
    sed = make_sed(warmup_steps=0, delta=10.0, alpha_max=0.5)
    current = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    ema = torch.tensor([[-0.5, -1.5]])
    entropy = torch.ones_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)

    loss, metrics = sed.compute_loss(
        current_log_probs=current, ema_log_probs=ema, entropy=entropy, response_mask=mask
    )

    per_token = torch.exp(ema) * (ema - current.detach())
    expected_kl = per_token.mean()
    expected_alpha = 0.5 * (1.0 - expected_kl.item() / 10.0)
    torch.testing.assert_close(metrics["explore/raw_kl"], expected_kl)
    torch.testing.assert_close(loss, -expected_alpha * expected_kl)


def test_sed_clamp_zeroes_gradient_at_boundary():
    sed = make_sed(warmup_steps=0, delta=1.0, alpha_max=0.5)
    current = torch.full((1, 2), -100.0, requires_grad=True)
    ema = torch.zeros_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)

    loss, metrics = sed.compute_loss(current_log_probs=current, ema_log_probs=ema, entropy=None, response_mask=mask)
    loss.backward()

    assert metrics["explore/status"].item() == StableExplorationDivergence.STATUS_CLAMPED_DORMANT
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_sed_entropy_gate_closes_below_threshold():
    sed = make_sed(warmup_steps=0, delta=10.0)
    sed.initial_entropy = torch.tensor(4.0)
    current = torch.tensor([[-1.0, -1.0]], requires_grad=True)
    ema = torch.tensor([[-0.5, -0.5]])
    entropy = torch.ones_like(current)
    mask = torch.ones_like(current, dtype=torch.bool)

    loss, metrics = sed.compute_loss(
        current_log_probs=current, ema_log_probs=ema, entropy=entropy, response_mask=mask
    )

    assert loss.item() == 0.0
    assert metrics["explore/entropy_gate"].item() == 0.0
    assert metrics["explore/status"].item() == StableExplorationDivergence.STATUS_ENTROPY_GATED


def test_sed_clamps_negative_infinity_ema_log_probs():
    sed = make_sed(warmup_steps=0, delta=10.0)
    current = torch.zeros(1, 2, requires_grad=True)
    ema = torch.full_like(current, -math.inf)
    mask = torch.ones_like(current, dtype=torch.bool)

    loss, metrics = sed.compute_loss(current_log_probs=current, ema_log_probs=ema, entropy=None, response_mask=mask)

    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["explore/raw_kl"])


def make_actor_config():
    return ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)


def make_ppo_data():
    data = TensorDict(
        {
            "prompts": torch.tensor([[1], [2]]),
            "responses": torch.tensor([[3, 4], [5, 6]]),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "response_mask": torch.ones(2, 2, dtype=torch.bool),
            "old_log_probs": torch.zeros(2, 2),
            "advantages": torch.zeros(2, 2),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)
    return data


def test_ppo_loss_matches_stock_when_exploration_not_passed():
    config = make_actor_config()
    data = make_ppo_data()
    model_output = {"log_probs": torch.zeros(6)}

    stock_loss, stock_metrics = ppo_loss(config=config, model_output=model_output, data=data.clone())
    disabled_loss, disabled_metrics = ppo_loss(config=config, model_output=model_output, data=data.clone())

    torch.testing.assert_close(disabled_loss, stock_loss)
    assert set(disabled_metrics) == set(stock_metrics)


def test_ppo_loss_adds_expected_exploration_loss():
    config = make_actor_config()
    data = make_ppo_data()
    sed = make_sed(warmup_steps=0, delta=10.0, alpha_max=0.5)
    model_output = {
        "log_probs": torch.zeros(6, requires_grad=True),
        "ema_log_probs": torch.full((6,), 0.25),
        "entropy": torch.ones(6),
    }

    base_loss, _ = ppo_loss(config=config, model_output={"log_probs": model_output["log_probs"]}, data=data.clone())
    total_loss, metrics = ppo_loss(config=config, model_output=model_output, data=data.clone(), exploration=sed)

    expected_explore_loss = metrics["explore/loss"].values[0]
    torch.testing.assert_close(total_loss, base_loss + torch.tensor(expected_explore_loss))


def test_actor_config_exploration_default_is_disabled():
    config = make_actor_config()
    assert config.exploration.enabled is False
