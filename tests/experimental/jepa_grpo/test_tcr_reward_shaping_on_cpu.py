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
"""CPU unit tests for jepa-tcr-dual teacher-alignment reward shaping (idea #2).

Targets the pure standardization helper `JEPARayPPOTrainer._stratified_shaping`,
which carries the load-bearing invariants:
  - within-(group, correctness) standardization (mean 0 per stratum),
  - global-shift invariance (the cos_correct≈cos_wrong shortcut earns 0),
  - independence across the correct/wrong boundary (never flips ground truth),
  - all-wrong / all-correct groups (zero Dr.GRPO advantage) still get a gradient,
  - singleton/degenerate strata contribute 0.
Pure CPU — no GPU, Ray, or model weights. Also checks config validation.
"""

from __future__ import annotations

import pytest
import torch

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.ray_trainer import JEPARayPPOTrainer

shape = JEPARayPPOTrainer._stratified_shaping


def test_within_stratum_zero_mean():
    # One group, all correct: standardized scores have mean 0 and are scaled by beta.
    s = torch.tensor([0.2, 0.4, 0.9])
    gids = ["u", "u", "u"]
    isc = torch.tensor([True, True, True])
    out, n = shape(s, gids, isc, beta=0.5, sigma_floor=0.1)
    assert n == 1
    assert out.shape == (3,)
    assert abs(float(out.mean())) < 1e-6           # centered
    assert out.argmax() == s.argmax()              # ordering preserved
    assert out.argmin() == s.argmin()


def test_all_wrong_group_gets_gradient():
    # Dr.GRPO gives 0 advantage to an all-wrong group; shaping must not.
    s = torch.tensor([0.1, 0.5, 0.8])
    gids = ["u", "u", "u"]
    isc = torch.tensor([False, False, False])
    out, n = shape(s, gids, isc, beta=0.5, sigma_floor=0.1)
    assert n == 1
    assert out.abs().sum() > 0                      # non-zero rescue signal
    assert out.argmax() == s.argmax()               # most-aligned wrong reinforced


def test_global_shift_invariance():
    # Adding a constant to every score in a stratum (the "shift everything toward
    # the target" shortcut) leaves the shaping unchanged.
    s = torch.tensor([0.1, 0.3, 0.7, 0.9])
    gids = ["u", "u", "u", "u"]
    isc = torch.tensor([True, True, True, True])
    a, _ = shape(s, gids, isc, beta=0.5, sigma_floor=0.05)
    # A realistic global shift (all latents drift toward the target direction).
    # Invariance is exact in real arithmetic; allow float32 round-off.
    b, _ = shape(s + 0.5, gids, isc, beta=0.5, sigma_floor=0.05)
    assert torch.allclose(a, b, atol=1e-4)


def test_strata_are_independent_across_correctness():
    # Mixed group: correct and wrong subsets are standardized separately, so each
    # subset is centered on its own — shaping never transfers mass across the boundary.
    s = torch.tensor([0.10, 0.20, 0.70, 0.90])
    gids = ["u", "u", "u", "u"]
    isc = torch.tensor([True, True, False, False])
    out, n = shape(s, gids, isc, beta=0.5, sigma_floor=0.1)
    assert n == 2
    assert abs(float(out[isc].mean())) < 1e-6        # correct subset centered
    assert abs(float(out[~isc].mean())) < 1e-6       # wrong subset centered


def test_singleton_stratum_zero():
    # A stratum with a single member cannot be re-ranked → contributes 0.
    s = torch.tensor([0.5, 0.2, 0.9])
    gids = ["u", "u", "v"]                            # "v" has one member
    isc = torch.tensor([True, False, True])
    out, n = shape(s, gids, isc, beta=0.5, sigma_floor=0.1)
    assert n == 0                                    # no stratum has >=2 members
    assert torch.allclose(out, torch.zeros(3))


def test_sigma_floor_caps_amplification():
    # Near-uniform scores: sigma_floor prevents tiny std from blowing up shaping.
    s = torch.tensor([0.500, 0.501])
    gids = ["u", "u"]
    isc = torch.tensor([False, False])
    out, _ = shape(s, gids, isc, beta=0.5, sigma_floor=0.1)
    # |ŝ| = |(s-mean)|/sigma_floor = 0.0005/0.1 = 0.005 → β·ŝ = 0.0025
    assert float(out.abs().max()) < 0.01


def test_config_validation_beta_and_floor():
    base = dict(enable=True, loss_type="jepa-tcr-dual",
                teacher_cache_path="/x.pt", code_teacher_cache_path="/y.pt", n_cot=4, n_code=4)
    JEPARayConfig.from_config({**base, "tcr_reward_beta": 0.5}).validate(8)  # ok
    with pytest.raises(ValueError):
        JEPARayConfig.from_config({**base, "tcr_reward_beta": -1.0}).validate(8)
    with pytest.raises(ValueError):
        JEPARayConfig.from_config({**base, "tcr_reward_sigma_floor": 0.0}).validate(8)
    with pytest.raises(ValueError):
        JEPARayConfig.from_config({**base, "teacher_cache_path": ""}).validate(8)
