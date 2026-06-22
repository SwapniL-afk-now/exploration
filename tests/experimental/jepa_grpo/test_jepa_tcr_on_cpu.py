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
"""CPU unit tests for the JEPA-TCR loss math, anchor selection, and config.

Covers the reward-stratified, prompt-averaged alignment loss (correct/wrong/all
anchor sets), wrong-count invariance, the back-compat flat-mean fallback,
stop-gradient on the teacher target, gradient flow to the student prediction, the
`_build_jepa_batch_tcr` anchor-selection modes, and the `jepa_anchor_set` config
field. Pure CPU — no GPU, Ray, or model weights required.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.core_algos import llm_jepa_tcr_loss
from verl.experimental.jepa_grpo.ray_trainer import JEPARayPPOTrainer

D = 16


def _unit(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, D, generator=g), dim=-1)


def _ell(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return 1.0 - (pred * tgt).sum(dim=-1)


def _manual_align(pred, tgt, group, corr):
    """Reference reward-stratified, prompt-averaged align (independent reimplementation)."""
    ell = _ell(pred, tgt)
    per_prompt = []
    for g in group.unique():
        gm = group == g
        c = ell[gm & corr]
        w = ell[gm & ~corr]
        if c.numel() and w.numel():
            per_prompt.append(0.5 * c.mean() + 0.5 * w.mean())
        elif c.numel():
            per_prompt.append(c.mean())
        elif w.numel():
            per_prompt.append(w.mean())
    return torch.stack(per_prompt).mean().item()


# --------------------------------------------------------------------------- loss


def test_both_class_group_matches_manual():
    pred, tgt = _unit(8, 1), _unit(8, 2)
    group = torch.zeros(8, dtype=torch.long)
    corr = torch.tensor([True, True, True] + [False] * 5)
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    assert abs(m["jepa/loss_align"] - _manual_align(pred, tgt, group, corr)) < 1e-5
    assert m["jepa/num_anchors_correct"] == 3 and m["jepa/num_anchors_wrong"] == 5


def test_correct_only_group():
    pred, tgt = _unit(4, 3), _unit(4, 4)
    group = torch.zeros(4, dtype=torch.long)
    corr = torch.ones(4, dtype=torch.bool)
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    assert abs(m["jepa/loss_align"] - _ell(pred, tgt).mean().item()) < 1e-5
    assert m["jepa/num_anchors_wrong"] == 0


def test_wrong_only_group():
    pred, tgt = _unit(5, 5), _unit(5, 6)
    group = torch.zeros(5, dtype=torch.long)
    corr = torch.zeros(5, dtype=torch.bool)
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    assert abs(m["jepa/loss_align"] - _ell(pred, tgt).mean().item()) < 1e-5
    assert m["jepa/num_anchors_correct"] == 0


def test_empty_is_safe():
    pred = torch.zeros(0, D)
    loss, m = llm_jepa_tcr_loss(
        pred, pred, pred,
        group_id=torch.zeros(0, dtype=torch.long),
        is_correct=torch.zeros(0, dtype=torch.bool), M=32,
    )
    assert torch.isfinite(loss) and m["jepa/loss_align"] == 0.0
    assert m["jepa/num_anchors_total"] == 0


def test_one_correct_seven_wrong_formula():
    # group 0: 1 correct + 7 wrong -> 0.5*ell_c + 0.5*mean(7 wrong)
    pred, tgt = _unit(8, 7), _unit(8, 8)
    group = torch.zeros(8, dtype=torch.long)
    corr = torch.tensor([True] + [False] * 7)
    ell = _ell(pred, tgt)
    expected = (0.5 * ell[0] + 0.5 * ell[1:].mean()).item()
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    assert abs(m["jepa/loss_align"] - expected) < 1e-5


def test_seven_correct_one_wrong_formula():
    pred, tgt = _unit(8, 9), _unit(8, 10)
    group = torch.zeros(8, dtype=torch.long)
    corr = torch.tensor([True] * 7 + [False])
    ell = _ell(pred, tgt)
    expected = (0.5 * ell[:7].mean() + 0.5 * ell[7]).item()
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    assert abs(m["jepa/loss_align"] - expected) < 1e-5


@pytest.mark.parametrize("n_wrong", [1, 7, 20])
def test_wrong_count_invariance(n_wrong):
    # One prompt: fixed correct anchor + N identical wrong anchors. The group
    # contribution must be 0.5*ell_c + 0.5*ell_w regardless of N.
    pc, tc = _unit(1, 11), _unit(1, 12)
    pw, tw = _unit(1, 13), _unit(1, 14)
    pred = torch.cat([pc] + [pw] * n_wrong)
    tgt = torch.cat([tc] + [tw] * n_wrong)
    group = torch.zeros(n_wrong + 1, dtype=torch.long)
    corr = torch.tensor([True] + [False] * n_wrong)
    expected = (0.5 * _ell(pc, tc)[0] + 0.5 * _ell(pw, tw)[0]).item()
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    assert abs(m["jepa/loss_align"] - expected) < 1e-5


def test_flat_fallback_matches_old_mean():
    pred, tgt = _unit(10, 15), _unit(10, 16)
    _, m = llm_jepa_tcr_loss(pred, tgt, pred, M=32)  # no group_id/is_correct
    assert abs(m["jepa/loss_align"] - _ell(pred, tgt).mean().item()) < 1e-5


def test_pred_equals_target_near_zero_align():
    pe = _unit(9, 17)
    group = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    corr = torch.tensor([True, False, False, True, False, True, True, False, False])
    _, m = llm_jepa_tcr_loss(pe, pe, pe, group_id=group, is_correct=corr, M=32)
    assert m["jepa/loss_align"] < 1e-5


def test_target_detached_no_grad():
    pred = _unit(6, 18).requires_grad_(True)
    tgt = _unit(6, 19).requires_grad_(True)
    group = torch.tensor([0, 0, 0, 1, 1, 1])
    corr = torch.tensor([True, False, False, True, True, False])
    loss, _ = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    loss.backward()
    assert pred.grad is not None and pred.grad.abs().sum() > 0
    assert tgt.grad is None or tgt.grad.abs().sum() == 0


def test_gradient_reaches_every_anchor():
    pred = _unit(8, 20).requires_grad_(True)
    tgt = _unit(8, 21)
    group = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1])  # both-class + wrong-only group
    corr = torch.tensor([True, False, False, False, False, False, False, False])
    loss, _ = llm_jepa_tcr_loss(pred, tgt, pred, group_id=group, is_correct=corr, M=32)
    loss.backward()
    per_row = pred.grad.abs().sum(dim=-1)
    assert (per_row > 0).all(), per_row


# --------------------------------------------------------------------- batch builder


def _fake_self(anchor_set: str, teacher_targets: dict):
    return types.SimpleNamespace(
        teacher_targets=teacher_targets,
        tokenizer=types.SimpleNamespace(pad_token_id=0),
        jepa_cfg=types.SimpleNamespace(
            jepa_anchor_set=anchor_set, tcr_match="cycle", min_valid_pairs=1
        ),
    )


def _fake_batch(n_per_prompt=4, n_prompts=2, prompt_len=3):
    # rollouts ordered prompt-major; uid groups them; reward set per row below.
    rows = n_per_prompt * n_prompts
    ids = torch.arange(1, rows * prompt_len + 1).reshape(rows, prompt_len)
    mask = torch.ones(rows, prompt_len, dtype=torch.long)
    uids = np.array([f"p{p}" for p in range(n_prompts) for _ in range(n_per_prompt)], dtype=object)
    extra = np.array([{"index": p} for p in range(n_prompts) for _ in range(n_per_prompt)], dtype=object)
    view = np.array(["cot"] * rows, dtype=object)
    batch = types.SimpleNamespace(
        batch={"input_ids": ids, "attention_mask": mask},
        non_tensor_batch={"uid": uids, "extra_info": extra, "view": view},
    )
    return batch, view, uids


def _call_builder(anchor_set, reward_rows, targets_by_index, n_per_prompt=4, n_prompts=2):
    batch, view, _ = _fake_batch(n_per_prompt, n_prompts)
    fake = _fake_self(anchor_set, {k: v for k, v in targets_by_index.items()})
    reward = torch.tensor(reward_rows, dtype=torch.float32).reshape(-1, 1)
    return JEPARayPPOTrainer._build_jepa_batch_tcr(fake, batch, reward, view)


def test_builder_correct_selects_only_positive_reward():
    # prompt0: rows [1,0,0,1]; prompt1: [0,0,1,0]
    rewards = [1, 0, 0, 1, 0, 0, 1, 0]
    tgt = {0: _unit(4, 30), 1: _unit(4, 31)}
    out = _call_builder("correct", rewards, tgt)
    assert bool(out.batch["anchor_is_correct"].all())
    assert out.batch["anchor_is_correct"].numel() == 3  # 2 + 1


def test_builder_wrong_selects_only_nonpositive_reward():
    rewards = [1, 0, 0, 1, 0, 0, 1, 0]
    tgt = {0: _unit(4, 30), 1: _unit(4, 31)}
    out = _call_builder("wrong", rewards, tgt)
    assert not bool(out.batch["anchor_is_correct"].any())
    assert out.batch["anchor_is_correct"].numel() == 5  # 2 + 3


def test_builder_all_selects_both():
    rewards = [1, 0, 0, 1, 0, 0, 1, 0]
    tgt = {0: _unit(4, 30), 1: _unit(4, 31)}
    out = _call_builder("all", rewards, tgt)
    assert out.batch["anchor_is_correct"].numel() == 8
    assert int(out.batch["anchor_is_correct"].sum()) == 3


def test_builder_skips_prompts_without_targets():
    rewards = [1, 1, 1, 1, 1, 1, 1, 1]
    tgt = {0: _unit(4, 30)}  # prompt 1 has NO cached target -> skipped
    out = _call_builder("all", rewards, tgt)
    # only prompt0's 4 anchors survive, all group_id == 0
    assert out.batch["anchor_group_id"].unique().tolist() == [0]
    assert out.batch["anchor_is_correct"].numel() == 4


def test_builder_lengths_aligned_and_is_correct_matches_reward():
    rewards = [1, 0, 1, 0, 0, 1, 0, 0]
    tgt = {0: _unit(4, 30), 1: _unit(4, 31)}
    out = _call_builder("all", rewards, tgt)
    A = out.batch["cot_input_ids"].shape[0]
    assert out.batch["cot_attn_mask"].shape[0] == A
    assert out.batch["cot_lengths"].shape[0] == A
    assert out.batch["teacher_target"].shape[0] == A
    assert out.batch["anchor_group_id"].shape[0] == A
    assert out.batch["anchor_is_correct"].shape[0] == A
    # group ids distinguish the two prompts, not individual rows
    assert out.batch["anchor_group_id"].unique().tolist() == [0, 1]
    expected_correct = torch.tensor([bool(r > 0) for r in rewards])
    assert torch.equal(out.batch["anchor_is_correct"], expected_correct)


def test_builder_returns_none_below_min_valid_pairs():
    batch, view, _ = _fake_batch()
    fake = _fake_self("correct", {0: _unit(4, 30), 1: _unit(4, 31)})
    fake.jepa_cfg.min_valid_pairs = 100
    reward = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32).reshape(-1, 1)
    assert JEPARayPPOTrainer._build_jepa_batch_tcr(fake, batch, reward, view) is None


# --------------------------------------------------------------------------- config


def test_config_default_is_correct():
    assert JEPARayConfig().jepa_anchor_set == "correct"


@pytest.mark.parametrize("val", ["correct", "all", "wrong"])
def test_config_accepts_valid_anchor_sets(val):
    cfg = JEPARayConfig.from_config(
        {"enable": True, "loss_type": "jepa-tcr-loss", "teacher_cache_path": "/x",
         "n_cot": 8, "n_code": 0, "jepa_anchor_set": val}
    )
    cfg.validate(rollout_n=8)  # must not raise


def test_config_rejects_invalid_anchor_set():
    cfg = JEPARayConfig.from_config(
        {"enable": True, "loss_type": "jepa-tcr-loss", "teacher_cache_path": "/x",
         "n_cot": 8, "n_code": 0, "jepa_anchor_set": "bogus"}
    )
    with pytest.raises(ValueError, match="jepa_anchor_set"):
        cfg.validate(rollout_n=8)


def test_config_has_no_per_class_weight_fields():
    names = {f.name for f in __import__("dataclasses").fields(JEPARayConfig)}
    assert "correct_jepa_weight" not in names
    assert "wrong_jepa_weight" not in names
