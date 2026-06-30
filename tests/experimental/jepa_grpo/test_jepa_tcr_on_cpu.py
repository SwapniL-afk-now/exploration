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
"""CPU unit tests for the jepa-tcr-dual loss math and config.

Covers the dual-target self-consistent loss (align_cot + align_code + self-consistency
+ SIGReg): both-view alignment, stop-gradient on the self target, self-consistency
self-gating, per-arm latch (a disabled arm contributes 0 gradient but still feeds the
SIGReg pool), the all-arms-off pure-SIGReg case, and the `jepa_anchor_set` config field.
Pure CPU — no GPU, Ray, or model weights required.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.ray_trainer import JEPARayPPOTrainer

D = 16


def _unit(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, D, generator=g), dim=-1)


# --------------------------------------------------------------------------- config


def test_config_default_is_correct():
    assert JEPARayConfig().jepa_anchor_set == "correct"


@pytest.mark.parametrize("val", ["correct", "all", "wrong"])
def test_config_accepts_valid_anchor_sets(val):
    cfg = JEPARayConfig.from_config(
        {"enable": True, "loss_type": "jepa-tcr-dual", "teacher_cache_path": "/x",
         "code_teacher_cache_path": "/y", "n_cot": 4, "n_code": 4, "jepa_anchor_set": val}
    )
    cfg.validate(rollout_n=8)  # must not raise


def test_config_rejects_invalid_anchor_set():
    cfg = JEPARayConfig.from_config(
        {"enable": True, "loss_type": "jepa-tcr-dual", "teacher_cache_path": "/x",
         "code_teacher_cache_path": "/y", "n_cot": 4, "n_code": 4, "jepa_anchor_set": "bogus"}
    )
    with pytest.raises(ValueError, match="jepa_anchor_set"):
        cfg.validate(rollout_n=8)


def test_config_has_no_per_class_weight_fields():
    names = {f.name for f in __import__("dataclasses").fields(JEPARayConfig)}
    assert "correct_jepa_weight" not in names
    assert "wrong_jepa_weight" not in names


# --------------------------------------------------------------------------- dual loss


def test_dual_align_sums_both_views():
    from verl.experimental.jepa_grpo.core_algos import llm_jepa_tcr_dual_loss
    Ac, Ak, d = 4, 4, 8
    pc, zc = _unit(Ac, 1), _unit(Ac, 2)
    pk, zk = _unit(Ak, 3), _unit(Ak, 4)
    st = _unit(Ac, 5)
    sm = torch.zeros(Ac, dtype=torch.bool)  # no self pairs -> self term 0
    g = torch.tensor([0, 0, 1, 1]); ic = torch.tensor([True, False, True, False])
    _, m = llm_jepa_tcr_dual_loss(pc, zc, pk, zk, st, sm, g, ic, g, ic, lambda_=0.0, M=32)
    # with lambda=0 and self_loss=0, tcr_loss == align_cot + align_code
    assert m["jepa/n_self_pairs"] == 0
    assert abs(m["jepa/tcr_loss"] - (m["jepa/align_cot"] + m["jepa/align_code"])) < 1e-5


def test_dual_self_target_detached():
    from verl.experimental.jepa_grpo.core_algos import llm_jepa_tcr_dual_loss
    Ac, Ak, d = 3, 3, 8
    pc = _unit(Ac, 1).clone().requires_grad_(True)
    pk = _unit(Ak, 3).clone().requires_grad_(True)
    st = _unit(Ac, 5).clone().requires_grad_(True)
    sm = torch.tensor([True, True, False])
    g = torch.tensor([0, 0, 1]); ic = torch.tensor([True, False, True])
    loss, _ = llm_jepa_tcr_dual_loss(pc, _unit(Ac, 2), pk, _unit(Ak, 4), st, sm, g, ic, g, ic, M=32)
    loss.backward()
    assert st.grad is None             # self target is stop-grad
    assert pc.grad.abs().sum() > 0     # grad reaches CoT preds
    assert pk.grad.abs().sum() > 0     # grad reaches Code preds


def test_dual_self_consistency_gates_to_zero():
    from verl.experimental.jepa_grpo.core_algos import llm_jepa_tcr_dual_loss
    Ac, Ak = 4, 4
    args = (_unit(Ac, 1), _unit(Ac, 2), _unit(Ak, 3), _unit(Ak, 4), _unit(Ac, 5))
    g = torch.tensor([0, 0, 1, 1]); ic = torch.tensor([True, False, True, False])
    _, m = llm_jepa_tcr_dual_loss(*args, torch.zeros(Ac, dtype=torch.bool), g, ic, g, ic, M=32)
    assert m["jepa/self_consist_loss"] == 0.0 and m["jepa/n_self_pairs"] == 0


def test_dual_arm_off_zeros_grad_but_keeps_pool():
    # Per-arm plateau latch: align_code_on=False must drop the gradient on the code
    # align term, yet code preds still feed SIGReg (lambda>0) so pk keeps a gradient.
    from verl.experimental.jepa_grpo.core_algos import llm_jepa_tcr_dual_loss
    Ac, Ak = 3, 3
    pc = _unit(Ac, 1).clone().requires_grad_(True)
    pk = _unit(Ak, 3).clone().requires_grad_(True)
    st = _unit(Ac, 5)
    sm = torch.zeros(Ac, dtype=torch.bool)
    g = torch.tensor([0, 0, 1]); ic = torch.tensor([True, False, True])
    loss, m = llm_jepa_tcr_dual_loss(
        pc, _unit(Ac, 2), pk, _unit(Ak, 4), st, sm, g, ic, g, ic,
        align_code_on=False, lambda_=0.5, M=64,
    )
    loss.backward()
    assert m["jepa/arm_code_on"] == 0.0 and m["jepa/arm_cot_on"] == 1.0
    # align_code is still REPORTED (diagnostic) but contributes no gradient...
    assert m["jepa/align_code"] != 0.0
    # ...code preds still get gradient via the SIGReg pool, CoT preds via their align term.
    assert pk.grad.abs().sum() > 0
    assert pc.grad.abs().sum() > 0


def test_dual_all_arms_off_is_pure_sigreg():
    from verl.experimental.jepa_grpo.core_algos import llm_jepa_tcr_dual_loss
    Ac, Ak = 3, 3
    pc, pk, st = _unit(Ac, 1), _unit(Ak, 3), _unit(Ac, 5)
    sm = torch.zeros(Ac, dtype=torch.bool)
    g = torch.tensor([0, 0, 1]); ic = torch.tensor([True, False, True])
    loss, m = llm_jepa_tcr_dual_loss(
        pc, _unit(Ac, 2), pk, _unit(Ak, 4), st, sm, g, ic, g, ic,
        align_cot_on=False, align_code_on=False, self_on=False, lambda_=0.5, M=64,
    )
    # with every align arm off, tcr_loss == lambda * sigreg
    assert abs(m["jepa/tcr_loss"] - 0.5 * m["jepa/tcr_sigreg_loss"]) < 1e-5


# --------------------------------------------------------------------------- dual builder


def _fake_self_dual(teacher_targets, code_teacher_targets):
    return types.SimpleNamespace(
        teacher_targets=teacher_targets,
        code_teacher_targets=code_teacher_targets,
        tokenizer=types.SimpleNamespace(pad_token_id=0),
        jepa_cfg=types.SimpleNamespace(
            jepa_anchor_set="correct", tcr_match="cycle", min_valid_pairs=1
        ),
    )


def _fake_batch_dual(n_per_view=2, n_prompts=2, prompt_len=3):
    # Per prompt: n_per_view cot rows then n_per_view code rows.
    per_prompt = 2 * n_per_view
    rows = per_prompt * n_prompts
    ids = torch.arange(1, rows * prompt_len + 1).reshape(rows, prompt_len)
    mask = torch.ones(rows, prompt_len, dtype=torch.long)
    uids, extra, view = [], [], []
    for p in range(n_prompts):
        for _ in range(n_per_view):
            uids.append(f"p{p}"); extra.append({"index": p}); view.append("cot")
        for _ in range(n_per_view):
            uids.append(f"p{p}"); extra.append({"index": p}); view.append("code")
    batch = types.SimpleNamespace(
        batch={"input_ids": ids, "attention_mask": mask},
        non_tensor_batch={"uid": np.array(uids, dtype=object),
                          "extra_info": np.array(extra, dtype=object),
                          "view": np.array(view, dtype=object)},
    )
    return batch, np.array(view, dtype=object)


def test_dual_builder_emits_both_blocks_and_self_partner():
    batch, view = _fake_batch_dual(n_per_view=2, n_prompts=2)
    fake = _fake_self_dual({0: _unit(4, 30), 1: _unit(4, 31)},
                           {0: _unit(4, 32), 1: _unit(4, 33)})
    # all rows correct -> 2 cot + 2 code anchors per prompt
    reward = torch.ones(len(view), 1)
    out = JEPARayPPOTrainer._build_jepa_batch_tcr_dual(fake, batch, reward, view)
    is_code = out.batch["is_code"]
    Ac = int((~is_code).sum()); Ak = int(is_code.sum())
    assert Ac == 4 and Ak == 4
    # combined block: CoT rows first, then Code rows
    assert torch.equal(is_code, torch.tensor([False] * 4 + [True] * 4))
    assert out.batch["teacher_target"].shape[0] == Ac + Ak
    assert out.batch["self_partner"].shape[0] == Ac + Ak
    # every CoT row is paired (equal counts) -> CoT-row partners valid & in code range
    sp_cot = out.batch["self_partner"][~is_code]
    assert (sp_cot >= 0).all() and (sp_cot < Ak).all()
    # code rows carry no partner
    assert int((out.batch["self_partner"][is_code] == -1).sum()) == Ak
    assert out.meta_info["n_self_pairs"] == Ac


def test_dual_builder_unpaired_rows_have_no_partner():
    # prompt0: 2 cot correct, 1 code correct -> one cot row unpaired (-1)
    batch, view = _fake_batch_dual(n_per_view=2, n_prompts=1)
    fake = _fake_self_dual({0: _unit(4, 30)}, {0: _unit(4, 32)})
    # rows: cot,cot,code,code -> make 2nd code wrong so only 1 code anchor
    reward = torch.tensor([1, 1, 1, 0], dtype=torch.float32).reshape(-1, 1)
    out = JEPARayPPOTrainer._build_jepa_batch_tcr_dual(fake, batch, reward, view)
    is_code = out.batch["is_code"]
    assert int(is_code.sum()) == 1
    sp_cot = out.batch["self_partner"][~is_code]
    assert int((sp_cot == -1).sum()) == 1 and int((sp_cot >= 0).sum()) == 1


def test_dual_builder_missing_code_cache_drops_code_anchors():
    batch, view = _fake_batch_dual(n_per_view=2, n_prompts=2)
    # prompt1 has no CODE target -> its code rows contribute no code anchors
    fake = _fake_self_dual({0: _unit(4, 30), 1: _unit(4, 31)}, {0: _unit(4, 32)})
    reward = torch.ones(len(view), 1)
    out = JEPARayPPOTrainer._build_jepa_batch_tcr_dual(fake, batch, reward, view)
    is_code = out.batch["is_code"]
    # cot: both prompts (4); code: only prompt0 (2)
    assert int((~is_code).sum()) == 4
    assert int(is_code.sum()) == 2


# --------------------------------------------------------------------------- dual config


def test_config_dual_requires_both_caches_and_code():
    base = {"enable": True, "loss_type": "jepa-tcr-dual", "n_cot": 4, "n_code": 4,
            "teacher_cache_path": "/a", "code_teacher_cache_path": "/b"}
    JEPARayConfig.from_config(base).validate(rollout_n=8)  # ok
    with pytest.raises(ValueError, match="code_teacher_cache_path"):
        JEPARayConfig.from_config({**base, "code_teacher_cache_path": ""}).validate(rollout_n=8)
    with pytest.raises(ValueError, match="n_code"):
        JEPARayConfig.from_config({**base, "n_cot": 8, "n_code": 0}).validate(rollout_n=8)


def test_config_dual_requires_both_caches():
    base = {"enable": True, "loss_type": "jepa-tcr-dual", "n_cot": 4, "n_code": 4,
            "teacher_cache_path": "/a", "code_teacher_cache_path": "/b"}
    JEPARayConfig.from_config(base).validate(rollout_n=8)  # ok
    with pytest.raises(ValueError, match="code_teacher_cache_path"):
        JEPARayConfig.from_config({**base, "code_teacher_cache_path": ""}).validate(rollout_n=8)
    with pytest.raises(ValueError, match="teacher_cache_path"):
        JEPARayConfig.from_config({**base, "teacher_cache_path": ""}).validate(rollout_n=8)


# --------------------------------------------------------------------- dual per-view shaping routing


class _FakeBatch:
    def __init__(self, n, prompt_len, views, idxs):
        self.batch = {
            "input_ids": torch.arange(1, n * prompt_len + 1).reshape(n, prompt_len),
            "attention_mask": torch.ones(n, prompt_len, dtype=torch.long),
        }
        self.non_tensor_batch = {
            "uid": np.array([f"u{idxs[i]}" for i in range(n)], dtype=object),
            "extra_info": np.array([{"index": idxs[i]} for i in range(n)], dtype=object),
            "view": np.array(views, dtype=object),
        }

    def __len__(self):
        return self.batch["input_ids"].shape[0]


def _fake_shaping_self(teacher, code_teacher, emb):
    wg = types.SimpleNamespace(score_cot_embeddings=lambda td: {"cot_emb": emb})
    return types.SimpleNamespace(
        teacher_targets=teacher,
        code_teacher_targets=code_teacher,
        tokenizer=types.SimpleNamespace(pad_token_id=0),
        actor_rollout_wg=wg,
        jepa_cfg=types.SimpleNamespace(
            min_valid_pairs=2, tcr_reward_beta=0.5, tcr_reward_sigma_floor=0.1
        ),
        _stratified_shaping=JEPARayPPOTrainer._stratified_shaping,
    )


def test_reward_dual_scores_code_rows_via_code_cache():
    # Prompt index 7 is present ONLY in the CODE cache (NOT the CoT cache). Two code
    # rows for it must still be scored — proving code rows route to the code cache.
    n, d = 2, D
    views = ["code", "code"]
    idxs = [7, 7]
    batch = _FakeBatch(n, prompt_len=3, views=views, idxs=idxs)
    emb = F.normalize(torch.randn(n, d), dim=-1)
    fake = _fake_shaping_self(
        teacher={},                       # CoT cache: empty -> would skip if mis-routed
        code_teacher={7: F.normalize(torch.randn(2, d), dim=-1)},
        emb=emb,
    )
    reward = torch.tensor([1.0, 1.0]).reshape(-1, 1)  # both correct -> one stratum, shapeable
    shape_per_row, metrics = JEPARayPPOTrainer._compute_tcr_reward_shaping(
        fake, batch, reward, batch.non_tensor_batch["view"]
    )
    assert metrics["shaping/n_rows_scored"] == 2.0
    assert metrics["shaping/n_rows_scored_code"] == 2.0
    assert metrics["shaping/n_rows_scored_cot"] == 0.0


def test_reward_dual_cot_rows_skipped_when_only_code_cache_has_index():
    # Same prompt only in code cache; a COT row for it must be SKIPPED (CoT cache empty).
    batch = _FakeBatch(2, 3, ["cot", "code"], [7, 7])
    emb = F.normalize(torch.randn(2, D), dim=-1)
    fake = _fake_shaping_self(teacher={}, code_teacher={7: F.normalize(torch.randn(2, D), dim=-1)}, emb=emb)
    # only 1 scorable row (the code one) < min_valid_pairs=2 -> early return, nothing scored
    reward = torch.tensor([1.0, 1.0]).reshape(-1, 1)
    _, metrics = JEPARayPPOTrainer._compute_tcr_reward_shaping(
        fake, batch, reward, batch.non_tensor_batch["view"]
    )
    assert metrics["shaping/n_rows_scored"] == 0.0  # only code row eligible, below min_valid_pairs


# --------------------------------------------------------------------- auto-off plateau latch


def _fake_autooff_self(global_steps=100, **cfg_over):
    cfg = dict(enable=True, auto_off_enable=True, auto_off_metric="m",
               auto_off_patience=3, auto_off_min_delta=0.01, auto_off_warmup_steps=0,
               loss_type="jepa-tcr-dual")
    cfg.update(cfg_over)
    arms = ("cot", "code", "self", "shaping", "global")
    fake = types.SimpleNamespace(
        jepa_cfg=types.SimpleNamespace(**cfg),
        global_steps=global_steps,
        _jepa_signal_off=False,
        _off_arm={k: False for k in arms},
        _align_best={k: float("-inf") for k in arms},
        _align_stall={k: 0 for k in arms},
    )
    fake._tracked_signals = types.MethodType(JEPARayPPOTrainer._tracked_signals, fake)
    return fake


def _step(fake, val):
    m = {"m": val}
    JEPARayPPOTrainer._maybe_disable_jepa_signal(fake, m)
    return m


def test_autooff_latches_after_plateau():
    f = _fake_autooff_self()
    _step(f, 0.50)                 # best=0.50, stall=0
    assert not f._jepa_signal_off
    _step(f, 0.50)                 # stall 1
    _step(f, 0.505)                # < min_delta improvement -> stall 2
    assert not f._jepa_signal_off
    m = _step(f, 0.50)            # stall 3 == patience -> latch
    assert f._jepa_signal_off and m["jepa/signal_off"] == 1.0


def test_autooff_improvement_resets_stall():
    f = _fake_autooff_self()
    _step(f, 0.50); _step(f, 0.50); _step(f, 0.50)   # stall 2
    _step(f, 0.60)                                     # big jump -> reset
    assert f._align_stall["global"] == 0 and not f._jepa_signal_off


def test_autooff_respects_warmup_and_disabled():
    # before warmup: never evaluates
    f = _fake_autooff_self(global_steps=5, auto_off_warmup_steps=20)
    for _ in range(10):
        _step(f, 0.0001)
    assert not f._jepa_signal_off
    # disabled entirely: no-op
    g = _fake_autooff_self(auto_off_enable=False)
    for _ in range(10):
        _step(g, 0.0)
    assert not g._jepa_signal_off


def test_autooff_dual_arms_latch_independently():
    # jepa-tcr-dual tracks cos_cot/cos_code/cos_self separately. Plateau cos_code only;
    # its arm must latch while cot/self stay live and the global signal stays ON.
    f = _fake_autooff_self(auto_off_metric="", loss_type="jepa-tcr-dual")

    def step(cot, code, slf):
        m = {"jepa/cos_cot": cot, "jepa/cos_code": code, "jepa/cos_self": slf}
        JEPARayPPOTrainer._maybe_disable_jepa_signal(f, m)
        return m

    # cot & self keep improving; code is flat from the first measurement.
    step(0.30, 0.30, 0.30)
    for i in range(3):
        step(0.40 + 0.05 * i, 0.30, 0.40 + 0.05 * i)
    assert f._off_arm["code"] and not f._off_arm["cot"] and not f._off_arm["self"]
    assert not f._jepa_signal_off          # global stays on until ALL three latch
    m = step(0.60, 0.30, 0.70)
    assert m["jepa/off_code"] == 1.0 and m["jepa/off_cot"] == 0.0


def test_autooff_ignores_zero_metric_steps():
    f = _fake_autooff_self()
    _step(f, 0.50)                 # best=0.50
    for _ in range(10):
        _step(f, 0.0)             # no anchors -> ignored, stall must not advance
    assert f._align_stall["global"] == 0 and not f._jepa_signal_off
