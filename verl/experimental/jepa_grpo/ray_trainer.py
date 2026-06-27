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
"""JEPA-GRPO Ray trainer.

Extends RayPPOTrainer with a two-view training loop:
  1. CoT rollout → standard Dr.GRPO update (via parent's update_actor)
  2. Code rollout → JEPA alignment update (via jepa_update on the worker)

The Code view uses the same vLLM rollout infrastructure (hybrid engine) as
the CoT view; only the system prompt differs.  JEPA embeddings are computed
on the actor worker itself so gradients are never shipped over Ray.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.fepo.math_parser import compute_math_reward
from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.metric import reduce_metrics
from verl.utils.profiler.performance import simple_timer

logger = logging.getLogger(__name__)


class JEPARayPPOTrainer(RayPPOTrainer):
    """Ray PPO trainer augmented with LeJEPA representation alignment.

    The training step becomes:
        1. CoT rollout       (vLLM, standard)
        2. CoT reward        (math string match)
        3. CoT advantages    (Dr.GRPO / GRPO)
        4. GRPO update       (parent._update_actor)
        5. Code rollout      (vLLM, code system prompt)
        6. Code reward       (math string match on extracted answer)
        7. Build JEPA pairs  (prompts correct in both views)
        8. JEPA update       (worker.jepa_update — separate backward)
        9. EMA sync          (happens inside worker.jepa_update)
       10. Weight sync to rollout
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jepa_cfg = JEPARayConfig.from_config(self.config.get("jepa", {}))
        self.jepa_cfg.validate(self.config.actor_rollout_ref.rollout.n)

        # jepa-tcr-loss: load the offline teacher-target cache once. Keyed by the
        # dataset row index (extra_info["index"]); each value is a (n_i, d) tensor
        # of L2-normalized teacher-correct target embeddings in student space.
        self.teacher_targets: dict[int, torch.Tensor] | None = None
        if self.jepa_cfg.enable and self.jepa_cfg.loss_type in (
            "jepa-tcr-loss", "jepa-tcr-reward", "jepa-tcr-hybrid", "jepa-tcr-dual",
            "jepa-tcr-reward-dual"
        ):
            raw = torch.load(self.jepa_cfg.teacher_cache_path, map_location="cpu")
            self.teacher_targets = {
                int(k): v.float() for k, v in raw.items() if v is not None and v.numel() > 0
            }

        # Dual modes: a SECOND cache of Code-view teacher targets (coder model
        # solutions encoded by the same frozen reference), same {index: (n_i, d)}
        # format and shared 1536-d student space. Used by both the differentiable
        # jepa-tcr-dual and the reward-shaping jepa-tcr-reward-dual.
        # Plateau-latch state for jepa.auto_off_enable (see _maybe_disable_jepa_signal).
        # Per-arm plateau latches. Each tracked signal (cot/code/self for jepa-tcr-dual,
        # or a single shaping/global signal for the reward modes) plateaus and latches
        # INDEPENDENTLY, disabling only its own loss arm (and matching-view shaping).
        # `_jepa_signal_off` (global) latches True only once EVERY tracked signal is off,
        # at which point the whole JEPA block is skipped to save the forward.
        self._jepa_signal_off = False
        self._off_arm: dict[str, bool] = {k: False for k in ("cot", "code", "self", "shaping", "global")}
        self._align_best: dict[str, float] = {k: float("-inf") for k in self._off_arm}
        self._align_stall: dict[str, int] = {k: 0 for k in self._off_arm}

        self.code_teacher_targets: dict[int, torch.Tensor] | None = None
        if self.jepa_cfg.enable and self.jepa_cfg.loss_type in ("jepa-tcr-dual", "jepa-tcr-reward-dual"):
            raw_code = torch.load(self.jepa_cfg.code_teacher_cache_path, map_location="cpu")
            self.code_teacher_targets = {
                int(k): v.float() for k, v in raw_code.items() if v is not None and v.numel() > 0
            }

    # ------------------------------------------------------ worker setup ----
    def init_workers(self):
        super().init_workers()
        if self.jepa_cfg.enable:
            import dataclasses

            if self.jepa_cfg.predictor_k > 0:
                self.jepa_cfg.predictor_token_id = self._resolve_predictor_token_id()

            cfg_dict = dataclasses.asdict(self.jepa_cfg)
            self.actor_rollout_wg.jepa_init(cfg_dict)

    def _resolve_predictor_token_id(self) -> int:
        """Pick a token id to use for the LLM-JEPA tied-weight predictor (paper §3.1).

        The paper introduces a literal new [PRED] token. We avoid resizing the
        embedding matrix (which would also have to be threaded through LoRA)
        by instead reusing an existing, otherwise-unused token:
          1. Prefer an unused reserved/special token already in the tokenizer's
             vocab (e.g. Qwen-style `<|extra_0|>`...) that isn't part of the
             active chat template — the model has trained (if rarely-used)
             embeddings for these and no architecture change is needed.
          2. Fall back to pad_token_id (or eos_token_id if no pad token) — a
             deliberate simplification, documented here rather than the paper's
             literal new-token approach.
        """
        tok = self.tokenizer
        for candidate in getattr(tok, "additional_special_tokens", []) or []:
            cid = tok.convert_tokens_to_ids(candidate)
            if cid is not None and cid != tok.unk_token_id:
                return int(cid)
        if tok.pad_token_id is not None:
            return int(tok.pad_token_id)
        return int(tok.eos_token_id)

    # ------------------------------------------- code-view tokenisation -----
    def _tokenize_code_prompts(self, batch: DataProto) -> DataProto:
        """Build code-view gen_batch by replacing the system prompt.

        The AgentLoop handles tokenization internally via apply_chat_template;
        we only need to supply raw_prompt (list of messages) in non_tensor_batch.
        """
        code_sys = self.jepa_cfg.code_system_prompt

        raw_problems = [
            info["problem"] if isinstance(info, dict) else str(info)
            for info in batch.non_tensor_batch["extra_info"]
        ]

        messages_list = np.array(
            [
                [
                    {"role": "system", "content": code_sys},
                    {"role": "user", "content": problem},
                ]
                for problem in raw_problems
            ],
            dtype=object,
        )

        # Build a DataProto with a dummy tensor so DataProto has a known batch size.
        bsz = len(raw_problems)
        dummy = torch.zeros(bsz, 1, dtype=torch.uint8)
        code_batch = DataProto.from_single_dict({"dummy_tensor": dummy})
        # AgentLoop looks for raw_prompt in non_tensor_batch
        code_batch.non_tensor_batch["raw_prompt"] = messages_list
        # Copy metadata needed by the reward fn
        code_batch.non_tensor_batch.update({
            k: v for k, v in batch.non_tensor_batch.items()
            if k in {"uid", "extra_info", "data_source", "reward_model"}
        })
        code_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        code_batch.meta_info["global_steps"] = batch.meta_info.get("global_steps", 0)
        return code_batch

    # --------------------------------------- JEPA batch construction --------
    @staticmethod
    def _extract_answer(response: str) -> str:
        """Extract the last printed number/expression from a Python code response."""
        import re
        # Look for boxed answer first (model may still produce it)
        m = re.search(r"\\boxed\{([^}]+)\}", response)
        if m:
            return m.group(1).strip()
        # Otherwise take the last line that looks numeric
        for line in reversed(response.strip().splitlines()):
            line = line.strip()
            if line and re.match(r"^-?\d", line):
                return line
        return response.strip()

    @staticmethod
    def _group_rows_by_uid(
        uids: np.ndarray, view_tags: np.ndarray, rew: torch.Tensor
    ) -> tuple[dict, dict, list]:
        """Group row indices by (uid, view), preserving first-seen prompt order.

        Replaces the old `flat_idx = p_idx * rollout_n + g` stride arithmetic,
        which assumed two SEPARATE, fixed-stride rollout_n-sized batches. Now
        that cot+code rows live in one combined batch (built via
        `DataProto.concat`, not positional interleaving — see fit()), the
        only thing that ties a prompt's rows together is a shared `uid`.

        Returns (cot_by_uid, code_by_uid, valid_uids) where valid_uids is the
        ordered list of uids with >=1 correct cot row AND >=1 correct code row.
        """
        cot_by_uid: dict = defaultdict(list)
        code_by_uid: dict = defaultdict(list)
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v == "cot":
                cot_by_uid[u].append(i)
            elif v == "code":
                code_by_uid[u].append(i)

        valid_uids = []
        for u in dict.fromkeys(uids):  # dedup, preserves first-seen order
            cot_idxs = cot_by_uid.get(u, [])
            code_idxs = code_by_uid.get(u, [])
            if cot_idxs and code_idxs and (rew[cot_idxs] > 0).any() and (rew[code_idxs] > 0).any():
                valid_uids.append(u)
        return cot_by_uid, code_by_uid, valid_uids

    def _build_jepa_batch(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build per-prompt JEPA pairs from a combined CoT+Code rollout batch.

        Only prompts where at least one CoT rollout AND at least one Code
        rollout are correct are included.

        The CoT view embedding uses the CoT PROMPT tokens (same for all
        rollouts of the same prompt).  The Code view embedding uses the full
        Code sequence (prompt + first correct response).

        Returns None if fewer than min_valid_pairs pairs exist.
        """
        uids = batch.non_tensor_batch["uid"]
        rew = reward_tensor.sum(dim=-1)  # (B,) summed over token dim

        cot_by_uid, code_by_uid, valid_uids = self._group_rows_by_uid(uids, view_tags, rew)

        if len(valid_uids) < self.jepa_cfg.min_valid_pairs:
            return None

        # -- CoT view: encode the CoT PROMPT tokens of each valid prompt --
        # Take the first cot rollout's input_ids up to the prompt (response_mask tells us)
        cot_input_ids = batch.batch["input_ids"]        # (B, L)
        response_mask = batch.batch.get("response_mask", None)

        cot_prompt_ids_list = []
        cot_prompt_mask_list = []
        cot_prompt_lengths = []
        for u in valid_uids:
            flat_idx = cot_by_uid[u][0]  # first cot rollout for this prompt
            ids = cot_input_ids[flat_idx]  # (L,)
            attn = batch.batch["attention_mask"][flat_idx]  # (L,)

            if response_mask is not None:
                # Prompt = positions where response_mask == 0 AND attention_mask == 1
                prompt_end = int((response_mask[flat_idx] == 0).sum())
                ids_p = ids[:prompt_end]
                attn_p = attn[:prompt_end]
            else:
                ids_p = ids
                attn_p = attn

            cot_prompt_ids_list.append(ids_p)
            cot_prompt_mask_list.append(attn_p)
            cot_prompt_lengths.append(int(attn_p.sum()))

        # Pad to same length
        cot_max_len = max(t.shape[0] for t in cot_prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id or 0
        cot_padded_ids = torch.stack([
            torch.nn.functional.pad(t, (0, cot_max_len - t.shape[0]), value=pad_id)
            for t in cot_prompt_ids_list
        ])
        cot_padded_mask = torch.stack([
            torch.nn.functional.pad(t, (0, cot_max_len - t.shape[0]), value=0)
            for t in cot_prompt_mask_list
        ])

        # -- Code view: full sequence (prompt + first correct response) --
        code_input_ids = batch.batch["input_ids"]     # same combined batch
        code_attn_mask = batch.batch["attention_mask"]

        code_ids_list = []
        code_mask_list = []
        code_lengths = []
        for u in valid_uids:
            # Find first correct code rollout for this prompt
            first_correct = None
            for flat_idx in code_by_uid[u]:
                if rew[flat_idx] > 0:
                    first_correct = flat_idx
                    break
            if first_correct is None:
                first_correct = code_by_uid[u][0]  # fallback (shouldn't happen)
            ids = code_input_ids[first_correct]
            attn = code_attn_mask[first_correct]
            code_ids_list.append(ids)
            code_mask_list.append(attn)
            code_lengths.append(int(attn.sum()))

        code_max_len = max(t.shape[0] for t in code_ids_list)
        code_padded_ids = torch.stack([
            torch.nn.functional.pad(t, (0, code_max_len - t.shape[0]), value=pad_id)
            for t in code_ids_list
        ])
        code_padded_mask = torch.stack([
            torch.nn.functional.pad(t, (0, code_max_len - t.shape[0]), value=0)
            for t in code_mask_list
        ])

        jepa_batch = DataProto.from_single_dict({
            "cot_input_ids": cot_padded_ids,
            "cot_attn_mask": cot_padded_mask,
            "cot_lengths": torch.tensor(cot_prompt_lengths, dtype=torch.long),
            "code_input_ids": code_padded_ids,
            "code_attn_mask": code_padded_mask,
            "code_lengths": torch.tensor(code_lengths, dtype=torch.long),
        })
        return jepa_batch

    # ------------------------------------ JEPA triplet batch construction ---
    def _build_jepa_batch_triplet(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the jepa-separation-loss batch: full correct-CoT response, first
        correct code response, and (when available) a "clean wrong" code
        response per prompt.

        Differs from `_build_jepa_batch` in two ways the triplet mode needs:
          - the CoT view is the FULL sequence (prompt + response) of the
            first CORRECT CoT rollout, not the bare prompt of rollout 0 — p^c
            must be the predictor embedding of an actual reasoning trace.
          - it additionally selects, per prompt, the first code rollout that
            is a "clean wrong" one: a definite wrong answer was extracted
            (`has_parseable_answer=True`, `is_correct=False` via
            `compute_math_reward`), not a crash/parse failure.

        Triplet-eligible prompts (have a clean-wrong rollout) are placed
        FIRST in the returned batch; the wrong-code tensors are padded to
        the same row count B as cot/code, with `wrong_lengths == 0` marking
        the non-triplet rows (worker.jepa_update filters those out before
        its forward — see `_extract_embeddings`'s existing `rlen==0` -> zero
        embedding convention).

        Returns None if fewer than min_valid_pairs (cot_correct AND
        code_correct) pairs exist. T == 0 (no triplet-eligible prompts) is
        allowed; the loss function handles it.
        """
        uids = batch.non_tensor_batch["uid"]
        rew = reward_tensor.sum(dim=-1)

        cot_by_uid, code_by_uid, valid_uids = self._group_rows_by_uid(uids, view_tags, rew)

        if len(valid_uids) < self.jepa_cfg.min_valid_pairs:
            return None

        cot_input_ids = batch.batch["input_ids"]
        cot_attn_mask = batch.batch["attention_mask"]
        code_input_ids = batch.batch["input_ids"]
        code_attn_mask = batch.batch["attention_mask"]
        code_response_mask = batch.batch.get("response_mask", None)
        reward_models = batch.non_tensor_batch.get("reward_model", [{}] * len(batch))
        data_sources = batch.non_tensor_batch.get("data_source", [None] * len(batch))

        pad_id = self.tokenizer.pad_token_id or 0

        per_prompt = {}
        triplet_eligible, others = [], []
        for u in valid_uids:
            # First correct CoT rollout (full sequence), and a count of how
            # many CoT rollouts were correct (audit metric).
            first_correct_cot = None
            n_correct_cot = 0
            for flat_idx in cot_by_uid[u]:
                if rew[flat_idx] > 0:
                    n_correct_cot += 1
                    if first_correct_cot is None:
                        first_correct_cot = flat_idx
            cot_flat_idx = first_correct_cot

            # First correct code rollout (full sequence)
            code_flat_idx = None
            for flat_idx in code_by_uid[u]:
                if rew[flat_idx] > 0:
                    code_flat_idx = flat_idx
                    break

            ground_truth = reward_models[code_flat_idx].get("ground_truth") if isinstance(
                reward_models[code_flat_idx], dict
            ) else None
            dataset_kind = data_sources[code_flat_idx]

            # First CLEAN-wrong code rollout: parseable answer extracted, but
            # not correct. Filtered to definite-wrong BEFORE taking "first"
            # (never let a crash/parse-failure stand in as e^w).
            wrong_flat_idx = None
            for cand_idx in code_by_uid[u]:
                if rew[cand_idx] > 0:
                    continue
                if code_response_mask is not None:
                    # response_mask is only the trailing `response_length` slice of
                    # the full attention_mask (see compute_response_mask), not a
                    # full-sequence-shaped mask — slice input_ids the same way before
                    # applying it.
                    response_length = code_response_mask.shape[-1]
                    resp_ids_full = code_input_ids[cand_idx][-response_length:]
                    resp_ids = resp_ids_full[code_response_mask[cand_idx].bool()]
                else:
                    resp_ids = code_input_ids[cand_idx][code_attn_mask[cand_idx].bool()]
                if resp_ids.numel() == 0:
                    continue
                resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                result = compute_math_reward(resp_text, ground_truth, dataset_kind=dataset_kind)
                if result.has_parseable_answer:
                    wrong_flat_idx = cand_idx
                    break

            per_prompt[u] = (cot_flat_idx, code_flat_idx, wrong_flat_idx, n_correct_cot)
            (triplet_eligible if wrong_flat_idx is not None else others).append(u)

        ordered_uids = triplet_eligible + others   # T-prefix invariant
        T = len(triplet_eligible)

        def _real_tokens(ids_tensor, mask_tensor, flat_idx):
            ids = ids_tensor[flat_idx]
            attn = mask_tensor[flat_idx]
            real = ids[attn.bool()]
            return real, int(attn.sum())

        cot_ids_list, cot_lengths = [], []
        code_ids_list, code_lengths = [], []
        wrong_ids_list, wrong_lengths = [], []
        n_correct_cot_list = []
        for u in ordered_uids:
            cot_flat_idx, code_flat_idx, wrong_flat_idx, n_correct_cot = per_prompt[u]
            ids, length = _real_tokens(cot_input_ids, cot_attn_mask, cot_flat_idx)
            cot_ids_list.append(ids)
            cot_lengths.append(length)
            ids, length = _real_tokens(code_input_ids, code_attn_mask, code_flat_idx)
            code_ids_list.append(ids)
            code_lengths.append(length)
            n_correct_cot_list.append(n_correct_cot)
            if wrong_flat_idx is not None:
                ids, length = _real_tokens(code_input_ids, code_attn_mask, wrong_flat_idx)
            else:
                ids, length = code_input_ids.new_zeros((1,)), 0
            wrong_ids_list.append(ids)
            wrong_lengths.append(length)

        def _pad_stack(seqs, lengths_list):
            max_len = max(s.shape[0] for s in seqs)
            padded = torch.stack([
                torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id)
                for t in seqs
            ])
            mask = torch.stack([
                torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, max_len - length))
                for length in lengths_list
            ])
            return padded, mask

        cot_padded_ids, cot_padded_mask = _pad_stack(cot_ids_list, cot_lengths)
        code_padded_ids, code_padded_mask = _pad_stack(code_ids_list, code_lengths)
        wrong_padded_ids, wrong_padded_mask = _pad_stack(wrong_ids_list, wrong_lengths)

        jepa_batch = DataProto.from_single_dict({
            "cot_input_ids": cot_padded_ids,
            "cot_attn_mask": cot_padded_mask,
            "cot_lengths": torch.tensor(cot_lengths, dtype=torch.long),
            "code_input_ids": code_padded_ids,
            "code_attn_mask": code_padded_mask,
            "code_lengths": torch.tensor(code_lengths, dtype=torch.long),
            "wrong_input_ids": wrong_padded_ids,
            "wrong_attn_mask": wrong_padded_mask,
            "wrong_lengths": torch.tensor(wrong_lengths, dtype=torch.long),
        })
        jepa_batch.meta_info["n_correct_cot_mean"] = float(np.mean(n_correct_cot_list)) if n_correct_cot_list else 0.0
        jepa_batch.meta_info["n_triplets"] = T
        return jepa_batch

    # ------------------------------------ JEPA CLReg batch construction (v3) -
    def _build_jepa_batch_clreg(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the jepa-clreg-loss batch: per GRPO group, ALL correct CoT
        anchors, a per-anchor correct-code positive, and a per-anchor MATCHED
        clean-wrong negative (cycled from that group's clean-wrong rollouts).

        Layout (all three blocks have exactly A rows, aligned 1:1 by anchor):
          - cot block: every CORRECT CoT rollout of every valid group is an anchor
            (not just the first), so A >= number of groups.
          - code block: anchor i paired with that group's correct code rollout
            `correct_code[i mod n_correct_code]` (a genuine e^c_i per anchor).
          - wrong block: anchor i paired with that group's clean-wrong code rollout
            `group_wrongs[i mod n_group_wrongs]`; if the group has NO clean-wrong
            rollout, that anchor's wrong row is empty (`wrong_lengths == 0`) and it
            contributes no separation term. This is a MATCHED 1:1 pairing, not a
            per-group cross product — the loss pairs anchor i only with wrong i.
        worker.jepa_update filters wrong rows by `wrong_lengths > 0`; those W real
        rows stay in anchor order, so they line up with `pred_text[wrong_mask]`.

        Returns None if fewer than min_valid_pairs valid groups exist.
        """
        uids = batch.non_tensor_batch["uid"]
        rew = reward_tensor.sum(dim=-1)

        cot_by_uid, code_by_uid, valid_uids = self._group_rows_by_uid(uids, view_tags, rew)

        if len(valid_uids) < self.jepa_cfg.min_valid_pairs:
            return None

        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        code_response_mask = batch.batch.get("response_mask", None)
        reward_models = batch.non_tensor_batch.get("reward_model", [{}] * len(batch))
        data_sources = batch.non_tensor_batch.get("data_source", [None] * len(batch))
        pad_id = self.tokenizer.pad_token_id or 0

        def _real_tokens(flat_idx):
            ids = all_input_ids[flat_idx]
            attn = all_attn_mask[flat_idx]
            return ids[attn.bool()], int(attn.sum())

        cot_ids_list, cot_lengths = [], []
        code_ids_list, code_lengths = [], []
        wrong_ids_list, wrong_lengths = [], []
        n_correct_cot_list = []
        for u in valid_uids:
            correct_cot = [i for i in cot_by_uid[u] if rew[i] > 0]
            correct_code = [i for i in code_by_uid[u] if rew[i] > 0]
            if not correct_cot or not correct_code:
                continue   # _group_rows_by_uid already guarantees both non-empty
            n_correct_cot_list.append(len(correct_cot))

            ground_truth = reward_models[correct_code[0]].get("ground_truth") if isinstance(
                reward_models[correct_code[0]], dict
            ) else None
            dataset_kind = data_sources[correct_code[0]]

            # This group's clean-wrong code rollouts (parseable answer, but wrong).
            group_wrongs = []
            for cand_idx in code_by_uid[u]:
                if rew[cand_idx] > 0:
                    continue
                if code_response_mask is not None:
                    response_length = code_response_mask.shape[-1]
                    resp_ids_full = all_input_ids[cand_idx][-response_length:]
                    resp_ids = resp_ids_full[code_response_mask[cand_idx].bool()]
                else:
                    resp_ids = all_input_ids[cand_idx][all_attn_mask[cand_idx].bool()]
                if resp_ids.numel() == 0:
                    continue
                resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
                result = compute_math_reward(resp_text, ground_truth, dataset_kind=dataset_kind)
                if result.has_parseable_answer:
                    group_wrongs.append(cand_idx)

            # One row per anchor; code + wrong cycled within the group (matched pair).
            for j, cot_idx in enumerate(correct_cot):
                ids, length = _real_tokens(cot_idx)
                cot_ids_list.append(ids)
                cot_lengths.append(length)
                ids, length = _real_tokens(correct_code[j % len(correct_code)])
                code_ids_list.append(ids)
                code_lengths.append(length)
                if group_wrongs:
                    ids, length = _real_tokens(group_wrongs[j % len(group_wrongs)])
                else:
                    ids, length = all_input_ids.new_zeros((1,)), 0
                wrong_ids_list.append(ids)
                wrong_lengths.append(length)

        A = len(cot_ids_list)
        W = int(sum(1 for length in wrong_lengths if length > 0))
        if A < self.jepa_cfg.min_valid_pairs:
            return None

        def _pad_stack(seqs, lengths_list):
            max_len = max(s.shape[0] for s in seqs)
            rows = torch.stack([
                torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in seqs
            ])
            masks = torch.stack([
                torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, max_len - length))
                for length in lengths_list
            ])
            return rows, masks

        cot_padded_ids, cot_padded_mask = _pad_stack(cot_ids_list, cot_lengths)
        code_padded_ids, code_padded_mask = _pad_stack(code_ids_list, code_lengths)
        wrong_padded_ids, wrong_padded_mask = _pad_stack(wrong_ids_list, wrong_lengths)

        jepa_batch = DataProto.from_single_dict({
            "cot_input_ids": cot_padded_ids,
            "cot_attn_mask": cot_padded_mask,
            "cot_lengths": torch.tensor(cot_lengths, dtype=torch.long),
            "code_input_ids": code_padded_ids,
            "code_attn_mask": code_padded_mask,
            "code_lengths": torch.tensor(code_lengths, dtype=torch.long),
            "wrong_input_ids": wrong_padded_ids,
            "wrong_attn_mask": wrong_padded_mask,
            "wrong_lengths": torch.tensor(wrong_lengths, dtype=torch.long),
        })
        jepa_batch.meta_info["n_correct_cot_mean"] = float(np.mean(n_correct_cot_list)) if n_correct_cot_list else 0.0
        jepa_batch.meta_info["n_anchors"] = A
        jepa_batch.meta_info["n_wrong"] = W
        return jepa_batch

    def _build_jepa_batch_tcr(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the jepa-tcr-loss batch: correct student anchors from BOTH the
        CoT and Code views, each paired with a PRECOMPUTED teacher-correct target
        embedding.

        Unlike the clreg/separation builders this has a single block:
          - anchor block: every CORRECT rollout (rew>0), CoT *or* Code, of every
            prompt whose dataset index has cached teacher targets is an anchor.
            Both views are pulled toward the SAME teacher target under the single
            TCR alpha, so they are drawn toward each other transitively — cross-view
            consistency falls out without a separate term or a second weight. Wrong
            rollouts are ignored unless jepa_anchor_set says otherwise (the teacher
            target replaces e^c, and there is no separation term).
          - teacher_target block: a (A, d) float tensor aligned 1:1 to the anchor
            rows, drawn from this prompt's cached targets `Z_u` (n_u, d) by either
            cycle (`j % n_u`) or random matching (jepa.tcr_match).

        Prompts with no cached teacher targets contribute no anchors (their JEPA
        signal is simply absent). Returns None if fewer than min_valid_pairs
        anchors exist (worker also guards via min_valid_pairs).
        """
        assert self.teacher_targets is not None, "teacher target cache not loaded"
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)

        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0

        # Group anchor-eligible rows by uid AND view, preserving first-seen order.
        # Record each uid's dataset index for the teacher-target cache lookup.
        rows_by_uid: dict = defaultdict(lambda: {"cot": [], "code": []})
        idx_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v not in ("cot", "code"):
                continue
            rows_by_uid[u][v].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None

        anchor_set = self.jepa_cfg.jepa_anchor_set

        def _select(idxs):
            # Select anchors by reward according to jepa_anchor_set, then drop
            # degenerate zero-length rows (they would yield a zero embedding and
            # pollute the stratified means; normal rollouts always carry the prompt).
            if anchor_set == "correct":
                sel = [i for i in idxs if rew[i] > 0]
            elif anchor_set == "wrong":
                sel = [i for i in idxs if rew[i] <= 0]
            else:  # "all"
                sel = list(idxs)
            return [i for i in sel if int(all_attn_mask[i].sum()) > 0]

        cot_ids_list, cot_lengths, target_list = [], [], []
        group_id_list, is_correct_list = [], []
        crossview_pairs: list = []
        n_correct_cot_list = []
        group_counter = 0
        for u in dict.fromkeys(uids):  # dedup, preserves first-seen order
            ds_idx = idx_by_uid.get(u)
            targets = self.teacher_targets.get(ds_idx) if ds_idx is not None else None
            if targets is None or targets.numel() == 0:
                continue
            cot_sel = _select(rows_by_uid[u]["cot"])
            code_sel = _select(rows_by_uid[u]["code"])
            if not cot_sel and not code_sel:
                continue
            n_u = targets.shape[0]
            n_correct_cot_list.append(sum(1 for i in cot_sel + code_sel if rew[i] > 0))
            gid = group_counter
            group_counter += 1
            # Pair the k-th correct CoT with the k-th correct Code and give the PAIR
            # one SHARED teacher target. Minimizing both views' alignment to the same
            # point pulls CoT and Code of this prompt together AND toward the teacher.
            # When a slot has BOTH views present we also record the (CoT-row, Code-row)
            # anchor positions in `crossview_pairs` so the worker can add the CoT<->Code
            # alignment as another term in the TCR align slot. Unequal counts:
            # leftover unpaired anchors still align to their slot's target but form no
            # cross-view pair. All anchors share the [x, y_S, [PRED]] format.
            n_slots = max(len(cot_sel), len(code_sel))
            for k in range(n_slots):
                if self.jepa_cfg.tcr_match == "random":
                    t_row = int(torch.randint(n_u, (1,)).item())
                else:  # cycle
                    t_row = k % n_u
                slot_pos = {}
                for view, sel in (("cot", cot_sel), ("code", code_sel)):
                    if k >= len(sel):
                        continue
                    idx = sel[k]
                    ids = all_input_ids[idx][all_attn_mask[idx].bool()]
                    slot_pos[view] = len(cot_ids_list)  # anchor-row position
                    cot_ids_list.append(ids)
                    cot_lengths.append(int(all_attn_mask[idx].sum()))
                    target_list.append(targets[t_row])
                    group_id_list.append(gid)
                    is_correct_list.append(bool(rew[idx] > 0))
                if "cot" in slot_pos and "code" in slot_pos:
                    crossview_pairs.append((slot_pos["cot"], slot_pos["code"]))

        A = len(cot_ids_list)
        if A < self.jepa_cfg.min_valid_pairs:
            return None

        max_len = max(s.shape[0] for s in cot_ids_list)
        cot_padded_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in cot_ids_list
        ])
        cot_padded_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, max_len - length))
            for length in cot_lengths
        ])
        teacher_target = torch.nn.functional.normalize(torch.stack(target_list, dim=0).float(), dim=-1)

        # Per-anchor cross-view partner row (or -1). Shape (A,) so it batches with the
        # anchor rows; the worker reconstructs unordered (CoT,Code) pairs from it.
        crossview_partner = torch.full((A,), -1, dtype=torch.long)
        for a, b in crossview_pairs:
            crossview_partner[a] = b
            crossview_partner[b] = a

        jepa_batch = DataProto.from_single_dict({
            "cot_input_ids": cot_padded_ids,
            "cot_attn_mask": cot_padded_mask,
            "cot_lengths": torch.tensor(cot_lengths, dtype=torch.long),
            "teacher_target": teacher_target,
            "anchor_group_id": torch.tensor(group_id_list, dtype=torch.long),
            "anchor_is_correct": torch.tensor(is_correct_list, dtype=torch.bool),
            "crossview_partner": crossview_partner,
        })
        jepa_batch.meta_info["n_correct_cot_mean"] = float(np.mean(n_correct_cot_list)) if n_correct_cot_list else 0.0
        jepa_batch.meta_info["n_anchors"] = A
        jepa_batch.meta_info["n_crossview_pairs"] = len(crossview_pairs)
        return jepa_batch

    def _build_jepa_batch_tcr_dual(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> DataProto | None:
        """Build the jepa-tcr-dual batch: SEPARATE CoT and Code student anchor
        blocks, each with its OWN precomputed teacher-correct target cache, plus a
        per-CoT-row self-consistency partner pointing at the paired Code row.

        Layout:
          - cot block (A_c rows): correct CoT rollouts of prompts with a cached CoT
            target; `teacher_target` (A_c, d) from `self.teacher_targets`.
          - code block (A_k rows): correct Code rollouts of prompts with a cached
            Code target; `code_teacher_target` (A_k, d) from
            `self.code_teacher_targets`.
          - `self_partner` (A_c,) long: for each CoT row, the row index INTO THE CODE
            BLOCK of its paired (same-prompt, same-slot) code rollout, or -1. The
            worker reads that code row's BOUNDARY embedding as the stop-grad
            self-consistency target z_self.

        Group ids are shared per prompt across both views so the stratified,
        prompt-averaged aggregation lines up. A prompt missing one view's cache
        still contributes the other view's anchors (with no self-consistency pair).
        Returns None if fewer than min_valid_pairs CoT anchors exist.
        """
        assert self.teacher_targets is not None, "CoT teacher target cache not loaded"
        assert self.code_teacher_targets is not None, "Code teacher target cache not loaded"
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)

        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0

        rows_by_uid: dict = defaultdict(lambda: {"cot": [], "code": []})
        idx_by_uid: dict = {}
        for i, (u, v) in enumerate(zip(uids, view_tags)):
            if v not in ("cot", "code"):
                continue
            rows_by_uid[u][v].append(i)
            if u not in idx_by_uid:
                info = extra_infos[i]
                idx_by_uid[u] = int(info["index"]) if isinstance(info, dict) else None

        anchor_set = self.jepa_cfg.jepa_anchor_set

        def _select(idxs):
            if anchor_set == "correct":
                sel = [i for i in idxs if rew[i] > 0]
            elif anchor_set == "wrong":
                sel = [i for i in idxs if rew[i] <= 0]
            else:  # "all"
                sel = list(idxs)
            return [i for i in sel if int(all_attn_mask[i].sum()) > 0]

        def _match_row(k, n_u):
            return int(torch.randint(n_u, (1,)).item()) if self.jepa_cfg.tcr_match == "random" else k % n_u

        # Single combined anchor block (CoT rows FIRST, then Code rows) so the
        # DataProto keeps one uniform batch dim even when the two views have
        # different anchor counts. `is_code` recovers the split; `self_partner`
        # (only set on CoT rows) indexes into the CODE sub-block (0..A_k-1).
        cot_ids_list, cot_lengths, cot_target_list, cot_gid, cot_ic = [], [], [], [], []
        code_ids_list, code_lengths, code_target_list, code_gid, code_ic = [], [], [], [], []
        self_pairs: list = []   # (cot_row_pos, code_row_pos) — positions within each sub-block
        n_correct_list = []
        group_counter = 0
        for u in dict.fromkeys(uids):
            ds_idx = idx_by_uid.get(u)
            Z_cot = self.teacher_targets.get(ds_idx) if ds_idx is not None else None
            Z_code = self.code_teacher_targets.get(ds_idx) if ds_idx is not None else None
            cot_sel = _select(rows_by_uid[u]["cot"]) if (Z_cot is not None and Z_cot.numel() > 0) else []
            code_sel = _select(rows_by_uid[u]["code"]) if (Z_code is not None and Z_code.numel() > 0) else []
            if not cot_sel and not code_sel:
                continue
            gid = group_counter
            group_counter += 1
            n_correct_list.append(sum(1 for i in cot_sel + code_sel if rew[i] > 0))
            n_slots = max(len(cot_sel), len(code_sel))
            for k in range(n_slots):
                cot_pos = code_pos = None
                if k < len(cot_sel):
                    idx = cot_sel[k]
                    cot_pos = len(cot_ids_list)
                    cot_ids_list.append(all_input_ids[idx][all_attn_mask[idx].bool()])
                    cot_lengths.append(int(all_attn_mask[idx].sum()))
                    cot_target_list.append(Z_cot[_match_row(k, Z_cot.shape[0])])
                    cot_gid.append(gid)
                    cot_ic.append(bool(rew[idx] > 0))
                if k < len(code_sel):
                    idx = code_sel[k]
                    code_pos = len(code_ids_list)
                    code_ids_list.append(all_input_ids[idx][all_attn_mask[idx].bool()])
                    code_lengths.append(int(all_attn_mask[idx].sum()))
                    code_target_list.append(Z_code[_match_row(k, Z_code.shape[0])])
                    code_gid.append(gid)
                    code_ic.append(bool(rew[idx] > 0))
                if cot_pos is not None and code_pos is not None:
                    self_pairs.append((cot_pos, code_pos))

        A_c = len(cot_ids_list)
        A_k = len(code_ids_list)
        if A_c < self.jepa_cfg.min_valid_pairs or A_k == 0:
            return None

        # Concatenate the two sub-blocks (CoT first) into one padded block.
        all_ids_list = cot_ids_list + code_ids_list
        all_lengths = cot_lengths + code_lengths
        max_len = max(s.shape[0] for s in all_ids_list)
        anchor_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in all_ids_list
        ])
        anchor_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(length, dtype=torch.long), (0, max_len - length))
            for length in all_lengths
        ])
        target = torch.nn.functional.normalize(
            torch.stack(cot_target_list + code_target_list, dim=0).float(), dim=-1
        )
        is_code = torch.tensor([False] * A_c + [True] * A_k, dtype=torch.bool)
        group_id = torch.tensor(cot_gid + code_gid, dtype=torch.long)
        is_correct = torch.tensor(cot_ic + code_ic, dtype=torch.bool)
        # self_partner aligned to the FULL block (length A_c + A_k); set on CoT rows only.
        self_partner = torch.full((A_c + A_k,), -1, dtype=torch.long)
        for c_pos, k_pos in self_pairs:
            self_partner[c_pos] = k_pos   # k_pos is the position within the CODE sub-block

        jepa_batch = DataProto.from_single_dict({
            "anchor_input_ids": anchor_ids,
            "anchor_attn_mask": anchor_mask,
            "anchor_lengths": torch.tensor(all_lengths, dtype=torch.long),
            "teacher_target": target,
            "is_code": is_code,
            "anchor_group_id": group_id,
            "anchor_is_correct": is_correct,
            "self_partner": self_partner,
        })
        jepa_batch.meta_info["n_correct_cot_mean"] = float(np.mean(n_correct_list)) if n_correct_list else 0.0
        jepa_batch.meta_info["n_anchors"] = A_c
        jepa_batch.meta_info["n_anchors_code"] = A_k
        jepa_batch.meta_info["n_self_pairs"] = len(self_pairs)
        return jepa_batch

    # ------------------------------------------- TCR reward shaping (idea #2) -
    @staticmethod
    def _stratified_shaping(
        s: torch.Tensor,
        group_ids: list,
        is_correct: torch.Tensor,
        beta: float,
        sigma_floor: float,
    ) -> tuple[torch.Tensor, int]:
        """Within-(group, is_correct)-stratum standardized shaping term β·ŝ.

        For each (group_id, correctness) stratum with >=2 members, standardize the
        scores `s` (mean 0, std clamped at `sigma_floor`) and scale by β. Singleton
        or degenerate strata contribute 0. Returns (shaped (n,), n_strata_shaped).

        Pure function of its tensors (no model/state) so it is unit-testable. Key
        invariants it guarantees: (a) a constant added to all of one stratum's
        scores leaves ŝ unchanged (global-shift invariance); (b) shaping is computed
        independently per correctness stratum, so it never moves mass across the
        correct/wrong boundary.
        """
        n = s.shape[0]
        shaped = torch.zeros(n, dtype=torch.float32)
        strata: dict = defaultdict(list)
        for j in range(n):
            strata[(group_ids[j], bool(is_correct[j]))].append(j)
        n_shaped = 0
        for members in strata.values():
            if len(members) < 2:
                continue
            idx = torch.tensor(members)
            sv = s[idx].float()
            shat = (sv - sv.mean()) / max(float(sv.std(unbiased=False)), sigma_floor)
            for m, val in zip(members, shat):
                shaped[m] = beta * float(val)
            n_shaped += 1
        return shaped, n_shaped

    def _compute_tcr_reward_shaping(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        view_tags: np.ndarray,
    ) -> tuple[torch.Tensor, dict]:
        """jepa-tcr-reward: teacher-alignment reward shaping (NO differentiable loss).

        Scores every CoT *and* Code rollout's [PRED] latent against its question's
        cached teacher-correct targets, standardizes the score WITHIN its
        (uid, view, is_correct) reward stratum, and returns a per-row additive
        advantage term β·ŝ_i. Both views are shaped symmetrically with the dual-view
        JEPA loss arm.

        Within-stratum centering makes the term (a) invariant to a global latent
        shift (defeating the cos_correct≈cos_wrong shortcut) and (b) unable to flip
        a correct-vs-wrong ordering (ground truth always dominates). Rows whose
        prompt has no cached target, or singleton/degenerate strata, get 0.

        Returns:
            (shape_per_row, metrics): shape_per_row is a (B,) CPU float tensor
            aligned 1:1 with `batch` rows (0 for non-cot / unshaped rows).
        """
        assert self.teacher_targets is not None, "teacher target cache not loaded"
        B = len(batch)
        shape_per_row = torch.zeros(B, dtype=torch.float32)
        uids = batch.non_tensor_batch["uid"]
        extra_infos = batch.non_tensor_batch["extra_info"]
        rew = reward_tensor.sum(dim=-1)
        all_input_ids = batch.batch["input_ids"]
        all_attn_mask = batch.batch["attention_mask"]
        pad_id = self.tokenizer.pad_token_id or 0

        # Per-view target cache: in jepa-tcr-reward-dual, CODE rows are scored against
        # the CODE teacher cache and COT rows against the COT cache (each rollout aligned
        # to its own view's teacher). Other reward modes have no code cache and score
        # both views against the single CoT cache (back-compat).
        code_cache = self.code_teacher_targets

        def _cache_for(view: str):
            return code_cache if (view == "code" and code_cache is not None) else self.teacher_targets

        # Collect all CoT *and* Code rows whose prompt has cached targets. Both views
        # are scored against the teacher and shaped, mirroring the dual-view JEPA arm
        # (a code rollout that lands near the teacher should earn the same advantage
        # bonus a CoT one does). View is tracked so each view is standardized within
        # its own (uid, view, is_correct) stratum below.
        scored_rows: list[int] = []
        scored_views: list[str] = []
        for i, v in enumerate(view_tags):
            if v not in ("cot", "code") or int(all_attn_mask[i].sum()) == 0:
                continue
            info = extra_infos[i]
            ds_idx = int(info["index"]) if isinstance(info, dict) else None
            if ds_idx is None or _cache_for(v).get(ds_idx) is None:
                continue
            scored_rows.append(i)
            scored_views.append(v)
        if len(scored_rows) < self.jepa_cfg.min_valid_pairs:
            return shape_per_row, {"shaping/frac_groups_shaped": 0.0, "shaping/n_rows_scored": 0.0}

        # Forward-only [PRED] embeddings for the scored rows (worker RPC, no backward).
        ids_list = [all_input_ids[i][all_attn_mask[i].bool()] for i in scored_rows]
        lengths = [int(all_attn_mask[i].sum()) for i in scored_rows]
        max_len = max(s.shape[0] for s in ids_list)
        padded_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.shape[0]), value=pad_id) for t in ids_list
        ])
        padded_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(L, dtype=torch.long), (0, max_len - L)) for L in lengths
        ])
        score_td = DataProto.from_single_dict({
            "cot_input_ids": padded_ids,
            "cot_attn_mask": padded_mask,
            "cot_lengths": torch.tensor(lengths, dtype=torch.long),
        }).to_tensordict()
        out = self.actor_rollout_wg.score_cot_embeddings(score_td)
        if isinstance(out, list):
            out = out[0] if out else None
        emb = out["cot_emb"].float()  # (n_rows, d), L2-normalized

        # s_i = max_k <p_i, z_k> over the question's cached targets.
        s = torch.empty(len(scored_rows), dtype=torch.float32)
        is_correct = torch.empty(len(scored_rows), dtype=torch.bool)
        gids = []
        for j, i in enumerate(scored_rows):
            ds_idx = int(extra_infos[i]["index"])
            Z = _cache_for(scored_views[j])[ds_idx].to(emb.dtype)  # (n_i, d), per-view cache
            s[j] = (emb[j].unsqueeze(0) * Z).sum(dim=-1).max()
            is_correct[j] = bool(rew[i] > 0)
            # Stratum key includes the view so CoT and Code are standardized against
            # their own kind (their similarity-to-teacher scales differ).
            gids.append((uids[i], scored_views[j]))

        # Standardize within each (uid, view, is_correct) stratum, then scale by β.
        beta = float(self.jepa_cfg.tcr_reward_beta)
        sigma_floor = float(self.jepa_cfg.tcr_reward_sigma_floor)
        shaped, n_shaped_groups = self._stratified_shaping(
            s, gids, is_correct, beta=beta, sigma_floor=sigma_floor,
        )
        for j in range(len(scored_rows)):
            shape_per_row[scored_rows[j]] = float(shaped[j])

        # Monitors (do not affect the optimized objective).
        s_np = s.numpy()
        corr = 0.0
        if is_correct.any() and (~is_correct).any():
            corr = float(np.corrcoef(s_np, is_correct.numpy().astype(np.float32))[0, 1])
        metrics = {
            "shaping/corr_s_correct": corr,
            "shaping/s_mean_correct": float(s[is_correct].mean()) if is_correct.any() else 0.0,
            "shaping/s_mean_wrong": float(s[~is_correct].mean()) if (~is_correct).any() else 0.0,
            "shaping/adv_std": float(shape_per_row[shape_per_row != 0].std(unbiased=False)) if (shape_per_row != 0).any() else 0.0,
            "shaping/n_rows_scored": float(len(scored_rows)),
            "shaping/n_rows_scored_cot": float(scored_views.count("cot")),
            "shaping/n_rows_scored_code": float(scored_views.count("code")),
            "shaping/n_strata_shaped": float(n_shaped_groups),
            "shaping/frac_rows_shaped": float((shape_per_row != 0).sum()) / max(1, len(scored_rows)),
        }
        return shape_per_row, metrics

    # --------------------------------------- auto-disable on alignment plateau -
    def _tracked_signals(self) -> list[tuple[str, str]]:
        """(arm_name, metric_key) pairs to track for the plateau latch (higher=better).

        jepa-tcr-dual tracks its three arms independently; the reward-shaping modes
        track a single shaping signal; other differentiable modes track CoT alignment.
        An explicit `auto_off_metric` override collapses to one global signal (legacy).
        """
        if self.jepa_cfg.auto_off_metric:
            return [("global", self.jepa_cfg.auto_off_metric)]
        lt = self.jepa_cfg.loss_type
        if lt == "jepa-tcr-dual":
            return [("cot", "jepa/cos_cot"), ("code", "jepa/cos_code"), ("self", "jepa/cos_self")]
        if lt in ("jepa-tcr-reward", "jepa-tcr-reward-dual", "jepa-tcr-hybrid"):
            return [("shaping", "shaping/s_mean_correct")]
        return [("cot", "jepa/cos_cot")]

    def _maybe_disable_jepa_signal(self, metrics: dict) -> None:
        """Latch each tracked JEPA signal OFF independently once it plateaus.

        For each tracked (arm, metric): after warmup, a step that fails to beat that
        arm's running best by `auto_off_min_delta` increments its own stall counter;
        once it reaches `auto_off_patience`, only THAT arm latches off for the rest of
        training (its loss term + matching-view shaping go to zero). The global
        `_jepa_signal_off` latches True only once EVERY tracked arm is off, skipping the
        whole JEPA block. Steps where a metric is absent/zero (no anchors of that view)
        are ignored so they neither advance nor reset that arm's counter.
        """
        cfg = self.jepa_cfg
        if not cfg.enable or not cfg.auto_off_enable:
            return
        metrics["jepa/signal_off"] = float(self._jepa_signal_off)
        if self._jepa_signal_off:
            return
        signals = self._tracked_signals()
        warm = self.global_steps >= cfg.auto_off_warmup_steps
        for arm, key in signals:
            metrics[f"jepa/off_{arm}"] = float(self._off_arm[arm])
            if not warm or self._off_arm[arm]:
                continue
            val = metrics.get(key)
            if val is None:
                continue
            val = float(val)
            if val == 0.0:   # no anchors of this view this step -> not a real measurement
                continue
            if val > self._align_best[arm] + cfg.auto_off_min_delta:
                self._align_best[arm] = val
                self._align_stall[arm] = 0
            else:
                self._align_stall[arm] += 1
                if self._align_stall[arm] >= cfg.auto_off_patience:
                    self._off_arm[arm] = True
                    logger.info(
                        "[jepa] arm '%s' plateaued on %s (val=%.4f best=%.4f, stalled %d>=%d steps) "
                        "-> disabling this arm (loss term + matching-view shaping)",
                        arm, key, val, self._align_best[arm], self._align_stall[arm], cfg.auto_off_patience,
                    )
            metrics[f"jepa/off_{arm}"] = float(self._off_arm[arm])
            metrics[f"jepa/align_best_{arm}"] = self._align_best[arm]
            metrics[f"jepa/align_stall_{arm}"] = float(self._align_stall[arm])
        # Fully off only when every tracked arm has individually latched.
        if signals and all(self._off_arm[arm] for arm, _ in signals):
            if not self._jepa_signal_off:
                logger.info("[jepa] all tracked arms disabled -> JEPA signal fully off for the rest of training")
            self._jepa_signal_off = True
        metrics["jepa/signal_off"] = float(self._jepa_signal_off)

    # ------------------------------------------------ memory diagnostics -----
    @staticmethod
    def _log_gpu_mem(tag: str) -> None:
        """Log device-level GPU memory (used/total MiB) at a phase boundary.

        Uses `nvidia-smi` rather than torch so the driver process does NOT create
        a CUDA context (which would itself consume GPU memory on this memory-tight
        colocated setup). Device-level used memory captures BOTH the vLLM EngineCore
        and the FSDP-actor worker processes. Best-effort: never raises into the loop.
        """
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            logger.info("[jepa-gpu-mem] %s: %s MiB (used,total per GPU)", tag, out.replace("\n", " | "))
        except Exception as e:  # noqa: BLE001 - diagnostics must never break training
            logger.warning("[jepa-gpu-mem] %s: snapshot failed (%s)", tag, e)

    # ---------------------------------------------------- training loop -----
    def fit(self):
        """JEPA-GRPO training loop.

        Compared to RayPPOTrainer.fit(), after each GRPO actor update we:
          1. Generate Code-view rollouts for the same prompts
          2. Score correctness with the math reward function
          3. Build JEPA pairs and run jepa_update on the worker
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.ray_trainer import compute_advantage, compute_response_mask
        from verl.trainer.ppo.reward import extract_reward
        from verl.utils.metric import reduce_metrics

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            logger.log(data=val_metrics, step=self.global_steps)
            self._maybe_save_best_checkpoint(val_metrics)

        from tqdm import tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps,
                            desc="JEPA-GRPO Training")

        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict[str, Any] = {}
                timing_raw: dict[str, float] = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # ── Step 1: unified CoT + Code rollout ──────────────────
                # Both views now contribute to the GRPO update below (the old
                # code generated a SEPARATE rollout_n-sized Code batch later,
                # whose reward only ever fed JEPA pairing — half the rollout
                # compute never reached the policy gradient). GRPO's grouping
                # is uid-based (compute_grpo_outcome_advantage groups by
                # data.non_tensor_batch["uid"], not by position), so the cot
                # and code sub-batches just need matching uids per prompt —
                # no positional interleaving is required.
                with simple_timer("cot_gen", timing_raw):
                    rollout_n = self.config.actor_rollout_ref.rollout.n
                    n_cot = self.jepa_cfg.n_cot
                    n_code = self.jepa_cfg.n_code

                    sub_batches = []

                    if n_cot > 0:
                        gen_batch_cot = self._get_gen_batch(batch)
                        gen_batch_cot.meta_info["global_steps"] = self.global_steps
                        gen_batch_cot_rep = gen_batch_cot.repeat(repeat_times=n_cot, interleave=True)
                        cot_gen_output = self.async_rollout_manager.generate_sequences(gen_batch_cot_rep)
                        # Each generate_sequences call attaches its own per-call "timing"
                        # diagnostics dict to meta_info. Pop+merge into timing_raw (the
                        # pattern used elsewhere in verl, e.g. ray_trainer.py's
                        # `timing_raw.update(combined_gen_output.meta_info["timing"])`)
                        # instead of letting it ride along on the DataProto — otherwise it
                        # collides later: cot's and code's "timing" dicts are different
                        # objects, and both union() and DataProto.concat() assert equality
                        # on overlapping non-"metrics" meta_info keys.
                        if "timing" in cot_gen_output.meta_info:
                            timing_raw.update(
                                {f"cot_gen/{k}": v for k, v in cot_gen_output.meta_info.pop("timing").items()}
                            )
                        # AgentLoop echoes raw_prompt back into its output. cot_sub is unioned
                        # onto `batch.repeat(...)` (not onto a gen_batch that itself carries
                        # raw_prompt — unlike the old code), so if only ONE of cot/code drops
                        # this key, DataProto.concat ends up with a non_tensor_batch entry whose
                        # length matches just one sub instead of the full combined batch size.
                        # Drop it symmetrically from both; nothing downstream reads it.
                        cot_gen_output.non_tensor_batch.pop("raw_prompt", None)

                        cot_sub = batch.repeat(repeat_times=n_cot, interleave=True)
                        # .repeat() passes meta_info by reference (verl/protocol.py), so
                        # cot_sub.meta_info IS batch.meta_info here — decouple with a shallow
                        # copy before union() mutates it, so this sub's union() can't leak
                        # into the shared `batch` object (and from there into code_sub below).
                        cot_sub.meta_info = dict(cot_sub.meta_info)
                        cot_sub = cot_sub.union(cot_gen_output)
                        cot_sub.non_tensor_batch["view"] = np.array(["cot"] * len(cot_sub), dtype=object)
                        sub_batches.append(cot_sub)

                    if n_cot > 0 and n_code > 0:
                        # Issuing a second vLLM engine RPC immediately after generate_sequences()
                        # returns reproducibly segfaults vLLM's executor (cuMemcpy) at the start
                        # of the first real training step — reproduced 3x, including once where
                        # the "second RPC" was a checkpoint_manager.sleep_replicas() call (NOT
                        # another generation), so this isn't specific to generation-vs-generation;
                        # it's specific to back-to-back vLLM RPCs with no real wall-clock gap.
                        # generate_sequences() is wrapped in asyncio.run (verl/utils/ray_utils.py),
                        # which should block until fully complete, but empirically something in
                        # vLLM's async engine (e.g. background request/KV-cache bookkeeping) is
                        # still settling when the next RPC submits new CUDA work. The old code
                        # never hit this because substantial real CPU/FSDP work (reward,
                        # advantage, old-logprob) always sat between its two generate_sequences
                        # calls — never back-to-back. A plain delay is a blunt instrument, but
                        # it's the minimal, lowest-risk way to give vLLM's async state time to
                        # drain without restructuring the data flow (see git history for two
                        # real bugs introduced by data-flow changes in this same unification).
                        import time as _time
                        _time.sleep(float(os.environ.get("JEPA_VLLM_DRAIN_S", "5")))

                    if n_code > 0:
                        code_gen_batch = self._tokenize_code_prompts(batch)  # batch is still un-repeated here
                        code_gen_batch.meta_info["global_steps"] = self.global_steps
                        code_gen_batch_rep = code_gen_batch.repeat(repeat_times=n_code, interleave=True)
                        code_gen_output = self.async_rollout_manager.generate_sequences(code_gen_batch_rep)
                        # See cot_gen_output comment above — drop symmetrically from both views.
                        code_gen_output.non_tensor_batch.pop("raw_prompt", None)
                        # See cot_gen_output comment above — pop+merge "timing" instead of
                        # letting it collide with cot's (different) "timing" dict at union/concat.
                        if "timing" in code_gen_output.meta_info:
                            timing_raw.update(
                                {f"code_gen/{k}": v for k, v in code_gen_output.meta_info.pop("timing").items()}
                            )

                        code_sub = batch.repeat(repeat_times=n_code, interleave=True)
                        code_sub.meta_info = dict(code_sub.meta_info)  # see cot_sub comment above
                        code_sub = code_sub.union(code_gen_output)
                        code_sub.non_tensor_batch["view"] = np.array(["code"] * len(code_sub), dtype=object)
                        sub_batches.append(code_sub)

                    batch = DataProto.concat(sub_batches)
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                # ── Step 2: rewards & advantages (full cot+code batch) ──
                with simple_timer("cot_reward_adv", timing_raw):
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        batch = batch.union(self._compute_reward_colocate(batch))
                    reward_tensor, reward_extra_infos = extract_reward(batch)
                    batch.batch["token_level_scores"] = reward_tensor
                    if not self.config.algorithm.use_kl_in_reward:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=rollout_n,
                        norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
                        config=self.config.algorithm,
                    )

                # Sleep rollout replicas BEFORE old_log_prob so vLLM releases its KV
                # reservation before the actor recomputes log-probs (the ~7 GiB lm_head
                # logits block). Previously the sleep sat AFTER old_log_prob, so that block
                # was allocated while vLLM still held its full KV pool -> OOM near the card
                # ceiling (logs: "Tried to allocate 6.90 GiB ... 92.14 GiB in use").
                # Called exactly once per step; vLLM is woken again only at weight_sync_2.
                #
                # DRAIN GUARD: issuing a vLLM RPC (sleep_replicas IS one) immediately after
                # generate_sequences() reproducibly segfaults vLLM's executor when there is no
                # real wall-clock gap (see the same-class issue + _time.sleep(5) guard in the
                # Step-1 cot/code generation block). The old ordering was safe only because
                # old_log_prob (~14 s of FSDP work) sat between generation and the sleep; the
                # reward/advantage block above is only ~15 ms, so we reinstate the documented
                # short drain before the sleep RPC.
                self._log_gpu_mem("after_gen_before_sleep")
                import time as _time
                _time.sleep(float(os.environ.get("JEPA_VLLM_DRAIN_S", "5")))
                with simple_timer("sleep_replicas_1", timing_raw):
                    self.checkpoint_manager.sleep_replicas()
                self._log_gpu_mem("after_sleep")

                # ── Step 3: Compute old log-probs & (optional) ref ──────
                # Actor/FSDP-only (compute_log_prob); does NOT call vLLM. All rollout outputs
                # it reads are already materialized in `batch` (DataProto.concat in Step 1),
                # so sleeping vLLM first is safe.
                self._log_gpu_mem("before_old_log_prob")
                with simple_timer("old_log_prob", timing_raw):
                    old_log_prob, _old_log_prob_mfu = self._compute_old_log_prob(batch)
                    batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        ref_log_prob = self._compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                self._log_gpu_mem("after_old_log_prob")

                # ── Step 3.5: TCR reward shaping (idea #2) ───────────────
                # Fold teacher-alignment into the advantage BEFORE the actor update,
                # so the signal rides the policy gradient (generation channel) rather
                # than a separate latent-pull backward. vLLM is asleep and the actor
                # FSDP is warm here, so the forward-only embedding pass is memory-safe.
                if self.jepa_cfg.enable and not self._jepa_signal_off and self.jepa_cfg.loss_type in (
                    "jepa-tcr-reward", "jepa-tcr-hybrid", "jepa-tcr-reward-dual", "jepa-tcr-dual"
                ):
                    with simple_timer("tcr_reward_shaping", timing_raw):
                        view_tags_s = batch.non_tensor_batch["view"]
                        shape_per_row, shaping_metrics = self._compute_tcr_reward_shaping(
                            batch=batch, reward_tensor=reward_tensor, view_tags=view_tags_s,
                        )
                        # Per-view plateau latch: zero shaping for a view whose arm is off.
                        # (jepa-tcr-dual only; reward modes use the single 'shaping' arm and
                        # are gated wholesale by `_jepa_signal_off` at the block entry.)
                        if self.jepa_cfg.loss_type == "jepa-tcr-dual":
                            if self._off_arm["cot"]:
                                shape_per_row[torch.from_numpy(view_tags_s == "cot")] = 0.0
                            if self._off_arm["code"]:
                                shape_per_row[torch.from_numpy(view_tags_s == "code")] = 0.0
                        rmask = batch.batch["response_mask"]
                        shape_term = shape_per_row.to(
                            device=batch.batch["advantages"].device,
                            dtype=batch.batch["advantages"].dtype,
                        ).unsqueeze(-1) * rmask
                        batch.batch["advantages"] = batch.batch["advantages"] + shape_term
                        metrics.update(shaping_metrics)

                # ── Step 4: GRPO actor update ────────────────────────────
                with simple_timer("update_actor", timing_raw):
                    actor_output = self._update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_metrics)

                # ── Step 5: JEPA (if enabled) ────────────────────────────
                # The code-framed rollouts already live in `batch` (tagged
                # view=="code" in Step 1) and already received reward in
                # Step 2 — no separate rollout/reward pass needed here
                # anymore, only pair-building for the auxiliary loss.
                view_tags = batch.non_tensor_batch["view"]
                rew_scalar_all = reward_tensor.sum(dim=-1)
                cot_mask_rows = (view_tags == "cot")
                code_mask_rows = (view_tags == "code")

                # jepa-tcr-reward AND jepa-tcr-reward-dual apply their signal purely as
                # advantage shaping in Step 3.5 (no auxiliary loss / backward), so skip the
                # JEPA loss block entirely. jepa-tcr-hybrid AND jepa-tcr-dual keep BOTH: the
                # Step-3.5 shaping (beta) AND this differentiable loss arm.
                if self.jepa_cfg.enable and not self._jepa_signal_off and self.jepa_cfg.loss_type not in (
                    "jepa-tcr-reward", "jepa-tcr-reward-dual"
                ):
                    metrics["jepa/n_cot"] = float(n_cot)
                    metrics["jepa/n_code"] = float(n_code)

                    # Build JEPA pairs
                    with simple_timer("jepa_build_batch", timing_raw):
                        # Both supported loss_types use a 3-view (cot/code/clean-wrong)
                        # builder; clreg (v3) collects ALL anchors/wrongs per group
                        # with group ids, the hinge mode one matched triplet per group.
                        if self.jepa_cfg.loss_type == "jepa-tcr-dual":
                            jepa_batch = self._build_jepa_batch_tcr_dual(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )
                        elif self.jepa_cfg.loss_type in ("jepa-tcr-loss", "jepa-tcr-hybrid"):
                            jepa_batch = self._build_jepa_batch_tcr(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )
                        elif self.jepa_cfg.loss_type == "jepa-clreg-loss":
                            jepa_batch = self._build_jepa_batch_clreg(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )
                        else:
                            jepa_batch = self._build_jepa_batch_triplet(
                                batch=batch,
                                reward_tensor=reward_tensor,
                                view_tags=view_tags,
                            )

                    if jepa_batch is not None:
                        # JEPA update on worker (embedding extract + backward + EMA sync)
                        with simple_timer("jepa_update", timing_raw):
                            jepa_td = jepa_batch.to_tensordict()
                            # Broadcast to match jepa_td's batch dim (TensorDict requires
                            # assigned tensors to share the leading batch_size shape).
                            jepa_td["global_step"] = torch.full(
                                (jepa_td.batch_size[0],), float(self.global_steps)
                            )
                            # Per-arm plateau latches -> worker zeroes the disabled align
                            # term's gradient (preds still feed SIGReg). dual mode only.
                            if self.jepa_cfg.loss_type == "jepa-tcr-dual":
                                _bs = jepa_td.batch_size[0]
                                jepa_td["align_cot_on"] = torch.full((_bs,), float(not self._off_arm["cot"]))
                                jepa_td["align_code_on"] = torch.full((_bs,), float(not self._off_arm["code"]))
                                jepa_td["self_on"] = torch.full((_bs,), float(not self._off_arm["self"]))
                            jepa_output = self.actor_rollout_wg.jepa_update(jepa_td)
                            # ONE_TO_ALL dispatch returns a list; take rank-0 output
                            if isinstance(jepa_output, list):
                                jepa_output = jepa_output[0] if jepa_output else None
                            if jepa_output is not None:
                                for k, v in jepa_output.items():
                                    if isinstance(v, torch.Tensor):
                                        metrics[k] = float(v.item())
                            if "n_correct_cot_mean" in jepa_batch.meta_info:
                                metrics["jepa/n_correct_cot_mean"] = jepa_batch.meta_info["n_correct_cot_mean"]
                    else:
                        metrics["jepa/skipped"] = 1.0
                        metrics["jepa/n_valid_pairs"] = 0.0

                    # NOTE: do NOT sleep the rollout replicas here. vLLM was already put
                    # to sleep at sleep_replicas_1 (above, before update_actor) and nothing
                    # wakes it again until weight_sync_2 at the end of the step. The
                    # pre-unification loop had a weight_sync_1 (wake) + a separate JEPA-only
                    # code generation between the two sleeps, so this second sleep acted on an
                    # AWAKE engine; unification removed that wake+generation but left this
                    # sleep behind. Sleeping an already-slept engine reproducibly segfaults
                    # vLLM's executor in cuMemcpy at the first training step (EngineCore dies
                    # -> "collective_rpc sleep ... cancelled" -> EngineDeadError). jepa_update
                    # is a worker-side FSDP RPC and does not touch the rollout engine.

                # Track code/cot accuracy (sliced from the single combined reward_tensor)
                code_rew_scalar = rew_scalar_all[torch.from_numpy(code_mask_rows)]
                metrics["code/pass_at_1"] = float((code_rew_scalar > 0).float().mean()) if len(code_rew_scalar) else 0.0
                metrics["code/avg_reward"] = float(code_rew_scalar.mean()) if len(code_rew_scalar) else 0.0

                # ── Step 6: Weight sync to rollout (wakes vLLM) ─────────
                self._log_gpu_mem("before_weight_sync_2")
                with simple_timer("weight_sync_2", timing_raw):
                    self.checkpoint_manager.update_weights(self.global_steps)
                self._log_gpu_mem("after_weight_sync_2")

                # ── CoT-based train metrics (rich grouped stats) ─────────
                cot_rew_scalar = rew_scalar_all[torch.from_numpy(cot_mask_rows)]
                metrics["cot/pass_at_1"] = float((cot_rew_scalar > 0).float().mean()) if len(cot_rew_scalar) else 0.0
                metrics["cot/avg_reward"] = float(cot_rew_scalar.mean()) if len(cot_rew_scalar) else 0.0
                # NOTE: compute_data_metrics/_compute_train_comparison_metrics below now span
                # the FULL combined (cot+code) batch, not cot-only as before this change — this
                # is intended (both modalities now receive gradient), but means train/accuracy,
                # response_length/* etc. will show a discontinuity at the cutover step in wandb.
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(self._compute_train_comparison_metrics(batch))
                metrics["train/global_step"] = self.global_steps

                # Auto-disable the JEPA aux signal (shaping/differentiable) once the
                # teacher-alignment metric this step has plateaued. Evaluated AFTER the
                # step's shaping/jepa metrics are in `metrics`; latches for all later steps.
                self._maybe_disable_jepa_signal(metrics)

                # ── Validation ───────────────────────────────────────────
                is_last_step = self.global_steps >= self.total_training_steps
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with simple_timer("validate", timing_raw):
                        val_metrics = self._validate()
                    metrics.update(val_metrics)
                    self._maybe_save_best_checkpoint(val_metrics)

                # ── Checkpoint ───────────────────────────────────────────
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with simple_timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()

                metrics.update({f"timing_s/{k}": v for k, v in timing_raw.items()})
                # Sum only top-level segments. The per-call generate_sequences
                # diagnostics merged in above as "cot_gen/*" / "code_gen/*" are
                # sub-timings ALREADY contained in the top-level "cot_gen" /
                # "code_gen" timers — including them double-counts and inflated
                # step_total to ~minutes (3625s observed). Drop any key with "/".
                metrics["timing_s/step_total"] = sum(
                    v for k, v in timing_raw.items() if "/" not in k
                )

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                # The full metrics dict (~150 keys) on the tqdm postfix floods the
                # console and slows the terminal. Only show it when explicitly opted in
                # via VERL_CONSOLE_FULL_METRICS=1; otherwise a tiny curated subset.
                if os.environ.get("VERL_CONSOLE_FULL_METRICS", "0") == "1":
                    progress_bar.set_postfix(metrics)
                else:
                    _short = {
                        k: round(metrics[k], 4)
                        for k in ("actor/loss", "actor/grad_norm", "jepa/tcr_loss",
                                  "jepa/grad_norm", "train/accuracy", "timing_s/step_total")
                        if k in metrics
                    }
                    progress_bar.set_postfix(_short)

                if is_last_step:
                    return

                self.global_steps += 1
