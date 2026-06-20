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

import uuid
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

    # ------------------------------------------------------ worker setup ----
    def init_workers(self):
        super().init_workers()
        if self.jepa_cfg.enable:
            import dataclasses

            if self.jepa_cfg.loss_type in ("llm-jepa-loss", "jepa-triplet-loss", "jepa-separation-loss") and self.jepa_cfg.predictor_k > 0:
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

    def _build_jepa_batch(
        self,
        batch_cot: DataProto,
        batch_code: DataProto,
        reward_tensor_cot: torch.Tensor,
        reward_tensor_code: torch.Tensor,
    ) -> DataProto | None:
        """Build per-prompt JEPA pairs from CoT and Code rollout batches.

        Only prompts where at least one CoT rollout AND at least one Code
        rollout are correct are included.

        The CoT view embedding uses the CoT PROMPT tokens (same for all n
        rollouts of the same prompt).  The Code view embedding uses the full
        Code sequence (prompt + first correct response).

        Returns None if fewer than min_valid_pairs pairs exist.
        """
        rollout_n = self.config.actor_rollout_ref.rollout.n
        n_prompts = len(batch_cot) // rollout_n

        # Rewards are (n_prompts * rollout_n,) — scalar per sequence
        rew_cot = reward_tensor_cot.sum(dim=-1)   # (B,) summed over token dim
        rew_code = reward_tensor_code.sum(dim=-1)

        # Reshape to (n_prompts, rollout_n)
        rew_cot_grouped = rew_cot.view(n_prompts, rollout_n)    # (P, G)
        cot_any_correct = (rew_cot_grouped > 0).any(dim=-1)     # (P,)

        # Code batch may also have rollout_n samples per prompt
        code_rollout_n = len(batch_code) // n_prompts
        rew_code_grouped = rew_code.view(n_prompts, code_rollout_n)  # (P, G_code)
        code_any_correct = (rew_code_grouped > 0).any(dim=-1)        # (P,)

        valid_mask = cot_any_correct & code_any_correct  # (P,)
        valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)

        if len(valid_indices) < self.jepa_cfg.min_valid_pairs:
            return None

        # -- CoT view: encode the CoT PROMPT tokens of each valid prompt --
        # Take the first rollout's input_ids up to the prompt (response_mask tells us)
        cot_input_ids = batch_cot.batch["input_ids"]        # (B, L)
        response_mask = batch_cot.batch.get("response_mask", None)

        cot_prompt_ids_list = []
        cot_prompt_mask_list = []
        cot_prompt_lengths = []
        for p_idx in valid_indices.tolist():
            flat_idx = p_idx * rollout_n  # first rollout for this prompt
            ids = cot_input_ids[flat_idx]  # (L,)
            attn = batch_cot.batch["attention_mask"][flat_idx]  # (L,)

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
        code_input_ids = batch_code.batch["input_ids"]     # (B_code, L_code)
        code_attn_mask = batch_code.batch["attention_mask"]

        code_ids_list = []
        code_mask_list = []
        code_lengths = []
        for p_idx in valid_indices.tolist():
            # Find first correct code rollout for this prompt
            first_correct = None
            for g in range(code_rollout_n):
                if rew_code_grouped[p_idx, g] > 0:
                    first_correct = g
                    break
            if first_correct is None:
                first_correct = 0  # fallback (shouldn't happen)
            flat_idx = p_idx * code_rollout_n + first_correct
            ids = code_input_ids[flat_idx]
            attn = code_attn_mask[flat_idx]
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
        batch_cot: DataProto,
        batch_code: DataProto,
        reward_tensor_cot: torch.Tensor,
        reward_tensor_code: torch.Tensor,
    ) -> DataProto | None:
        """Build the jepa-triplet-loss batch: full correct-CoT response, first
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
        rollout_n = self.config.actor_rollout_ref.rollout.n
        n_prompts = len(batch_cot) // rollout_n

        rew_cot = reward_tensor_cot.sum(dim=-1)
        rew_code = reward_tensor_code.sum(dim=-1)

        rew_cot_grouped = rew_cot.view(n_prompts, rollout_n)
        cot_any_correct = (rew_cot_grouped > 0).any(dim=-1)

        code_rollout_n = len(batch_code) // n_prompts
        rew_code_grouped = rew_code.view(n_prompts, code_rollout_n)
        code_any_correct = (rew_code_grouped > 0).any(dim=-1)

        valid_mask = cot_any_correct & code_any_correct
        valid_indices_all = valid_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

        if len(valid_indices_all) < self.jepa_cfg.min_valid_pairs:
            return None

        cot_input_ids = batch_cot.batch["input_ids"]
        cot_attn_mask = batch_cot.batch["attention_mask"]
        code_input_ids = batch_code.batch["input_ids"]
        code_attn_mask = batch_code.batch["attention_mask"]
        code_response_mask = batch_code.batch.get("response_mask", None)
        reward_models = batch_code.non_tensor_batch.get("reward_model", [{}] * len(batch_code))
        data_sources = batch_code.non_tensor_batch.get("data_source", [None] * len(batch_code))

        pad_id = self.tokenizer.pad_token_id or 0

        per_prompt = {}
        triplet_eligible, others = [], []
        for p_idx in valid_indices_all:
            # First correct CoT rollout (full sequence), and a count of how
            # many CoT rollouts were correct (audit metric).
            first_correct_cot = None
            n_correct_cot = 0
            for g in range(rollout_n):
                if rew_cot_grouped[p_idx, g] > 0:
                    n_correct_cot += 1
                    if first_correct_cot is None:
                        first_correct_cot = g
            cot_flat_idx = p_idx * rollout_n + first_correct_cot

            # First correct code rollout (full sequence)
            first_correct_code = None
            for g in range(code_rollout_n):
                if rew_code_grouped[p_idx, g] > 0:
                    first_correct_code = g
                    break
            code_flat_idx = p_idx * code_rollout_n + first_correct_code

            ground_truth = reward_models[code_flat_idx].get("ground_truth") if isinstance(
                reward_models[code_flat_idx], dict
            ) else None
            dataset_kind = data_sources[code_flat_idx]

            # First CLEAN-wrong code rollout: parseable answer extracted, but
            # not correct. Filtered to definite-wrong BEFORE taking "first"
            # (never let a crash/parse-failure stand in as e^w).
            wrong_flat_idx = None
            for g in range(code_rollout_n):
                if rew_code_grouped[p_idx, g] > 0:
                    continue
                cand_idx = p_idx * code_rollout_n + g
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

            per_prompt[p_idx] = (cot_flat_idx, code_flat_idx, wrong_flat_idx, n_correct_cot)
            (triplet_eligible if wrong_flat_idx is not None else others).append(p_idx)

        ordered_indices = triplet_eligible + others   # T-prefix invariant
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
        for p_idx in ordered_indices:
            cot_flat_idx, code_flat_idx, wrong_flat_idx, n_correct_cot = per_prompt[p_idx]
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

                # ── Step 1: CoT rollout ──────────────────────────────────
                with simple_timer("cot_gen", timing_raw):
                    gen_batch = self._get_gen_batch(batch)
                    gen_batch.meta_info["global_steps"] = self.global_steps
                    rollout_n = self.config.actor_rollout_ref.rollout.n
                    gen_batch_rep = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                    cot_gen_output = self.async_rollout_manager.generate_sequences(gen_batch_rep)
                    batch = batch.repeat(repeat_times=rollout_n, interleave=True)
                    batch = batch.union(cot_gen_output)
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                # ── Step 2: CoT rewards & advantages ────────────────────
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

                # ── Step 3: Compute old log-probs & (optional) ref ──────
                with simple_timer("old_log_prob", timing_raw):
                    old_log_prob, _old_log_prob_mfu = self._compute_old_log_prob(batch)
                    batch = batch.union(old_log_prob)
                    if self.use_reference_policy:
                        ref_log_prob = self._compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # Sleep rollout replicas before backward (frees KV cache)
                with simple_timer("sleep_replicas_1", timing_raw):
                    self.checkpoint_manager.sleep_replicas()

                # ── Step 4: GRPO actor update ────────────────────────────
                with simple_timer("update_actor", timing_raw):
                    actor_output = self._update_actor(batch)
                    actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_metrics)

                # ── Step 5: JEPA (if enabled) ────────────────────────────
                if self.jepa_cfg.enable:
                    # 5a. Code-view rollout (wake rollout replicas first)
                    with simple_timer("weight_sync_1", timing_raw):
                        self.checkpoint_manager.update_weights(self.global_steps)

                    with simple_timer("code_gen", timing_raw):
                        code_gen_batch = self._tokenize_code_prompts(
                            # Use un-repeated batch so prompts appear once
                            DataProto(
                                batch=batch.batch[:len(batch.batch) // rollout_n],
                                non_tensor_batch={
                                    k: v[:len(batch.batch) // rollout_n]
                                    for k, v in batch.non_tensor_batch.items()
                                },
                                meta_info=batch.meta_info,
                            )
                        )
                        code_gen_batch_rep = code_gen_batch.repeat(repeat_times=rollout_n, interleave=True)
                        code_gen_output = self.async_rollout_manager.generate_sequences(code_gen_batch_rep)
                        # AgentLoop echoes raw_prompt back into output; remove before union to avoid key collision
                        code_gen_output.non_tensor_batch.pop("raw_prompt", None)

                        code_batch = code_gen_batch.repeat(repeat_times=rollout_n, interleave=True)
                        code_batch = code_batch.union(code_gen_output)
                        if "response_mask" not in code_batch.batch.keys():
                            code_batch.batch["response_mask"] = compute_response_mask(code_batch)

                    # 5b. Code rewards (math match)
                    with simple_timer("code_reward", timing_raw):
                        if self.use_rm and "rm_scores" not in code_batch.batch.keys():
                            code_batch = code_batch.union(self._compute_reward_colocate(code_batch))
                        code_reward_tensor, _ = extract_reward(code_batch)

                    # Sleep rollout before JEPA backward
                    with simple_timer("sleep_replicas_2", timing_raw):
                        self.checkpoint_manager.sleep_replicas()

                    # 5c. Build JEPA pairs
                    with simple_timer("jepa_build_batch", timing_raw):
                        build_fn = (
                            self._build_jepa_batch_triplet
                            if self.jepa_cfg.loss_type in ("jepa-triplet-loss", "jepa-separation-loss")
                            else self._build_jepa_batch
                        )
                        jepa_batch = build_fn(
                            batch_cot=batch,
                            batch_code=code_batch,
                            reward_tensor_cot=reward_tensor,
                            reward_tensor_code=code_reward_tensor,
                        )

                    if jepa_batch is not None:
                        # 5d. JEPA update on worker (embedding extract + backward + EMA sync)
                        with simple_timer("jepa_update", timing_raw):
                            jepa_td = jepa_batch.to_tensordict()
                            # Broadcast to match jepa_td's batch dim (TensorDict requires
                            # assigned tensors to share the leading batch_size shape).
                            jepa_td["global_step"] = torch.full(
                                (jepa_td.batch_size[0],), float(self.global_steps)
                            )
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

                    # Track code accuracy
                    code_rew_scalar = code_reward_tensor.sum(dim=-1)
                    metrics["code/pass_at_1"] = float((code_rew_scalar > 0).float().mean())
                    metrics["code/avg_reward"] = float(code_rew_scalar.mean())

                # ── Step 6: Weight sync to rollout (wakes vLLM) ─────────
                with simple_timer("weight_sync_2", timing_raw):
                    self.checkpoint_manager.update_weights(self.global_steps)

                # ── CoT-based train metrics (rich grouped stats) ─────────
                cot_rew_scalar = reward_tensor.sum(dim=-1)
                metrics["cot/pass_at_1"] = float((cot_rew_scalar > 0).float().mean())
                metrics["cot/avg_reward"] = float(cot_rew_scalar.mean())
                metrics.update(compute_data_metrics(batch=batch, use_critic=False))
                metrics.update(self._compute_train_comparison_metrics(batch))
                metrics["train/global_step"] = self.global_steps

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
                metrics["timing_s/step_total"] = sum(timing_raw.values())

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                progress_bar.set_postfix(metrics)

                if is_last_step:
                    return

                self.global_steps += 1
