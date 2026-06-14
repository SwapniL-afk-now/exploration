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
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.metric import reduce_metrics


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
            cfg_dict = dataclasses.asdict(self.jepa_cfg)
            self.actor_rollout_wg.jepa_init(cfg_dict)

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
                old_log_prob, _old_log_prob_mfu = self._compute_old_log_prob(batch)
                batch = batch.union(old_log_prob)
                if self.use_reference_policy:
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

                # Sleep rollout replicas before backward (frees KV cache)
                self.checkpoint_manager.sleep_replicas()

                # ── Step 4: GRPO actor update ────────────────────────────
                actor_output = self._update_actor(batch)
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_metrics)

                # ── Step 5: JEPA (if enabled) ────────────────────────────
                if self.jepa_cfg.enable:
                    # 5a. Code-view rollout (wake rollout replicas first)
                    self.checkpoint_manager.update_weights(self.global_steps)

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
                    if self.use_rm and "rm_scores" not in code_batch.batch.keys():
                        code_batch = code_batch.union(self._compute_reward_colocate(code_batch))
                    code_reward_tensor, _ = extract_reward(code_batch)

                    # Sleep rollout before JEPA backward
                    self.checkpoint_manager.sleep_replicas()

                    # 5c. Build JEPA pairs
                    jepa_batch = self._build_jepa_batch(
                        batch_cot=batch,
                        batch_code=code_batch,
                        reward_tensor_cot=reward_tensor,
                        reward_tensor_code=code_reward_tensor,
                    )

                    if jepa_batch is not None:
                        # 5d. JEPA update on worker (embedding extract + backward + EMA sync)
                        jepa_td = jepa_batch.to_tensordict()
                        jepa_output = self.actor_rollout_wg.jepa_update(jepa_td)
                        # ONE_TO_ALL dispatch returns a list; take rank-0 output
                        if isinstance(jepa_output, list):
                            jepa_output = jepa_output[0] if jepa_output else None
                        if jepa_output is not None:
                            for k, v in jepa_output.items():
                                if isinstance(v, torch.Tensor):
                                    metrics[k] = float(v.item())
                    else:
                        metrics["jepa/skipped"] = 1.0
                        metrics["jepa/n_valid_pairs"] = 0.0

                    # Track code accuracy
                    code_rew_scalar = code_reward_tensor.sum(dim=-1)
                    metrics["code/pass_at_1"] = float((code_rew_scalar > 0).float().mean())
                    metrics["code/avg_reward"] = float(code_rew_scalar.mean())

                # ── Step 6: Weight sync to rollout (wakes vLLM) ─────────
                self.checkpoint_manager.update_weights(self.global_steps)

                # ── CoT accuracy metrics ─────────────────────────────────
                cot_rew_scalar = reward_tensor.sum(dim=-1)
                metrics["cot/pass_at_1"] = float((cot_rew_scalar > 0).float().mean())
                metrics["cot/avg_reward"] = float(cot_rew_scalar.mean())
                metrics["train/global_step"] = self.global_steps

                # ── Validation ───────────────────────────────────────────
                is_last_step = self.global_steps >= self.total_training_steps
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    val_metrics = self._validate()
                    metrics.update(val_metrics)

                # ── Checkpoint ───────────────────────────────────────────
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    self._save_checkpoint()

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                progress_bar.set_postfix(metrics)

                if is_last_step:
                    return

                self.global_steps += 1
