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
"""JEPA-GRPO Ray worker.

Extends ActorRolloutRefWorker with:
  - EMA target encoder for the Code view (stored as plain param dict, not PEFT adapter)
  - Last-layer hook-based embedding extraction (avoids output_hidden_states=True overhead)
  - jepa_update() RPC: runs a JEPA-only forward+backward+optimizer step
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.core_algos import lejepa_loss


class JEPAActorRolloutRefWorker(ActorRolloutRefWorker):
    """Extends ActorRolloutRefWorker with JEPA embedding extraction and EMA target encoder.

    The standard GRPO loss (update_actor) is unchanged.  A new ``jepa_update``
    RPC runs a *separate* backward+optimizer-step for the LeJEPA loss, keeping
    the implementation self-contained without touching the FSDP engine internals.

    EMA is stored as a plain ``dict[str, Tensor]`` keyed on LoRA adapter param
    names so vLLM never sees unknown weight keys.
    """

    # ------------------------------------------------------------------ init --
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        # Initialised lazily on first jepa_init call so config is available
        self.ema_weights: Optional[dict[str, torch.Tensor]] = None
        self.jepa_cfg: Optional[JEPARayConfig] = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def jepa_init(self, jepa_cfg_dict: dict) -> None:
        """Initialise EMA weights from current LoRA adapter parameters."""
        self.jepa_cfg = JEPARayConfig(**jepa_cfg_dict)
        self.ema_weights = {
            n: p.data.clone().detach()
            for n, p in self._adapter_named_params()
        }

    # --------------------------------------------------------------- helpers --
    def _adapter_named_params(self):
        """Yield (name, param) for LoRA adapter parameters only."""
        module = self.actor.engine.module
        # FSDP wraps the module; unwrap to access peft internals
        inner = getattr(module, "_fsdp_wrapped_module", module)
        for n, p in inner.named_parameters():
            if "lora_" in n:
                yield n, p

    def _last_layer(self) -> torch.nn.Module:
        """Return the last transformer decoder layer of the policy model."""
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        # Handles Qwen2/LLaMA-style naming: model.model.layers or model.layers
        try:
            return inner.model.model.layers[-1]
        except AttributeError:
            return inner.model.layers[-1]

    @contextlib.contextmanager
    def _ema_ctx(self):
        """Context manager: temporarily swap live LoRA params for EMA copies."""
        saved = {}
        for n, p in self._adapter_named_params():
            saved[n] = p.data
            p.data = self.ema_weights[n].to(p.device, dtype=p.dtype)
        try:
            yield
        finally:
            for n, p in self._adapter_named_params():
                p.data = saved[n]

    @torch.no_grad()
    def _sync_ema(self) -> None:
        """Exponential moving average update: ema ← decay*ema + (1-decay)*live."""
        decay = self.jepa_cfg.ema_decay
        for n, p in self._adapter_named_params():
            self.ema_weights[n].mul_(decay).add_(p.data.detach(), alpha=1.0 - decay)

    # ------------------------------------------------ embedding extraction ---
    def _extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        use_ema: bool,
        requires_grad: bool,
    ) -> torch.Tensor:
        """Run the policy model and return last-token embeddings.

        Uses output_hidden_states=True so the returned hidden states are always
        (B, T, d) regardless of use_remove_padding internals.

        Args:
            input_ids: (N, L) padded token ids
            attention_mask: (N, L) attention mask
            seq_lengths: (N,) actual sequence lengths (1-indexed)
            use_ema: if True, swap in EMA weights before the forward pass
            requires_grad: if False, wrap forward in torch.no_grad()

        Returns:
            (N, d) unit-normalised embeddings (float32)
        """
        mb = self.jepa_cfg.embed_micro_batch_size
        device = next(self.actor.engine.module.parameters()).device
        all_embs = []

        for start in range(0, input_ids.shape[0], mb):
            ids = input_ids[start:start + mb].to(device)
            mask = attention_mask[start:start + mb].to(device)
            lens = seq_lengths[start:start + mb]

            ema_ctx = self._ema_ctx() if use_ema else contextlib.nullcontext()
            grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
            with ema_ctx, grad_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.actor.engine.module(
                    input_ids=ids,
                    attention_mask=mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
            # hidden_states[-1] is always (B, T, d) — model repads internally
            last_h = outputs.hidden_states[-1].float()  # (B, T, d)
            B = ids.shape[0]
            last_pos = (lens - 1).to(device)
            embs = last_h[torch.arange(B, device=device), last_pos, :]  # (B, d)
            embs = torch.nn.functional.normalize(embs, dim=-1)
            all_embs.append(embs)

        return torch.cat(all_embs, dim=0)

    # --------------------------------------------------------- JEPA update ---
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def jepa_update(self, data: TensorDict) -> TensorDict:
        """Run a JEPA-only forward+backward+optimizer step.

        Args:
            data: TensorDict with keys:
                cot_input_ids  (N, L_cot)   — CoT prompt token ids (padded)
                cot_attn_mask  (N, L_cot)   — CoT attention mask
                cot_lengths    (N,)          — actual CoT sequence lengths
                code_input_ids (N, L_code)  — Code prompt+response token ids
                code_attn_mask (N, L_code)  — Code attention mask
                code_lengths   (N,)         — actual Code sequence lengths
                  where N = number of valid (cot_correct AND code_correct) pairs

        Returns:
            TensorDict with JEPA training metrics.
        """
        assert self.jepa_cfg is not None, "Call jepa_init() before jepa_update()"
        assert self.ema_weights is not None, "EMA not initialised"

        n_pairs = data["cot_input_ids"].shape[0]
        if n_pairs < self.jepa_cfg.min_valid_pairs:
            return TensorDict(
                {"jepa/skipped": torch.tensor(1.0),
                 "jepa/n_valid_pairs": torch.tensor(float(n_pairs))},
                batch_size=[],
            )

        engine = self.actor.engine

        with engine.train_mode():
            engine.optimizer_zero_grad()

            # -- Pass 1: CoT prompt → enc_q_cot  (live weights, grad=True) --
            enc_q_cot = self._extract_embeddings(
                data["cot_input_ids"],
                data["cot_attn_mask"],
                data["cot_lengths"],
                use_ema=False,
                requires_grad=True,
            )

            # -- Pass 2: Code prompt+response → enc_a_code  (EMA, grad=False) --
            enc_a_code = self._extract_embeddings(
                data["code_input_ids"],
                data["code_attn_mask"],
                data["code_lengths"],
                use_ema=True,
                requires_grad=False,
            )

            # -- LeJEPA loss --
            all_pool = torch.cat([enc_q_cot, enc_a_code.detach()], dim=0)
            cfg = self.jepa_cfg
            loss, jepa_metrics = lejepa_loss(
                enc_q_cot=enc_q_cot,
                enc_a_code=enc_a_code.detach(),
                all_embeddings=all_pool,
                lambda_=cfg.sigreg_lambda,
                M=cfg.n_projections,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                s=cfg.epps_pulley_s,
            )

            scaled_loss = cfg.alpha * loss
            scaled_loss.backward()
            grad_norm = engine.optimizer_step()

        # Update EMA after optimizer step
        self._sync_ema()

        out = {
            "jepa/lejepa_loss": torch.tensor(loss.detach().item()),
            "jepa/n_valid_pairs": torch.tensor(float(n_pairs)),
            "jepa/skipped": torch.tensor(0.0),
            "jepa/grad_norm": torch.tensor(float(grad_norm) if grad_norm is not None else 0.0),
        }
        out.update({f"jepa/{k}": torch.tensor(float(v)) for k, v in jepa_metrics.items()})
        return TensorDict(out, batch_size=[])
