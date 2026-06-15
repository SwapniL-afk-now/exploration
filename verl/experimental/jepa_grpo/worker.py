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
  - Hook-based embedding extraction on the final norm (avoids output_hidden_states=True overhead)
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

    def _final_norm(self) -> torch.nn.Module:
        """Return the final layer norm of the base transformer.

        Mirrors the FSDP-unwrap logic in _last_layer() but targets the norm
        applied after all decoder layers (the one that produces last_hidden_state).
        """
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        try:
            return inner.model.model.norm  # Qwen2-style double nesting
        except AttributeError:
            return inner.model.norm        # LLaMA-style

    def _decoder_layers(self) -> list:
        """Return the list of decoder layer modules (Qwen2DecoderLayer etc.)."""
        module = self.actor.engine.module
        inner = getattr(module, "_fsdp_wrapped_module", module)
        for attr in ("model.model.layers", "model.layers"):
            obj = inner
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                pass
        return []

    @contextlib.contextmanager
    def _no_gc_ctx(self):
        """Temporarily disable gradient checkpointing on each decoder layer.

        Patches the per-layer ``gradient_checkpointing`` boolean directly instead
        of calling ``model.gradient_checkpointing_disable()``.  The HF API also
        removes ``enable_input_require_grads`` hooks which breaks LoRA backward;
        this surgical patch avoids that side effect.

        With GC off, the forward stores all intermediate activations and backward
        runs without recompute — eliminating the cudaErrorIllegalAddress that
        FSDP1's GC-recompute path triggers on single-GPU NO_SHARD configurations
        when backward is computed from an intermediate hook capture rather than
        the model's final output tensor.

        Note: each element of ``self.layers`` is an FSDP wrapper around the
        actual Qwen2DecoderLayer.  We must unwrap via ``_fsdp_wrapped_module``
        to reach the ``GradientCheckpointingLayer`` that owns the flag.
        """
        layers = self._decoder_layers()
        gc_states = {}
        for fsdp_layer in layers:
            # Unwrap FSDP to reach the GradientCheckpointingLayer
            layer = getattr(fsdp_layer, "_fsdp_wrapped_module", fsdp_layer)
            if hasattr(layer, "gradient_checkpointing"):
                gc_states[id(layer)] = layer.gradient_checkpointing
                layer.gradient_checkpointing = False
        try:
            yield
        finally:
            for fsdp_layer in layers:
                layer = getattr(fsdp_layer, "_fsdp_wrapped_module", fsdp_layer)
                if id(layer) in gc_states:
                    layer.gradient_checkpointing = gc_states[id(layer)]

    @contextlib.contextmanager
    def _ema_ctx(self):
        """Context manager: temporarily overwrite live LoRA params with EMA values.

        Uses in-place copy (not data-pointer swap) so that FSDP1's FlatParameter
        view pointers remain intact.  Swapping p.data to a different tensor would
        redirect the parameter away from FSDP1's managed flat storage.
        """
        saved = {}
        for n, p in self._adapter_named_params():
            saved[n] = p.data.clone()                                   # save live values
            p.data.copy_(self.ema_weights[n].to(p.device, dtype=p.dtype))  # overwrite in-place
        try:
            yield
        finally:
            for n, p in self._adapter_named_params():
                p.data.copy_(saved[n])                                  # restore live values in-place

    @torch.no_grad()
    def _sync_ema(self) -> None:
        """Exponential moving average update: ema ← decay*ema + (1-decay)*live."""
        decay = self.jepa_cfg.ema_decay
        for n, p in self._adapter_named_params():
            self.ema_weights[n].mul_(decay).add_(p.data.detach(), alpha=1.0 - decay)

    # ------------------------------------------------ embedding extraction ---
    def _pack_left_padded(
        self,
        input_ids: torch.Tensor,   # (B, L) left-padded
        seq_lengths: torch.Tensor, # (B,) real lengths
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove left-padding and pack into verl's (1, total_nnz) format.

        verl's monkey-patched flash attention (use_remove_padding=True) expects
        packed input with no attention_mask — it infers cu_seqlens from
        position_ids.  Passing a padded [B, L] tensor + attention_mask instead
        triggers HuggingFace's _upad_input → torch.nonzero, which causes a
        cudaErrorIllegalAddress on the FSDP-managed CUDA context.

        Returns:
            packed_ids:  (1, total_nnz) token ids, padding removed
            packed_pos:  (1, total_nnz) position ids, 0-indexed per sequence
            last_idx:    (B,) index in the packed dim of each sequence's last token
        """
        B, L = input_ids.shape
        chunks, pos_chunks, last_indices = [], [], []
        offset = 0
        for b in range(B):
            rlen = int(seq_lengths[b].item())
            pad = L - rlen
            chunks.append(input_ids[b, pad:])                              # real tokens only
            pos_chunks.append(torch.arange(rlen, device=device, dtype=torch.long))
            offset += rlen
            last_indices.append(offset - 1)

        packed_ids = torch.cat(chunks).unsqueeze(0)           # (1, total_nnz)
        packed_pos = torch.cat(pos_chunks).unsqueeze(0)       # (1, total_nnz)
        last_idx   = torch.tensor(last_indices, device=device) # (B,)
        return packed_ids, packed_pos, last_idx

    def _extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        use_ema: bool,
        requires_grad: bool,
    ) -> torch.Tensor:
        """Run the policy model on the full batch and return last-token embeddings.

        All N sequences are processed in a SINGLE forward call so that FSDP1
        sees exactly one forward per backward — its expected usage pattern.
        Running N separate forwards before one backward confuses FSDP1's hook
        state machine and causes CUBLAS_STATUS_INTERNAL_ERROR.

        Sequences are re-packed into an (N, max_rlen) tensor with real tokens
        at the start (right-padded with zeros).  A monotonically-increasing
        position_ids shared across rows ensures _is_packed_sequence() returns
        False, routing flash attention through flash_attn_func (not varlen).
        A proper 2D attention_mask prevents padding positions from polluting
        the representations of shorter sequences.

        Args:
            input_ids: (N, L) token ids, may be left or right padded
            attention_mask: (N, L) 1 for real tokens, 0 for padding
            seq_lengths: (N,) actual sequence lengths (= attention_mask.sum per row)
            use_ema: if True, temporarily swap in EMA weights
            requires_grad: if False, wrap forward in torch.no_grad()

        Returns:
            (N, d) unit-normalised embeddings (float32)
        """
        device = next(self.actor.engine.module.parameters()).device
        N = input_ids.shape[0]
        rlens = [int(seq_lengths[b].item()) for b in range(N)]
        max_rlen = max(rlens) if rlens else 1

        # Build (N, max_rlen) batch: real tokens contiguous at start, zeros for padding.
        # Works for both left-padded and right-padded inputs via the attention_mask.
        packed_ids = torch.zeros(N, max_rlen, dtype=input_ids.dtype, device=device)
        packed_attn = torch.zeros(N, max_rlen, dtype=torch.long, device=device)
        for b, rlen in enumerate(rlens):
            if rlen > 0:
                real_toks = input_ids[b][attention_mask[b].bool()].to(device)  # (rlen,)
                packed_ids[b, :rlen] = real_toks
                packed_attn[b, :rlen] = 1

        # Monotonically increasing position_ids for ALL positions (including padding).
        # _is_packed_sequence() returns False for monotonic positions → regular
        # flash_attn_func, not the varlen path.
        packed_pos = torch.arange(max_rlen, device=device, dtype=torch.long).unsqueeze(0).expand(N, -1)

        norm = self._final_norm()
        captured: dict[str, torch.Tensor] = {}

        def _hook(module, inp, out, _cap=captured):
            _cap["h"] = out[0] if isinstance(out, tuple) else out

        ema_ctx = self._ema_ctx() if use_ema else contextlib.nullcontext()
        grad_ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()

        handle = norm.register_forward_hook(_hook)
        try:
            with ema_ctx, grad_ctx, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.actor.engine.module(
                    input_ids=packed_ids,
                    attention_mask=packed_attn,
                    position_ids=packed_pos,
                    use_cache=False,
                )
        finally:
            handle.remove()

        last_h = captured["h"].float()   # (N, max_rlen, d)

        # Keep a zero-valued scalar connected to the model's returned logits.
        # This forces backward to flow through the model's actual output tensor,
        # which is required for FSDP1's root-module post-backward hook to fire.
        # Without it, backward enters from the intermediate captured["h"] and
        # skips the root FSDP output, leaving its state machine in an
        # inconsistent state that causes cudaErrorIllegalAddress.
        # The multiplier is exactly 0.0 so this contributes zero gradient.
        logits_anchor = outputs.logits[:, 0, 0].sum() * 0.0

        all_embs = []
        for b, rlen in enumerate(rlens):
            if rlen == 0:
                all_embs.append(torch.zeros(last_h.shape[-1], device=device))
            else:
                emb = torch.nn.functional.normalize(last_h[b, rlen - 1, :], dim=-1)
                all_embs.append(emb)

        return torch.stack(all_embs, dim=0), logits_anchor  # (N, d), scalar

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
            enc_q_cot, logits_anchor = self._extract_embeddings(
                data["cot_input_ids"],
                data["cot_attn_mask"],
                data["cot_lengths"],
                use_ema=False,
                requires_grad=True,
            )

            # -- Pass 2: Code prompt+response → enc_a_code  (EMA, grad=True) --
            enc_a_code, logits_anchor2 = self._extract_embeddings(
                data["code_input_ids"],
                data["code_attn_mask"],
                data["code_lengths"],
                use_ema=True,
                requires_grad=True,
            )

            # -- LeJEPA loss --
            all_pool = torch.cat([enc_q_cot, enc_a_code], dim=0)
            cfg = self.jepa_cfg
            loss, jepa_metrics = lejepa_loss(
                enc_q_cot=enc_q_cot,
                enc_a_code=enc_a_code,
                all_embeddings=all_pool,
                lambda_=cfg.sigreg_lambda,
                M=cfg.n_projections,
                t_min=cfg.t_min,
                t_max=cfg.t_max,
                s=cfg.epps_pulley_s,
            )

            # Both logits anchors tie backward through each forward's root FSDP output.
            scaled_loss = cfg.alpha * loss + logits_anchor + logits_anchor2
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
