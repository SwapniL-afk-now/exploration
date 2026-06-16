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
  - jepa_update() RPC: runs a JEPA-only forward+backward+optimizer step, using either
    the LeJEPA loss (default) or the LLM-JEPA paper's prediction loss (jepa.loss_type)
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
from verl.experimental.jepa_grpo.core_algos import lejepa_loss, llm_jepa_loss


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

    @staticmethod
    def _pad_concat_batch(
        ids_a: torch.Tensor, mask_a: torch.Tensor,
        ids_b: torch.Tensor, mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate two (N, L) batches along the batch dim into one (N_a+N_b, L').

        Used to merge the CoT and Code views into a single batch for the
        LLM-JEPA joint live forward (see jepa_update). The shorter of the two
        L dimensions is zero-padded to match; this is safe because
        `_extract_embeddings` only ever reads real tokens via `attention_mask`
        (extra zero-padded columns with mask=0 are simply ignored).
        """
        La, Lb = ids_a.shape[1], ids_b.shape[1]
        L = max(La, Lb)
        if La < L:
            pad = (0, L - La)
            ids_a = torch.nn.functional.pad(ids_a, pad)
            mask_a = torch.nn.functional.pad(mask_a, pad)
        if Lb < L:
            pad = (0, L - Lb)
            ids_b = torch.nn.functional.pad(ids_b, pad)
            mask_b = torch.nn.functional.pad(mask_b, pad)
        return torch.cat([ids_a, ids_b], dim=0), torch.cat([mask_a, mask_b], dim=0)

    def _extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        use_ema: bool,
        requires_grad: bool,
        predictor_k: "int | list[int]" = 0,
        predictor_token_id: int | None = None,
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
            predictor_k: number of LLM-JEPA tied-weight predictor tokens
                (arXiv:2509.14252 §3.1) to append after each row's real tokens.
                Either a single int broadcast to every row, or a per-row
                list/sequence of ints (used to mix predictor and non-predictor
                rows in one joint forward — e.g. CoT rows get k>0, Code rows
                get k=0). k=0 is a no-op — Pred(x) = x, identical to prior
                behavior. When k>0 for a row, ``predictor_token_id`` copies
                are appended and that row's returned embedding is read from
                the LAST predictor token instead of the last real token,
                reusing the model's own weights (no new parameters) as the
                "tied-weight predictor".
            predictor_token_id: token id to repeat for the predictor tokens.
                Required (and otherwise ignored) when any row has predictor_k > 0.

        Returns:
            (N, d) unit-normalised embeddings (float32)
        """
        device = next(self.actor.engine.module.parameters()).device
        N = input_ids.shape[0]
        rlens = [int(seq_lengths[b].item()) for b in range(N)]
        if isinstance(predictor_k, int):
            predictor_ks = [predictor_k] * N
        else:
            predictor_ks = [int(k) for k in predictor_k]
            assert len(predictor_ks) == N, "predictor_k list must match batch size"
        base_max_rlen = max(rlens) if rlens else 1
        max_rlen = base_max_rlen + (max(predictor_ks) if predictor_ks else 0)

        # Build (N, max_rlen) batch: real tokens contiguous at start, zeros for padding.
        # Works for both left-padded and right-padded inputs via the attention_mask.
        # Where predictor_ks[b] > 0, predictor_token_id copies follow that row's
        # real tokens (still causally attending to them) before trailing padding.
        packed_ids = torch.zeros(N, max_rlen, dtype=input_ids.dtype, device=device)
        packed_attn = torch.zeros(N, max_rlen, dtype=torch.long, device=device)
        for b, rlen in enumerate(rlens):
            pk = predictor_ks[b]
            if rlen > 0:
                real_toks = input_ids[b][attention_mask[b].bool()].to(device)  # (rlen,)
                packed_ids[b, :rlen] = real_toks
                packed_attn[b, :rlen] = 1
                if pk > 0:
                    packed_ids[b, rlen:rlen + pk] = predictor_token_id
                    packed_attn[b, rlen:rlen + pk] = 1

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
                # With predictor_ks[b]=0 this is rlen-1 (last real token),
                # matching prior behavior exactly. With predictor_ks[b]>0, this
                # reads that row's last predictor token = Pred(Enc(Text)).
                last_idx = rlen + predictor_ks[b] - 1
                emb = torch.nn.functional.normalize(last_h[b, last_idx, :], dim=-1)
                all_embs.append(emb)

        return torch.stack(all_embs, dim=0), logits_anchor  # (N, d), scalar

    # --------------------------------------------------------- JEPA update ---
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def jepa_update(self, data: TensorDict) -> TensorDict:
        """Run a JEPA-only forward+backward+optimizer step.

        Two mutually-exclusive objectives, selected by ``self.jepa_cfg.loss_type``:
          - "lejepa" (default): squared-Euclidean align between a live CoT
            encoder and an EMA target Code encoder, + SIGReg (unchanged).
          - "llm-jepa-loss": the LLM-JEPA paper's (arXiv:2509.14252) literal
            symmetric architecture — ONE live encoder for both CoT and Code
            (no EMA/target network, no stop-gradient), cosine-distance
            prediction loss between Pred(Enc(CoT)) and Enc(Code), combined
            with SIGReg for anti-collapse since no NTP term is added here
            (see core_algos.llm_jepa_loss).

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
        if self.jepa_cfg.loss_type == "llm-jepa-loss" and self.jepa_cfg.predictor_k > 0:
            assert self.jepa_cfg.predictor_token_id >= 0, (
                "predictor_k > 0 requires a resolved predictor_token_id; "
                "JEPARayPPOTrainer.init_workers() should have set this before jepa_init()"
            )

        n_pairs = data["cot_input_ids"].shape[0]
        if n_pairs < self.jepa_cfg.min_valid_pairs:
            return TensorDict(
                {"jepa/skipped": torch.tensor(1.0),
                 "jepa/n_valid_pairs": torch.tensor(float(n_pairs))},
                batch_size=[],
            )

        engine = self.actor.engine
        cfg = self.jepa_cfg
        use_llm_jepa = cfg.loss_type == "llm-jepa-loss"

        with engine.train_mode():
            engine.optimizer_zero_grad()

            if use_llm_jepa:
                # -- LLM-JEPA (arXiv:2509.14252), literal symmetric architecture:
                # ONE live encoder for both views, no EMA/target network, no
                # stop-gradient — gradient flows through both Enc(Text)=
                # Pred(...) and Enc(Code). CoT and Code rows are packed into a
                # SINGLE joint batch and run through ONE forward call so FSDP1
                # still sees exactly one forward per backward; only the CoT
                # rows get `predictor_k` tied-weight predictor tokens appended
                # (k=0 -> Pred(x) = x, per the paper §3.1).
                n_cot = data["cot_input_ids"].shape[0]
                n_code = data["code_input_ids"].shape[0]
                joint_ids, joint_mask = self._pad_concat_batch(
                    data["cot_input_ids"], data["cot_attn_mask"],
                    data["code_input_ids"], data["code_attn_mask"],
                )
                joint_lengths = torch.cat([data["cot_lengths"], data["code_lengths"]], dim=0)
                joint_predictor_k = [cfg.predictor_k] * n_cot + [0] * n_code

                joint_emb, logits_anchor = self._extract_embeddings(
                    joint_ids,
                    joint_mask,
                    joint_lengths,
                    use_ema=False,
                    requires_grad=True,
                    predictor_k=joint_predictor_k,
                    predictor_token_id=cfg.predictor_token_id,
                )
                enc_q_cot = joint_emb[:n_cot]
                enc_a_code = joint_emb[n_cot:]

                # No .detach() on either view — both contribute gradient,
                # matching the paper's no-stop-gradient design.
                all_pool = torch.cat([enc_q_cot, enc_a_code], dim=0)
                loss, jepa_metrics = llm_jepa_loss(
                    pred_text=enc_q_cot,
                    enc_code=enc_a_code,
                    all_embeddings=all_pool,
                    lambda_=cfg.sigreg_lambda,
                    M=cfg.n_projections,
                    t_min=cfg.t_min,
                    t_max=cfg.t_max,
                    s=cfg.epps_pulley_s,
                )
            else:
                # -- LeJEPA (default, unchanged): live CoT encoder + EMA Code
                # target encoder, two separate forwards (Code pass is
                # no_grad so it never enters the autograd/FSDP1 hook graph).
                enc_q_cot, logits_anchor = self._extract_embeddings(
                    data["cot_input_ids"],
                    data["cot_attn_mask"],
                    data["cot_lengths"],
                    use_ema=False,
                    requires_grad=True,
                )
                enc_a_code, _ = self._extract_embeddings(
                    data["code_input_ids"],
                    data["code_attn_mask"],
                    data["code_lengths"],
                    use_ema=True,
                    requires_grad=False,
                )
                all_pool = torch.cat([enc_q_cot, enc_a_code.detach()], dim=0)
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

            # logits_anchor (= 0 * logits_scalar) ties backward to the root FSDP
            # module's actual output so its post-backward hook fires correctly.
            scaled_loss = cfg.alpha * loss + logits_anchor
            scaled_loss.backward()
            grad_norm = engine.optimizer_step()

        # Update EMA after optimizer step
        self._sync_ema()

        out = {
            # Generic key valid for either loss_type; mode-specific totals
            # ("jepa/lejepa_loss" / "jepa/llm_jepa_loss") are also present via
            # jepa_metrics below.
            "jepa/total_loss": torch.tensor(loss.detach().item()),
            "jepa/n_valid_pairs": torch.tensor(float(n_pairs)),
            "jepa/skipped": torch.tensor(0.0),
            "jepa/grad_norm": torch.tensor(float(grad_norm) if grad_norm is not None else 0.0),
        }
        out.update({f"jepa/{k}": torch.tensor(float(v)) for k, v in jepa_metrics.items()})
        return TensorDict(out, batch_size=[])
