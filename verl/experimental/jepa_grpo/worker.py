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
from typing import Callable, Optional

import torch
from tensordict import TensorDict

from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils import tensordict_utils as tu
from verl.utils.memory_utils import aggressive_empty_cache
from verl.workers.engine_workers import ActorRolloutRefWorker

from verl.experimental.jepa_grpo.config_ray import JEPARayConfig
from verl.experimental.jepa_grpo.core_algos import (
    llm_jepa_tcr_dual_loss,
)


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
    def _pad_concat_batches(
        groups: "list[tuple[torch.Tensor, torch.Tensor]]",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Concatenate N (N_i, L_i) batches along the batch dim into one (sum(N_i), L').

        Used to merge the CoT, Code, and (for the triplet mode) wrong-Code
        views into a single batch for a joint live forward (see jepa_update).
        Every group's L dimension is zero-padded to the longest one; this is
        safe because `_extract_embeddings` only ever reads real tokens via
        `attention_mask` (extra zero-padded columns with mask=0 are simply
        ignored).
        """
        L = max(ids.shape[1] for ids, _ in groups)
        padded_ids, padded_mask = [], []
        for ids, mask in groups:
            pad = (0, L - ids.shape[1])
            padded_ids.append(torch.nn.functional.pad(ids, pad) if pad[1] else ids)
            padded_mask.append(torch.nn.functional.pad(mask, pad) if pad[1] else mask)
        return torch.cat(padded_ids, dim=0), torch.cat(padded_mask, dim=0)

    def _pad_concat_batch(
        self,
        ids_a: torch.Tensor, mask_a: torch.Tensor,
        ids_b: torch.Tensor, mask_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-group convenience wrapper around `_pad_concat_batches`."""
        return self._pad_concat_batches([(ids_a, mask_a), (ids_b, mask_b)])

    def _extract_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_lengths: torch.Tensor,
        use_ema: bool,
        requires_grad: bool,
        predictor_k: "int | list[int]" = 0,
        predictor_token_id: int | None = None,
        also_boundary: bool = False,
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
            also_boundary: if True, ALSO return the per-row BOUNDARY embedding —
                the last REAL token (index rlen-1), i.e. the read taken BEFORE any
                appended [PRED] tokens. This is a free second read of the same
                packed forward (used for the jepa-tcr-dual self-consistency target
                z_self = Enc(Code_S) boundary). No-op cost: the hidden states are
                already materialized. Changes the return arity (see Returns).

        Returns:
            also_boundary=False: ((N, d) [PRED]/last-token embeddings, logits_anchor)
            also_boundary=True:  ((N, d) [PRED] embeddings, (N, d) boundary
                                  embeddings, logits_anchor)
            All embeddings are unit-normalised float32.
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
                    # Only the anchor scalar below needs *a* logits tensor, not the
                    # full (N, max_rlen, vocab) one HF computes by default (logits_to_keep=0).
                    # With long packed batches this materializes tens of GB just to be
                    # multiplied by zero — logits_to_keep=1 keeps only the last position's
                    # logits, cutting that allocation by a factor of max_rlen.
                    logits_to_keep=1,
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
        boundary_embs = [] if also_boundary else None
        for b, rlen in enumerate(rlens):
            if rlen == 0:
                all_embs.append(torch.zeros(last_h.shape[-1], device=device))
                if also_boundary:
                    boundary_embs.append(torch.zeros(last_h.shape[-1], device=device))
            else:
                # With predictor_ks[b]=0 this is rlen-1 (last real token),
                # matching prior behavior exactly. With predictor_ks[b]>0, this
                # reads that row's last predictor token = Pred(Enc(Text)).
                last_idx = rlen + predictor_ks[b] - 1
                emb = torch.nn.functional.normalize(last_h[b, last_idx, :], dim=-1)
                all_embs.append(emb)
                if also_boundary:
                    # Boundary = last REAL token (pre-[PRED]); equals `emb` when k=0.
                    bnd = torch.nn.functional.normalize(last_h[b, rlen - 1, :], dim=-1)
                    boundary_embs.append(bnd)

        if also_boundary:
            return torch.stack(all_embs, dim=0), torch.stack(boundary_embs, dim=0), logits_anchor
        return torch.stack(all_embs, dim=0), logits_anchor  # (N, d), scalar

    def _embed_chunked_no_grad(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        use_ema: bool,
        micro_bs: int,
        predictor_k: "int | list[int]" = 0,
        predictor_token_id: int | None = None,
    ) -> torch.Tensor:
        """No-grad embedding extraction, chunked to bound forward activation memory.

        No backward is needed here (used for the EMA Code target encoder in
        "lejepa" mode, and for the jepa-tcr-reward forward-only scoring pass), so
        this is just a plain loop + concat — no GradCache machinery required.

        `predictor_k`/`predictor_token_id` mirror `_extract_embeddings`: pass
        cfg.predictor_k to read the Pred(Enc(text)) [PRED]-token embedding instead
        of the last real token. A per-row list is sliced per chunk.
        """
        N = ids.shape[0]
        per_row_k = isinstance(predictor_k, (list, tuple))

        def _k(start, end):
            return list(predictor_k[start:end]) if per_row_k else predictor_k

        if N <= micro_bs:
            emb, _ = self._extract_embeddings(
                ids, mask, lengths, use_ema=use_ema, requires_grad=False,
                predictor_k=_k(0, N), predictor_token_id=predictor_token_id,
            )
            return emb
        chunks = []
        for start in range(0, N, micro_bs):
            end = min(start + micro_bs, N)
            emb, _ = self._extract_embeddings(
                ids[start:end], mask[start:end], lengths[start:end], use_ema=use_ema, requires_grad=False,
                predictor_k=_k(start, end), predictor_token_id=predictor_token_id,
            )
            chunks.append(emb)
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------ TCR reward scoring ---
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def score_cot_embeddings(self, data: TensorDict) -> TensorDict:
        """Forward-only [PRED]-token embeddings for jepa-tcr-reward shaping.

        Unlike jepa_update this runs NO backward, NO EMA, and NO optimizer step:
        it just returns p_i = Pred(Enc([x, y_S,i, [PRED]])) for every passed CoT
        rollout row, L2-normalized, so the trainer can score them against the
        cached teacher-correct targets and fold the result into the advantage.

        Args:
            data: TensorDict with cot_input_ids (N, L), cot_attn_mask (N, L),
                cot_lengths (N,).
        Returns:
            TensorDict with key "cot_emb" (N, d) float32 unit embeddings.
        """
        assert self.jepa_cfg is not None, "Call jepa_init() before score_cot_embeddings()"
        engine = self.actor.engine
        cfg = self.jepa_cfg
        micro_bs = max(1, cfg.embed_micro_batch_size)
        N = data["cot_input_ids"].shape[0]
        with engine.eval_mode():
            emb = self._embed_chunked_no_grad(
                data["cot_input_ids"], data["cot_attn_mask"], data["cot_lengths"],
                use_ema=False, micro_bs=micro_bs,
                predictor_k=[cfg.predictor_k] * N,
                predictor_token_id=cfg.predictor_token_id,
            )
        aggressive_empty_cache(force_sync=True)
        return TensorDict({"cot_emb": emb.float().cpu()}, batch_size=[])

    def _embed_chunked_with_backward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        lengths: torch.Tensor,
        predictor_k: "list[int]",
        loss_fn: "Callable[..., tuple[torch.Tensor, dict]]",
        micro_bs: int,
        alpha: float,
        also_boundary: bool = False,
    ) -> dict:
        """Embed the joint (N, L) batch, compute a loss over it, and run backward
        — splitting the forward+backward into micro-batches of size `micro_bs`
        rows when N > micro_bs, so peak activation memory is bounded by one
        micro-batch instead of the full joint batch (the cause of the
        "Tried to allocate ... GiB" OOM during `scaled_loss.backward()` at full
        batch size: 64 prompts x up to 3 views/prompt x up to ~3072+predictor_k
        tokens each, all packed into a single forward, easily exceeds the
        94.97 GiB card once activations for backward are retained).

        Implements GradCache (Gao et al., "Scaling Deep Contrastive Learning
        Batch Size under Memory Limited Setup"): the JEPA/SIGReg/triplet losses
        are global statistics over the *whole* embedding pool, so they can't be
        computed independently per chunk — but the loss's gradient w.r.t. each
        row's embedding CAN be computed cheaply once the full (N, d) embedding
        tensor is assembled (d is tiny vs. activation memory). Three passes:
          1. Forward each chunk (grad enabled) to get its embeddings, then
             immediately detach+clone into a fresh leaf tensor and let that
             chunk's transformer activation graph be freed before processing
             the next chunk.
          2. Concatenate the detached per-chunk leaves into the full (N, d)
             tensor, run the actual loss function on it (cheap — no transformer
             involved), and call .backward(). This populates `.grad` on each
             detached per-chunk leaf with exactly the gradient the full-batch
             loss would have produced for that chunk's rows.
          3. Re-forward each chunk (grad enabled, same inputs) and backward
             using that cached `.grad` as the upstream gradient — this is the
             ONLY pass that touches model parameters, and it only ever holds
             one chunk's activations at a time. The per-chunk `logits_anchor`
             is included in this same backward call (see `_extract_embeddings`)
             so FSDP1's root-module post-backward hook still fires correctly
             for every chunk.

        Trades 2x forward compute (each chunk's transformer forward runs twice:
        once to cache embeddings, once for the real backward) for activation
        memory bounded by `micro_bs` rows instead of N rows — exact, not an
        approximation, since the loss is computed once on the full pool.

        When ``also_boundary`` is True, the per-row BOUNDARY embedding (last real
        token, pre-[PRED]) is captured DETACHED from the SAME pass-1 forward (no
        extra forward) and ``loss_fn`` is called as ``loss_fn(joint_emb, boundary)``
        instead of ``loss_fn(joint_emb)``. Used by jepa-tcr-dual so the stop-grad
        self-consistency target z_self rides for free on the encode that already
        produces the [PRED] reads.

        Returns the metrics dict from `loss_fn`; backward is a side effect, the
        caller must still call `engine.optimizer_step()` afterward.
        """
        N = ids.shape[0]
        cfg = self.jepa_cfg

        def _run_loss(emb, boundary):
            return loss_fn(emb, boundary) if also_boundary else loss_fn(emb)

        if N <= micro_bs:
            if also_boundary:
                joint_emb, boundary, logits_anchor = self._extract_embeddings(
                    ids, mask, lengths, use_ema=False, requires_grad=True,
                    predictor_k=predictor_k, predictor_token_id=cfg.predictor_token_id,
                    also_boundary=True,
                )
                boundary = boundary.detach()
            else:
                joint_emb, logits_anchor = self._extract_embeddings(
                    ids, mask, lengths, use_ema=False, requires_grad=True,
                    predictor_k=predictor_k, predictor_token_id=cfg.predictor_token_id,
                )
                boundary = None
            loss, metrics = _run_loss(joint_emb, boundary)
            (alpha * loss + logits_anchor).backward()
            return metrics

        bounds = [(s, min(s + micro_bs, N)) for s in range(0, N, micro_bs)]

        # Pass 1: cache detached per-chunk embeddings (each chunk's forward
        # graph is freed once its embedding is detached and we move on). The
        # boundary read (also detached) is captured here too when requested.
        cached = []
        boundary_chunks = [] if also_boundary else None
        for start, end in bounds:
            if also_boundary:
                chunk_emb, chunk_bnd, _ = self._extract_embeddings(
                    ids[start:end], mask[start:end], lengths[start:end], use_ema=False, requires_grad=True,
                    predictor_k=predictor_k[start:end], predictor_token_id=cfg.predictor_token_id,
                    also_boundary=True,
                )
                boundary_chunks.append(chunk_bnd.detach())
            else:
                chunk_emb, _ = self._extract_embeddings(
                    ids[start:end], mask[start:end], lengths[start:end], use_ema=False, requires_grad=True,
                    predictor_k=predictor_k[start:end], predictor_token_id=cfg.predictor_token_id,
                )
            cached.append(chunk_emb.detach().clone().requires_grad_(True))

        # Pass 2: compute the real loss on the full assembled pool, backward
        # into the cached per-chunk leaves only (cheap — no transformer graph).
        joint_emb_cached = torch.cat(cached, dim=0)
        boundary_all = torch.cat(boundary_chunks, dim=0) if also_boundary else None
        loss, metrics = _run_loss(joint_emb_cached, boundary_all)
        (alpha * loss).backward()

        # Pass 3: re-forward each chunk live and backprop the cached gradient
        # through it into the model parameters, one chunk's activations at a time.
        for (start, end), cached_chunk in zip(bounds, cached):
            chunk_emb_live, chunk_logits_anchor = self._extract_embeddings(
                ids[start:end], mask[start:end], lengths[start:end], use_ema=False, requires_grad=True,
                predictor_k=predictor_k[start:end], predictor_token_id=cfg.predictor_token_id,
            )
            torch.autograd.backward(
                [chunk_emb_live, chunk_logits_anchor],
                [cached_chunk.grad, torch.ones((), device=chunk_logits_anchor.device, dtype=chunk_logits_anchor.dtype)],
            )

        return metrics

    # --------------------------------------------------------- JEPA update ---
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def jepa_update(self, data: TensorDict) -> TensorDict:
        """Run a JEPA-only forward+backward+optimizer step.

        Objective: ``jepa-tcr-dual`` (the only supported loss_type) — ONE live
        encoder for all views (no EMA/target network on the loss path), predictor
        tokens on the CoT and Code anchor rows so each view's p = Pred(Enc(student
        response)). Each view's pred is pulled toward its OWN cached teacher-correct
        target (align_cot / align_code), plus a self-consistency pull of pred_CoT
        toward the stop-grad Code_S boundary read, with SIGReg over [pred_cot,
        pred_code] for anti-collapse (see core_algos.llm_jepa_tcr_dual_loss).

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
        assert self.jepa_cfg.loss_type == "jepa-tcr-dual", (
            f"worker.jepa_update only supports loss_type 'jepa-tcr-dual', "
            f"got {self.jepa_cfg.loss_type!r}"
        )
        if self.jepa_cfg.predictor_k > 0:
            assert self.jepa_cfg.predictor_token_id >= 0, (
                "predictor_k > 0 requires a resolved predictor_token_id; "
                "JEPARayPPOTrainer.init_workers() should have set this before jepa_init()"
            )

        n_pairs = (data["anchor_input_ids"].shape[0] if "anchor_input_ids" in data
                   else data["cot_input_ids"].shape[0])
        if n_pairs < self.jepa_cfg.min_valid_pairs:
            return TensorDict(
                {"jepa/skipped": torch.tensor(1.0),
                 "jepa/n_valid_pairs": torch.tensor(float(n_pairs))},
                batch_size=[],
            )

        engine = self.actor.engine
        cfg = self.jepa_cfg

        # Bounds peak activation memory during the joint forward+backward to
        # roughly `micro_bs` rows instead of the full joint batch (which can be
        # up to 3 views x train_batch_size rows in triplet mode) — see
        # `_embed_chunked_with_backward`'s docstring for the OOM this fixes.
        micro_bs = max(1, cfg.embed_micro_batch_size)

        # Linear alpha warmup: ramps the EFFECTIVE alpha from 0 -> cfg.alpha
        # over cfg.alpha_warmup_steps JEPA-update calls (disabled when 0, the
        # default — full alpha from the first call, matching prior behavior).
        # `global_step` is set by ray_trainer.py on the input TensorDict;
        # falls back to "no warmup" (full alpha) if absent for any reason.
        if cfg.alpha_warmup_steps > 0:
            default_step = torch.tensor([float(cfg.alpha_warmup_steps)])
            step = float(data.get("global_step", default_step)[0].item())
            alpha = cfg.alpha * min(1.0, step / cfg.alpha_warmup_steps)
        else:
            alpha = cfg.alpha

        with engine.train_mode():
            engine.optimizer_zero_grad()

            # ONE live encoder for all views, predictor tokens on the CoT rows
            # only so p^c = Pred(Enc(correct CoT)). The cot/code/wrong blocks of
            # the input TensorDict are each padded to a common row count with
            # `*_lengths == 0` marking padding rows (see ray_trainer builders) —
            # filter those out before the joint forward so only real rows are
            # encoded. The joint forward + loss differ per loss_type below.
            if cfg.loss_type == "jepa-tcr-dual":
                # Dual-target self-consistent TCR. One combined anchor block (CoT rows
                # first, then Code rows; `is_code` recovers the split), each row with
                # [PRED]. Each view's pred is pulled toward its OWN cached teacher
                # target; each CoT pred is additionally pulled toward the stop-grad
                # Code_S BOUNDARY read of its paired code rollout (self-consistency).
                # SIGReg over both views' preds.
                anchor_ids = data["anchor_input_ids"]
                anchor_mask = data["anchor_attn_mask"]
                anchor_lengths = data["anchor_lengths"]
                is_code = data["is_code"].bool()
                cot_sel = ~is_code
                N = anchor_ids.shape[0]

                # z_self (the Code_S boundary read, stop-grad) rides on the SAME forward
                # that produces the [PRED] reads: `_embed_chunked_with_backward(..,
                # also_boundary=True)` hands the loss_fn the per-row boundary embeddings
                # (last real token, pre-[PRED]) for free — no separate forward. The
                # loss_fn gathers each CoT row's partner (self_partner indexes the code
                # sub-block) from the boundary of the corresponding ABSOLUTE code row.
                joint_predictor_k = [cfg.predictor_k] * N
                teacher_target = data["teacher_target"]
                group_id = data.get("anchor_group_id", None)
                is_correct = data.get("anchor_is_correct", None)
                self_partner = data.get("self_partner", None)
                # Per-arm plateau latches (broadcast scalars from the trainer). When an
                # arm is off, its align term contributes 0 grad but preds stay in SIGReg.
                def _arm_on(key):
                    v = data.get(key, None)
                    return True if v is None else bool(v.reshape(-1)[0].item())
                align_cot_on = _arm_on("align_cot_on")
                align_code_on = _arm_on("align_code_on")
                self_on = _arm_on("self_on")
                # Absolute row indices of the code sub-block (code rows are last, in order).
                code_abs = is_code.nonzero(as_tuple=True)[0]
                cot_rows = cot_sel.nonzero(as_tuple=True)[0]
                n_cot = int(cot_rows.numel())
                n_code = int(code_abs.numel())

                def _loss_fn(joint_emb, boundary, _cot=cot_sel, _code=is_code,
                             _tt=teacher_target, _gid=group_id, _ic=is_correct,
                             _sp=self_partner, _code_abs=code_abs, _cot_rows=cot_rows,
                             _n_cot=n_cot, _n_code=n_code,
                             _cot_on=align_cot_on, _code_on=align_code_on, _self_on=self_on):
                    dev, dt = joint_emb.device, joint_emb.dtype
                    tt = _tt.to(device=dev, dtype=dt)
                    gid = _gid.to(dev) if _gid is not None else None
                    ic = _ic.to(dev) if _ic is not None else None
                    # Build z_self from the boundary reads of the SAME forward. Boundary
                    # is detached; partner index (into the code sub-block) -> absolute row.
                    d = joint_emb.shape[-1]
                    self_target = boundary.new_zeros((_n_cot, d))
                    self_mask = torch.zeros(_n_cot, dtype=torch.bool, device=dev)
                    if _sp is not None:
                        sp_cot = _sp.to(device=dev, dtype=torch.long)[_cot_rows.to(dev)]
                        has = (sp_cot >= 0) & (sp_cot < _n_code)
                        if has.any():
                            rows = torch.arange(_n_cot, device=dev)[has]
                            abs_idx = _code_abs.to(dev)[sp_cot[has]]
                            self_target[rows] = boundary[abs_idx]
                            self_mask[rows] = True
                    return llm_jepa_tcr_dual_loss(
                        pred_cot=joint_emb[_cot],
                        teacher_target_cot=tt[_cot],
                        pred_code=joint_emb[_code],
                        teacher_target_code=tt[_code],
                        self_target=self_target,
                        self_mask=self_mask,
                        cot_group_id=gid[_cot] if gid is not None else None,
                        cot_is_correct=ic[_cot] if ic is not None else None,
                        code_group_id=gid[_code] if gid is not None else None,
                        code_is_correct=ic[_code] if ic is not None else None,
                        self_consist_w=cfg.self_consist_w,
                        align_cot_on=_cot_on,
                        align_code_on=_code_on,
                        self_on=_self_on,
                        lambda_=cfg.triplet_sigreg_lambda,
                        M=cfg.n_projections,
                        t_min=cfg.t_min, t_max=cfg.t_max, s=cfg.epps_pulley_s,
                    )

                jepa_metrics = self._embed_chunked_with_backward(
                    anchor_ids, anchor_mask, anchor_lengths, joint_predictor_k, _loss_fn,
                    micro_bs, alpha, also_boundary=True,
                )
                grad_norm = engine.optimizer_step(clip_grad_override=self.jepa_cfg.max_grad_norm)
                self._sync_ema()
                aggressive_empty_cache(force_sync=True)
                _total_loss_value = jepa_metrics.get("jepa/llm_jepa_loss", 0.0)
                out = {
                    "jepa/total_loss": torch.tensor(float(_total_loss_value)),
                    "jepa/n_valid_pairs": torch.tensor(float(n_pairs)),
                    "jepa/skipped": torch.tensor(0.0),
                    "jepa/grad_norm": torch.tensor(float(grad_norm) if grad_norm is not None else 0.0),
                }
                out.update({k: torch.tensor(float(v)) for k, v in jepa_metrics.items()})
                return TensorDict(out, batch_size=[])
