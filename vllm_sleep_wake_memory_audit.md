# JEPA-GRPO — vLLM Sleep/Wake GPU Memory Audit

> Analysis only. No code changed. Scope: the **`main_ray` / `ray_trainer.py`** path
> (the one the live training run uses), which is a **single-GPU, HYBRID, colocated
> FSDP-actor + vLLM** setup with LoRA and **no KL / no reference model**.

Relevant config (from `examples/jepa_grpo_trainer/run_qwen_1_5b_ray.sh`,
`config/jepa_grpo_ray_trainer.yaml`, `config/jepa_grpo_ray_qwen25_1_5b.yaml`):
- `trainer.n_gpus_per_node=1`, model = Qwen2.5-Math-1.5B-Instruct (bf16 ≈ 3 GB).
- LoRA rank 512, `target_modules=all-linear`, **no merge** → `lora_as_adapter = True`.
- `rollout.gpu_memory_utilization ≈ 0.50–0.55`, `rollout.free_cache_engine: true`.
- `actor.use_kl_loss=false`, `algorithm.use_kl_in_reward=false` → **reference model NOT built**.
- `actor.param_offload` / `optimizer_offload` unset → **default False** (actor stays on GPU).
- Verifier reward = CPU function (`verl.experimental.fepo.math_parser.compute_math_reward`) → **no reward model on GPU**.
- Teacher/JEPA targets = precomputed offline, loaded `map_location="cpu"` (`ray_trainer.py:69`) → **no teacher encoder on GPU during training**.

---

## 1. High-level answer

- **Why memory rises when vLLM wakes:** `update_weights()` (the "weight sync", which *is*
  the wake in HYBRID mode) calls `rollout.resume(tags=["weights"])` then
  `resume(tags=["kv_cache"])` → vLLM's **EngineCore process** re-materializes (a) the model
  weights it offloaded to CPU during sleep, (b) the **KV-cache pool** sized to
  `gpu_memory_utilization`, and (c) (default `enforce_eager=False`) **CUDA graphs**. The KV
  pool is the big one — it is *reserved up front* to the configured fraction, so it shows as
  a large jump even before any tokens are generated.
- **What remains after vLLM sleeps:** the **colocated FSDP training actor** (a *separate*
  process from vLLM's EngineCore): base 1.5B weights + LoRA adapters + Adam optimizer state +
  grads + activation/cuDNN workspace + PyTorch caching-allocator *reserved* blocks, plus the
  small JEPA **EMA clone of the LoRA adapters**. vLLM's own process drops to ~a bare CUDA
  context (sleep level 1 moves its weights to CPU and frees KV).
- **Is this expected:** **Yes.** This is the intended hybrid-engine behavior: the trainer
  model is persistent (it must run backward/logprob/JEPA), and only the rollout engine
  sleeps/wakes around the actor update to hand the GPU back for the backward pass.

---

## 2. Current model/process map (single GPU, this run)

| Process / model | GPU | Purpose | Approx memory | Stay alive after vLLM sleep? |
|---|---|---|---|---|
| **FSDP training actor** (base 1.5B + LoRA + Adam + grads), worker process | GPU0 | GRPO backward, logprob recompute, JEPA `[PRED]` embedding extraction | base ≈3 GB + LoRA/optim/grad + reserved cache → **this is the persistent "~20 GB"** | **Yes** — it is the trainer; never sleeps |
| **JEPA EMA clone** (LoRA adapter params only, `worker.py:67`) | GPU0 | EMA target encoder (used by `lejepa` mode; maintained but **not on the loss path** for `tcr`/`clreg`) | small (hundreds of MB) | Yes (lives with actor) |
| **vLLM rollout EngineCore** (own process; `VLLM::EngineCore` in `nvidia-smi`) | GPU0 | CoT + Code rollouts | weights ≈3 GB + KV pool (≈0.5×GPU) + CUDA graphs **when awake**; ~0 when asleep (lvl 1) | **No** — sleeps between generation and weight-sync |
| Reference model | — | KL anchor | **not loaded** (KL disabled) | n/a |
| Reward / verifier model | — | correctness | **not loaded** (CPU function) | n/a |
| Teacher / target encoder | — | TCR targets | **not on GPU** (precomputed offline, CPU cache) | n/a |

---

## 3. Sleep/wake behavior

**Call sites**
- Sleep: `ray_trainer.py:876-877` `simple_timer("sleep_replicas_1") → checkpoint_manager.sleep_replicas()` (before `update_actor`, so the GPU is free for backward).
- Wake (folded into weight sync): `ray_trainer.py:963-964` `simple_timer("weight_sync_2") → checkpoint_manager.update_weights(global_steps)`; also at init `ray_trainer.py:730`.
- `sleep_replicas` → `checkpoint_engine/base.py:410` → `replica.sleep()` → `vllm_async_server.py:672 sleep()` → HYBRID → `_sleep_hybrid()` (`:970`).
- `update_weights` (naive) → `engine_workers.py:752` → `rollout.resume(tags=["weights"])` (`:792`) … `rollout.resume(tags=["kv_cache"])` (`:827`) → `vllm_rollout.py:150 resume()` → `wake_up(tags=...)`.
- The HTTP-server variant (`trainer.py`, `/sleep` `/wake`, `vllm_sleep_level`) is a *different* code path, not used by `main_ray`.

**What WAKE does** (`engine_workers.update_weights`, naive mode):
- `set_expandable_segments(False)`; logs `Before resume weights`.
- `resume(tags=["weights"])` → vLLM copies model weights **back from CPU** into GPU.
- Syncs the **LoRA adapter** weights trainer→vLLM (`lora_as_adapter` path; base already present).
- `resume(tags=["kv_cache"])` → reallocates the **KV-cache pool** to `gpu_memory_utilization`.
- On first wake also captures **CUDA graphs** (default `enforce_eager=False`).
- `reset_prefix_cache()` after wake. KV is reserved up front, not lazily.

**What SLEEP does** (`_sleep_hybrid`, `:970`):
- `lora_as_adapter=True` here ⇒ `engine.collective_rpc("sleep", level=1)`.
- **Level 1** = offload vLLM **weights to CPU RAM** + **free KV cache** (and `reset_encoder_cache()` on vLLM ≥0.17). Weights are *kept* (on host), so re-wake is a host→device copy, not a reload.
- (If LoRA were *merged*, `lora_as_adapter` would be False ⇒ **level 2** = discard weights entirely; re-wake needs a full weight re-sync.)

**What SLEEP does NOT do**
- Does **not** touch the FSDP **training actor** (separate process) — base weights, LoRA,
  optimizer, grads all stay on GPU (no `param_offload`).
- Does **not** shrink the **trainer process's** PyTorch caching-allocator reserved pool.
- Level 1 does **not** free vLLM weights from *host* RAM (keeps the CPU copy).
- Does not by itself run `torch.cuda.empty_cache()` for the trainer; that is done separately
  via `aggressive_empty_cache(force_sync=True)` in `engine_workers.update_weights:823` and in
  `worker.py:615,746` after the JEPA step.

---

## 4. Explanation of the ~20 GB "model" memory

- **Most likely source:** the **persistent colocated FSDP training-actor process** —
  base 1.5B weights (~3 GB) + LoRA adapters + **Adam optimizer state** (on the LoRA params) +
  gradients + activation/cuDNN workspace + **PyTorch caching-allocator reserved blocks**
  (which `nvidia-smi` reports as "used" even when not currently allocated). This is the part
  that stays after vLLM sleeps. If the ~20 GB is observed *while vLLM is awake*, then on top
  of it sits vLLM's rollout reservation (weights + KV pool ≈ `gpu_memory_utilization` × VRAM +
  CUDA graphs) in the **separate EngineCore process**.
- **Evidence:**
  - vLLM runs as its own process — `nvidia-smi` currently shows `VLLM::EngineCore … 86562 MiB`
    (that is the **precompute** 7B at 0.85, not training — illustrates only that vLLM owns its
    memory in a distinct PID).
  - `use_kl_loss=false` / `use_kl_in_reward=false` ⇒ no ref model; reward is a CPU function;
    teacher targets load to CPU ⇒ none of these contribute GPU memory.
  - `param_offload`/`optimizer_offload` default False ⇒ the actor + optimizer never leave GPU.
  - `is_param_offload_enabled` guards the only actor→CPU offload (`engine_workers.py:821`), and
    it is disabled here ⇒ the actor is the resident block that "stays active."
- **Useful or waste:** **mostly useful** — the actor must stay resident for the GRPO backward,
  logprob recompute, and JEPA `[PRED]` extraction. The *reserved-but-unused* slice of the
  caching allocator is not true waste (it is reused next step), but it can make `nvidia-smi`
  overstate "active" memory vs. `torch.cuda.memory_allocated()`.

---

## 5. Bugs / inefficiencies found

| Prio | Where | Finding | Why it matters |
|---|---|---|---|
| **P2** | `worker.py:67` (`jepa_init`) + `_sync_ema` (`:173`) | The JEPA **EMA clone of the LoRA adapters is maintained every step** (`_sync_ema` at `:614,736`) but is **only consumed on the loss path in `lejepa` mode**. For the active `tcr`/`clreg` modes the EMA is never read, so it is pure GPU resident + per-step compute. | Small wasted GPU (adapter-sized) and a small per-step EMA update for nothing in TCR/CLReg runs. Candidate to skip when `loss_type != lejepa`. |
| **P2** | rollout config (no `enforce_eager`) | `enforce_eager` is unset ⇒ vLLM captures **CUDA graphs** on each wake. | Adds wake latency and extra GPU on every wake/sleep cycle; with frequent sleep/wake this is recurrent overhead. Setting `enforce_eager=true` trades a little decode speed for faster, lighter wakes. |
| **P2** | `gpu_memory_utilization ≈ 0.5` on a ~96 GB card for a 1.5B model | The KV pool is reserved to ~48 GB regardless of need. | Most of the reserved vLLM memory is never used by a 1.5B model; it is fine functionally but means "vLLM awake" looks huge in `nvidia-smi`. Could lower util to free headroom for the JEPA forward, or raise it intentionally for throughput. |
| **P3 (not a bug)** | `ray_trainer.py:946-961` | A second sleep was deliberately **removed** because sleeping an already-asleep engine "reproducibly segfaults" (`EngineDeadError`). | Documented invariant: there is exactly one sleep (`sleep_replicas_1`) and one wake (`weight_sync_2`) per step. Re-adding a sleep would crash. Confirmed correct as-is. |
| **P3** | `release_kv_cache` / `resume_kv_cache` (`vllm_async_server.py:687,694`) | These are **no-ops** ("TODO: support true release of kv_cache"). | In the naive/hybrid path the KV is freed only via full sleep; the "release KV only" optimization in `update_weights` step 3/7 does nothing here. Not incorrect, just not yet implemented. |

No evidence of: duplicate model load, ref/reward/teacher model stuck on GPU, rollout failing to unload, or cache not emptied where the code intends to.

---

## 6. Recommended experiments / fixes (not yet applied)

**Measure the three states directly (do this on the real training run, not the precompute):**
```bash
# While the jepa training run is live, in a loop:
nvidia-smi --query-gpu=memory.used,memory.reserved,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
# -> you should see TWO pids: the FSDP worker (python, the "~20GB") and VLLM::EngineCore.
```
- The repo already logs `Before/After resume weights`, `After update_weights`, `After resume
  kv_cache` via `log_gpu_memory_usage` (`engine_workers.py:788-828`). **Grep the training log**
  for these to get allocated-vs-reserved at wake:
  ```bash
  grep -nE "resume weights|update_weights|resume kv_cache|GPU memory" /tmp/<training>.log
  ```
- Add (if you want a clean before/during/after) a `log_gpu_memory_usage("after sleep_replicas_1")`
  right after `ray_trainer.py:877` and one before/after `weight_sync_2` — compare
  `memory_allocated` (true) vs `nvidia-smi` "used" (reserved) to confirm the ~20 GB is the
  persistent actor, not leaked vLLM.

**Snapshots to compare:** `torch.cuda.memory_allocated()` vs `torch.cuda.memory_reserved()`
in the trainer process at: (a) just after `sleep_replicas_1`, (b) mid `update_actor` backward,
(c) just after `weight_sync_2`. If (a) allocated ≈ (c) allocated but `nvidia-smi` differs, the
delta is reserved cache, not a leak.

**Offload candidates (only if you need GPU headroom):**
- **EMA:** gate `_sync_ema`/EMA clone on `loss_type == "lejepa"` to drop a small always-on cost in TCR/CLReg.
- **Optimizer offload:** `actor.optimizer_offload=true` would push Adam state to CPU between
  steps (frees GPU, costs PCIe traffic) — only worth it if the backward is memory-bound.
- **vLLM:** lower `gpu_memory_utilization` and/or set `enforce_eager=true` to shrink and speed up the awake footprint.

**Nothing should be deleted.** The persistent ~20 GB is the training actor and is required;
the only safe trims are EMA-in-TCR, CUDA-graph capture, and the KV reservation fraction.
