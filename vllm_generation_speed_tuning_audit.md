# JEPA-GRPO — vLLM Generation-Speed & GPU-Utilization Tuning Audit

> Analysis only, no code changed. Single-GPU (94.97 GiB) HYBRID colocated FSDP-actor + vLLM,
> Qwen2.5-Math-1.5B, LoRA r512 all-linear (no merge, `lora_as_adapter=True`), no KL/ref model,
> CPU verifier, precomputed CPU TCR targets. Evidence drawn from real run logs
> (`/tmp/jepa_nowarmup_run.log`, `/tmp/jepa_*run.log`) and the verl rollout config.

## Measured per-step phase timings (clreg run, n_cot=4 + n_code=4, vLLM awake util≈0.5)

| Phase | line | vLLM state | time (s) | notes |
|---|---|---|---|---|
| `cot_gen` (CoT **and** Code generation) | 763 | **awake** | **~56–63** | **largest phase** |
| `old_log_prob` | 868 | **awake** | ~13–15 | logprob recompute, **vLLM still awake** |
| `sleep_replicas_1` | 876 | →asleep | ~0.85 | sleep level 1 |
| `update_actor` (GRPO backward) | 880 | asleep | ~39–43 | KV freed, full card free |
| `jepa_build_batch` | 900 | asleep | ~0.03 | trivial |
| `jepa_update` (`[PRED]` fwd+bwd) | 925 | asleep | ~11–14 | |
| `weight_sync_2` (= wake) | 963 | →awake | ~2.4–3.9 | resume weights+KV |

Compute-step ≈ **130 s**, of which **generation ≈ 46 %** and the backward/JEPA ≈ 42 %.
(`step_total` ~6000 s in the log includes the periodic AIME/AMC validation pass; ignore it.)

> For the **TCR run (n_code=0)** only `cot_gen` (CoT) runs — no code view — so generation
> volume and KV pressure are **lower**, giving more headroom than the clreg numbers above.

## Critical evidence (two OOMs in the logs)

1. `Tried to allocate 6.90 GiB … 1.13 GiB free … 92.14 GiB in use` — near-full card, the
   ~6.9 GiB **lm_head logits block**. This is the `old_log_prob`/actor logits cost the run
   script already fights (`PPO_MAX_TOKEN_LEN` cut 32768→24576 "to shrink that block to ~6.9 GB").
2. `Tried to allocate 66.05 GiB … 53.43 GiB free` — a vLLM profiling/wake-time block.

3. At **util 0.70** (a sibling HTTP-server run) vLLM reported: `Available KV cache memory:
   60.92 GiB`, `GPU KV cache size: 2,281,232 tokens`, `Maximum concurrency … 69.62x`, and a
   CUDA-graph profiling note (0.70 ≈ effective 0.6931). → KV capacity is **enormous** for a
   1.5B model; KV is **not** the limiter.

---

## 1. High-level recommendation

- **Increase `gpu_memory_utilization`? → YES, modestly.**
- **Recommended first value: 0.62** (TCR/CoT-only run can try **0.65** directly).
- **Maximum to test carefully: 0.70.** Above that the risk is not rollout but `old_log_prob`.
- **Main risk:** OOM during **`old_log_prob`** (and the lm_head logits block), because that
  phase runs **while vLLM is still awake** (line 868 is *before* `sleep_replicas_1` at 876), so
  vLLM's reserved fraction is subtracted from the card exactly when the actor needs its ~7 GiB
  logits + activations. Rollout generation itself has huge KV headroom and will not OOM first.

## 2. Current bottleneck

- **Bottleneck phase:** **rollout generation (`cot_gen`, ~60 s, ~46 %)**, secondarily the
  actor backward (~40 s).
- **Evidence:** phase timers above; generation is the single largest slice; KV at 0.70 holds
  2.28 M tokens / 69× concurrency, so generation is **partly KV/concurrency-limited at 0.50**
  (the full 512-seq batch = ~2.1 M token-slots barely fits only at ~0.65–0.70).
- **Missing evidence:** no per-phase **GPU-memory** logging and no **tokens/sec** or
  `num_preempted` value surfaced in the training log (the key exists but prints no number).
  Add these (see §12) to confirm whether raising util removes preemption waves.

## 3. Memory phase table (1.5B, ~95 GiB card)

| Phase | vLLM EngineCore | FSDP actor proc | total (approx) | effect of ↑ gpu_memory_utilization |
|---|---|---|---|---|
| Before wake / asleep | ~0 (weights on CPU, KV freed) | ~12–20 GiB resident+reserved | ~12–20 GiB | none (vLLM asleep) |
| After resume **weights** | +~3 GiB | same | ~15–23 GiB | small (weights fixed) |
| After resume **KV** (awake) | + KV pool (**util×95 − weights − graphs**) | same | rises to util-bound | **directly scales KV pool** |
| **`old_log_prob`** (awake) | full vLLM reservation | + ~7 GiB logits + activations | **highest combined** | **↑ util → ↑ OOM risk here** |
| Generation | full vLLM reservation | idle-ish | high but stable | ↑ util → more concurrency, faster |
| `sleep_replicas_1` → backward | ~0 | + grads + activations (~peak actor) | actor-bound only | **none** (vLLM asleep) |
| `jepa_update` | ~0 | actor + `[PRED]` fwd | actor-bound | none |
| `weight_sync_2` (wake) | re-reserves | actor present | transient spike | ↑ util → bigger spike |

**When ↑util consumes memory:** only while vLLM is **awake** — i.e. `weight_sync_2` →
`cot_gen` → `old_log_prob`. It is fully released at `sleep_replicas_1`, so `update_actor` and
`jepa_update` are unaffected by util.

## 3b. Safe-upper-bound estimate (which phase OOMs first)

| util | KV tokens (~) | wake | generation | **old_log_prob (awake)** | backward (asleep) | JEPA (asleep) | verdict |
|---|---|---|---|---|---|---|---|
| 0.55 | ~1.8 M | safe | safe | safe | safe | safe | current-ish, fine |
| 0.60 | ~2.0 M | safe | safe | low risk | safe | safe | **recommended** |
| 0.65 | ~2.1 M | safe | safe (batch ~fits) | **watch** | safe | safe | good for TCR |
| 0.70 | ~2.28 M | watch | safe | **moderate OOM risk** | safe | safe | ceiling to test |
| 0.75 | ~2.45 M | risk | safe | **high OOM risk** | safe | safe | not advised |
| 0.80 | ~2.6 M | high risk | safe | **likely OOM** | safe | safe | unsafe |

**First phase to OOM as util rises: `old_log_prob`** (vLLM awake + lm_head logits block), not
rollout, not backward.

## 4. Does higher util actually increase speed?

- It raises **KV-cache capacity → max concurrent sequences → rollout batch size → tokens/sec**,
  **until the whole rollout batch fits** (≈util 0.65–0.70 here). Beyond that it only buys
  unused KV and **does not** speed anything.
- Current rollout at 0.50 is **partly KV/concurrency-limited** (batch ≈2.1 M slots vs ~1.6 M
  KV) → expect a real speedup going to ~0.62–0.65. It is **not** CPU- or scheduler-limited
  (verifier is async/CPU, off the gen path). Sleep/wake overhead is tiny (~3 s/step).
- **Clear statement:** util **is** a (partial) lever for generation speed here, but with
  **diminishing returns past ~0.65–0.70**; the bigger structural cost is that generation is
  inherently ~46 % of the step.

## 5. Offloading interaction

- `optimizer_offload=True`: Adam state lives **only on LoRA params** (base frozen) → small
  (~hundreds of MB to ~1 GiB). Offloading frees little and adds PCIe traffic each step. **Low
  value here.**
- `param_offload=True`: pushes the **base 1.5B (~3 GiB)** to CPU between steps. Frees ~3 GiB
  but base must be re-loaded for every forward (gen logprob, backward, JEPA) → real speed
  penalty. **Not worth it** while the actor is the active model every phase.
- Offload only helps the **actor-update memory** phase — which is *already* safe (vLLM asleep).
  It does **nothing** for vLLM wake/generation (different process). 
- **Recommendation: enable neither.** The memory pressure is in `old_log_prob` (vLLM awake),
  which offload does not touch.

## 6. CUDA graphs vs `enforce_eager`

- Logs show graph capture/compile is a **one-time ~12 s at startup**; `weight_sync_2` wake is
  only ~2.4–3.9 s → **graphs are NOT re-captured per wake** (only KV+weights resume). So eager
  mode would **not** speed up wakes.
- `enforce_eager=True` would save the CUDA-graph memory slice (~0.7 % of util per the log) and
  startup compile, but **reduce decode throughput** (graphs help the many-small-step decode of
  3072-token generations). Since generation is the bottleneck, **keep CUDA graphs** (eager is a
  net loss here). Optionally test eager once to quantify, but do not adopt by default.

## 7. vLLM settings — current vs recommended

| setting | current | recommend |
|---|---|---|
| `gpu_memory_utilization` | 0.50–0.55 | **0.62** (TCR: 0.65), ceiling 0.70 |
| `max_num_seqs` | 1024 | keep (batch ≤512 seqs; not binding) |
| `max_num_batched_tokens` | 8192 (base default) | **raise to 16384** if prefill-bound; verify in log |
| `max_model_len` | null → prompt+resp = 1024+3072 = **4096** | keep 4096 (matches workload) |
| `enable_chunked_prefill` | True | keep |
| `enable_prefix_caching` | True | keep (math prompts share little, but harmless) |
| `enforce_eager` | False (graphs on) | **keep False** |
| `free_cache_engine` | True | keep (required for sleep/wake) |
| TP / PP | TP=1, no PP | keep (single GPU, 1.5B) |
| LoRA sync | adapter-only, level-1 sleep | keep (efficient) |
| sleep level | 1 (lora_as_adapter) | keep |

## 8. Rollout workload

- `rollout.n=8`; for **TCR** `n_cot=8, n_code=0` (CoT only); for **clreg** `n_cot=4, n_code=4`
  (CoT **and** Code → 2 views, ~2× gen).
- prompt ≤ **1024**, response ≤ **3072**, `max_model_len ≈ 4096`. train_batch 64 prompts ×8 =
  **512 sequences/step**.
- Generation `generate_sequences` mean ~9–10 s but **max ~25–29 s** → long-tail 3072-token
  sequences dominate wall time (KV held longest by the slowest sequences).
- **Lowering `max_response_length`** (e.g. 3072→2048) would cut generation time **more
  reliably than raising util** (less long-tail KV occupancy + fewer decode steps), at the cost
  of truncating some long CoTs — worth an ablation.
- Increasing concurrency only helps until the batch fits (~util 0.65); it's already at 512 seqs.

## 9. Weight-sync overhead

- `weight_sync_2` ≈ **2.4–3.9 s/step** — cheap. Since LoRA is an adapter, only **adapter
  weights** sync (base stays resident on CPU and is restored by sleep-level-1 wake), so sync is
  small. **Not a bottleneck.**
- Base is loaded/offloaded correctly (sleep level 1 keeps base on CPU; no full reload).
- One sleep + one wake per step already (a second sleep was deliberately removed — sleeping an
  asleep engine segfaults, see `ray_trainer.py:946-961`). **No redundant cycles to remove.**

## 10. Memory-saving opportunities

| idea | worthwhile? | why |
|---|---|---|
| Disable JEPA **EMA clone** when `loss_type != lejepa` | **Yes (P2)** | EMA of LoRA adapters is built+synced every step but unused in TCR/CLReg — frees a little GPU + per-step compute |
| `optimizer_offload` | No | tiny (LoRA-only Adam); adds PCIe cost |
| `param_offload` | No | base needed every phase; real slowdown |
| `enforce_eager` to cut graph memory | No | hurts decode (the bottleneck) |
| Lower `max_model_len` | No | already matches 4096 workload |
| Lower `max_num_batched_tokens` | No | would slow prefill |
| Lower `gpu_memory_utilization` | No (raise instead) | currently slightly KV-limited |
| **Lower `max_response_length`** 3072→2048 | **Maybe (test)** | biggest single gen-time lever |
| `empty_cache` placement | Already done | `aggressive_empty_cache` after sync + JEPA |
| Avoid model clones | Already minimal | only the small EMA adapter clone |
| **Move `old_log_prob` after sleep** (code change) | **Yes (P1, needs approval)** | would remove the only util-vs-OOM conflict — see §Bugs |

## 11. Proposed tuning experiments (staged)

All edits are env vars on `run_qwen_1_5b_ray.sh` unless noted.

| Exp | change | expected speed | expected memory | OOM risk | compare | success |
|---|---|---|---|---|---|---|
| **Baseline** | current (util 0.55) | ref | ref | low | `cot_gen`, `step` timers | stable run |
| **A** | `ROLLOUT_GPU_MEM_UTIL=0.60` | +5–12 % gen | +~5 GiB awake | low | `cot_gen` ↓, no OOM | gen faster, no OOM in `old_log_prob` |
| **B** | `0.65` | +8–18 % gen | +~9 GiB awake | low–med | `cot_gen` ↓ plateauing | gen faster, `old_log_prob` survives |
| **C** | `0.70` | ≤ B (plateau) | +~13 GiB awake | **med** | watch `old_log_prob` mem | no OOM; if gen≈B, stop at B |
| **D** | `enforce_eager=True` (+ util best) | gen slower | −graph mem, −12 s startup | low | `cot_gen` vs C | only adopt if gen unchanged |
| **E** | `actor.optimizer_offload=True` | slightly slower | −<1 GiB | low | `update_actor` time/mem | only if memory-bound (won't be) |
| **F** | `actor.param_offload=True` | slower (reload) | −~3 GiB | low | per-phase times | likely reject |
| **G** | best of A–C + `MAX_RESPONSE_LENGTH=2048` ablation | **largest gen win** | lower KV | low | `cot_gen`, eval acc | gen ↓ a lot, acc not hurt |

Recommended order: **A → B → (C only if B still preempts) → G**. Skip D/E/F unless a phase OOMs.

## 12. Commands / logs to run

```bash
# Process-level GPU memory (run during a LIVE training step; shows the two PIDs):
watch -n1 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# Tail the training log and watch the phase the OOM lives in:
tail -f /tmp/<run>.log | grep -E "old_log_prob|cot_gen|update_actor|jepa_update|weight_sync|OutOfMemory"

# Per-phase timers (already logged):
grep -oE "timing_s/(cot_gen|old_log_prob|sleep_replicas_1|update_actor|jepa_update|weight_sync_2):[0-9.]+" /tmp/<run>.log | tail -40

# vLLM KV cache size / concurrency / cudagraph note at startup:
grep -E "Available KV cache memory|GPU KV cache size|Maximum concurrency|CUDA graph memory profiling" /tmp/<run>.log

# Did any phase preempt (KV pressure)? add a value print, or check:
grep -oE "num_preempted/(max|mean):[0-9.]+" /tmp/<run>.log | tail

# Is offload actually active?
grep -iE "param_offload|optimizer_offload|is_param_offload_enabled|to\\('cpu'\\)" /tmp/<run>.log

# PyTorch allocated vs reserved at wake (already emitted by engine_workers):
grep -nE "Before resume weights|After resume weights|After update_weights|After resume kv_cache" /tmp/<run>.log
```

**Suggested log additions (no behavior change):** wrap `old_log_prob` and `cot_gen` with a
`log_gpu_memory_usage(...)` before/after (uses `torch.cuda.memory_allocated/reserved`), and
surface vLLM `num_preempted` as a scalar metric, so the util sweep is decided by data not guesswork.

---

## 8. Bugs / inefficiencies

| Prio | File / function | Issue | Proposed fix (not yet applied) |
|---|---|---|---|
| **P1** | `ray_trainer.py` — `old_log_prob` (868) runs **before** `sleep_replicas_1` (876) | The logprob recompute (≈7 GiB lm_head block) executes while vLLM still holds its full KV reservation → this is the phase that OOMs and the reason util can't go high. | Move the `sleep_replicas_1` call **before** `old_log_prob` (sleep vLLM right after generation). Frees the KV pool for the logprob block, decoupling util from the OOM and letting util go to ~0.75. Verify vLLM isn't needed between gen and sleep. |
| **P2** | `worker.py:67,173` — EMA clone of LoRA adapters | Maintained every step but only consumed in `lejepa` mode (unused in TCR/CLReg). | Gate EMA build+`_sync_ema` on `loss_type == "lejepa"`. |
| **P2** | rollout `max_num_batched_tokens=8192` (base default) vs `max_model_len=4096`, chunked prefill on | Prefill of 64×1024-token prompts may be throttled at 8192 batched tokens. | Test `max_num_batched_tokens=16384`; measure prefill share of `cot_gen`. |
| **P3** | No per-phase GPU-mem / tokens-per-sec / preemption metric | Tuning is currently blind to allocated-vs-reserved and preemption. | Add `log_gpu_memory_usage` around gen/logprob and a `num_preempted` scalar. |

## 9. Final answer

- **Best likely config for SPEED:** `gpu_memory_utilization=0.65` (CoT-only TCR) + keep CUDA
  graphs + `max_num_batched_tokens=16384`, and (P1) sleep vLLM before `old_log_prob` so 0.70 is
  safe; plus the `MAX_RESPONSE_LENGTH=2048` ablation (biggest single gen win).
- **Best likely config for SAFETY:** `gpu_memory_utilization=0.60`, everything else as-is, no
  offload, CUDA graphs on. Stable, modest speedup, no `old_log_prob` OOM.
- **Monitor on the first run:** `timing_s/cot_gen` (should drop), `timing_s/old_log_prob` and a
  `nvidia-smi` snapshot **during** `old_log_prob` (must stay < card), any `OutOfMemory`,
  `num_preempted`, and eval accuracy (if you also cut response length). If `old_log_prob`
  memory approaches the card, stop raising util (or apply the P1 sleep-reorder first).

> No code changed. Awaiting approval before applying any P1/P2 fix or config edit.
