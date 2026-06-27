# Dr GRPO — Math & Coding Benchmark Report

Base model: **DeepSeek-R1-Distill-Qwen-1.5B**. All numbers are %, 16 samples per prompt
(`val_kwargs.n=16`).

**Method ↔ checkpoint mapping**
| Report name | Checkpoint | Selected at |
|---|---|---|
| Dr GRPO | `drgrpo_fsdp-20260621` | latest (step 240)¹ |
| + CoT+code | `jepa_grpo_ray_nosep_nokl-20260622` | val-best (step 80) |
| + Distilled CoT-JEPA | `tcr_hybrid_b05_a005_correct_util07-20260623` | val-best (step 190) |

¹ Dr GRPO's val-best (step 140) checkpoint weights were lost to the best-ckpt eviction
bug, so its **latest** surviving checkpoint (step 240) is used here.

**Metric convention**
- **Math tables** (reproduced from source): `pass/avg/best` = `maj@k / avg@k / best@k`.
- **Coding tables**: `avg@8 / pass@8` (avg@8 = mean accuracy over 8 samples; pass@8 = best-of-8).

Bold = better of the two methods in that table for the metric.

---

## Table 1 — Dr GRPO vs CoT+code

### Math
| Method | AIME25<br>(pass/avg/best) | AIME26<br>(pass/avg/best) | AMC23<br>(pass/avg/best) | Overall<br>(pass/avg/best) |
|---|---|---|---|---|
| Dr GRPO | 26.7 / 10.4 / **33.3** | **16.7** / 7.1 / 23.3 | 75.0 **/ 56.2 /** 90.0 | 39.4 / 24.6 / 48.9 |
| + CoT+code | **30.0 / 13.3 / 33.3** | **16.7 / 10.4 / 26.7** | **77.5** / 55.6 / 87.5 | **41.4 / 26.5 / 49.2** |

### Coding (pass@8, %)
| Method | HumanEval+ | MBPP+ | LiveCodeBench | Overall |
|---|---|---|---|---|
| Dr GRPO | 41.5 | 37.8 | 17.5 | 32.3 |
| + CoT+code | **47.0** | **41.3** | **18.2** | **35.5** |

---

## Table 2 — Dr GRPO vs Distilled CoT–JEPA

### Math
| Method | AIME25<br>(pass/avg/best) | AIME26<br>(pass/avg/best) | AMC23<br>(pass/avg/best) | Overall<br>(pass/avg/best) |
|---|---|---|---|---|
| Dr GRPO | 26.7 / 10.4 / 33.3 | 16.7 / 7.1 / 23.3 | 75.0 / 56.2 / **90.0** | 39.4 / 24.6 / 48.9 |
| + Distilled CoT–JEPA | **36.7 / 11.7 / 36.7** | **20.0 / 11.2 / 30.0** | **82.5 / 56.6** / 85.0 | **46.4 / 26.5 / 50.6** |

### Coding
| Method | HumanEval+<br>(avg@8/pass@8) | MBPP+<br>(avg@8/pass@8) | LiveCodeBench<br>(avg@8/pass@8) | Overall<br>(avg@8/pass@8) |
|---|---|---|---|---|
| Dr GRPO | **16.9** / 41.5 | 15.4 / 37.8 | **5.2** / 17.5 | **12.5** / 32.3 |
| + Distilled CoT–JEPA | 15.6 / **42.1** | **15.9 / 39.4** | 5.1 / **17.8** | 12.2 / **33.1** |

---

## Notes & takeaways

- **Coding prompts:** HumanEval+ = 164, MBPP+ = 378, LiveCodeBench = 269 (after filtering),
  each ×16 samples. Scored via code execution.
- **pass@8 (best-of-8) is where both additions help.** On the Overall coding column,
  CoT+code lifts pass@8 from 32.3 → **35.5** and Distilled CoT–JEPA to **33.1**, vs Dr GRPO.
- **avg@8 is essentially flat** across methods (~12.2–12.9 overall) — the gains are in
  best-of-k coverage (exploration/diversity), not single-attempt accuracy. This mirrors the
  math results, where `best` moves more than `avg`.
- **Per-benchmark leaders (pass@8):** HumanEval+ → CoT+code (47.0); MBPP+ → CoT+code (41.3);
  LiveCodeBench → CoT+code (18.2).
- **Caveat:** Dr GRPO here is the *latest* (step 240) checkpoint, not a val-best, so the
  comparison is not strictly best-for-best on the Dr GRPO side.

Source data: `code_benchmark_evals.csv` (9 rows, full avg/pass@{1,4,8,16} + response length,
repetition rate). Math tables reproduced from the provided source figure.
