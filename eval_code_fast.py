"""FAST variant of eval_code_benchmarks.py.

Difference from the canonical script: instead of 3 separate generate() calls with
fixed seeds 1234/2025/7777 (the paper protocol), this does ONE generate() with
n=24 and splits the 24 samples per question into 3 consecutive groups of 8.

WARNING: the 3 groups are partitions of a single sampling run, NOT independent
fixed seeds. The reported std is across these partitions and UNDERSTATES true
seed-to-seed variance. Do not use this number as the canonical paper result --
use eval_code_benchmarks.py (fixed seeds) for that.
"""
import argparse, json, os, re, subprocess, sys
import numpy as np, pandas as pd
from math import comb
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

EVAL_DIR = "/workspace/jepa-grpo-cache/eval_data"
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_judge_worker.py")
BENCHMARKS = ["humanevalplus", "mbppplus", "livecodebench"]
GROUPS = 3          # number of pseudo-seed partitions
N = 8               # samples per group -> total n = GROUPS * N
MAXTOK = 3072
MAXLEN = 4096
MAX_NUM_SEQS = 1024
JOB_TIMEOUT = 30
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def _extract_code(text):
    m = _FENCE.findall(text)
    return m[-1] if m else text

def make_job(ds, text, gt):
    spec = json.loads(gt) if isinstance(gt, str) else dict(gt)
    completion = _extract_code(text)
    if ds in ("humanevalplus", "mbppplus"):
        ep = spec["entry_point"]
        code = completion if f"def {ep}" in completion else f"{spec.get('prompt','')}\n{completion}"
        return {"kind": "assert", "code": code, "test": spec["test"], "entry_point": ep}
    else:
        return {"kind": "stdio", "code": completion,
                "inputs": list(spec["inputs"]), "outputs": list(spec["outputs"])}

def score_one(ds, text, gt, ei):
    try:
        job = make_job(ds, text, gt)
    except Exception:
        return 0
    try:
        r = subprocess.run([sys.executable, WORKER], input=json.dumps(job),
                           capture_output=True, text=True, timeout=JOB_TIMEOUT)
        return 1 if r.returncode == 0 else 0
    except Exception:
        return 0

def main():
    global BENCHMARKS, GROUPS, N
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="code_eval_fast.json")
    ap.add_argument("--benchmarks", default=None)
    ap.add_argument("--groups", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-num-seqs", type=int, default=MAX_NUM_SEQS)
    args = ap.parse_args()
    if args.benchmarks: BENCHMARKS = [b.strip() for b in args.benchmarks.split(",")]
    if args.groups: GROUPS = args.groups
    if args.n: N = args.n
    TOTAL = GROUPS * N

    tok = AutoTokenizer.from_pretrained(args.model)
    flat_prompts, prompt_lens, meta, spans = [], [], [], {}
    for b in BENCHMARKS:
        df = pd.read_parquet(f"{EVAL_DIR}/{b}.parquet")
        start = len(flat_prompts)
        for _, row in df.iterrows():
            text = tok.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
            flat_prompts.append(text)
            prompt_lens.append(len(tok(text)["input_ids"]))
            ei = row["extra_info"]
            ei = dict(ei) if hasattr(ei, "keys") else ei
            meta.append((row["data_source"], row["reward_model"]["ground_truth"], ei))
        spans[b] = (start, len(flat_prompts))
    per_max = [max(256, min(MAXTOK, MAXLEN - pl - 8)) for pl in prompt_lens]
    print(f"loaded {len(flat_prompts)} prompts; single generate n={TOTAL} "
          f"split into {GROUPS} groups of {N} (pseudo-seeds, not fixed seeds)")

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.95,
              max_model_len=MAXLEN, max_num_seqs=args.max_num_seqs)
    pool = ThreadPoolExecutor(max_workers=(os.cpu_count() or 16))

    sps = [SamplingParams(n=TOTAL, temperature=0.6, top_p=0.95, max_tokens=m, seed=args.seed)
           for m in per_max]
    outs = llm.generate(flat_prompts, sps)
    # Flatten (sample, meta) and score in parallel; TOTAL consecutive flags per question.
    jobs = []
    for o, (ds, gt, ei) in zip(outs, meta):
        for s in o.outputs:
            jobs.append((ds, s.text, gt, ei))
    flags = list(pool.map(lambda a: score_one(*a), jobs))
    flat = np.array(flags).reshape(len(meta), TOTAL)   # [Q, TOTAL]

    results = {b: {"avg@1": [], "pass@8": [], "n_q": spans[b][1]-spans[b][0]} for b in BENCHMARKS}
    for g in range(GROUPS):
        grp = flat[:, g*N:(g+1)*N]          # [Q, N]
        pc = grp.sum(axis=1)
        for b in BENCHMARKS:
            s, e = spans[b]
            seg = pc[s:e]
            avg1 = (seg / N).mean()
            p8 = np.mean([passk(c, N, 8) for c in seg])
            results[b]["avg@1"].append(float(avg1))
            results[b]["pass@8"].append(float(p8))
            print(f"[group {g}] {b:14} avg@1={avg1:.4f} pass@8={p8:.4f}")

    print("\n======= CODE BENCHMARKS — FAST (n=%d split into %dx%d pseudo-seeds) =======" % (TOTAL, GROUPS, N))
    print(f"model: {args.model}")
    print("NOTE: groups are partitions of ONE sampling run, not fixed seeds; std understates seed variance.")
    summary = {}; a_means, p_means = [], []
    for b in BENCHMARKS:
        a = np.array(results[b]["avg@1"]); p = np.array(results[b]["pass@8"])
        summary[b] = {"n_q": results[b]["n_q"], "avg@1_mean": float(a.mean()), "avg@1_std": float(a.std()),
                      "pass@8_mean": float(p.mean()), "pass@8_std": float(p.std()),
                      "avg@1_groups": results[b]["avg@1"], "pass@8_groups": results[b]["pass@8"]}
        a_means.append(a.mean()); p_means.append(p.mean())
        print(f"{b:14} {results[b]['n_q']:>4}  avg@1={a.mean():.4f}±{a.std():.4f}  pass@8={p.mean():.4f}±{p.std():.4f}")
    json.dump({"model": args.model, "method": "fast_single_run_split", "total_n": TOTAL,
               "groups": GROUPS, "n_per_group": N, "seed": args.seed, "max_tok": MAXTOK,
               "per_benchmark": summary}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")

if __name__ == "__main__":
    main()
