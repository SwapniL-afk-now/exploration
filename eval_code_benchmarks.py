import argparse, json, os, re, subprocess, sys
import numpy as np, pandas as pd
from math import comb
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

EVAL_DIR = "/workspace/jepa-grpo-cache/eval_data"
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "code_judge_worker.py")
BENCHMARKS = ["humanevalplus", "mbppplus", "livecodebench"]
SEEDS = [1234, 2025, 7777]
N = 8
MAXTOK = 3072
MAXLEN = 4096
JOB_TIMEOUT = 30  # overall wall-clock cap per program (backstop to the per-test SIGALRM)
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def _extract_code(text):
    m = _FENCE.findall(text)
    return m[-1] if m else text

def make_job(ds, text, gt):
    """Build a judge job (same pass/fail logic as verl's codegen_plus / stdio_code)."""
    spec = json.loads(gt) if isinstance(gt, str) else dict(gt)
    completion = _extract_code(text)
    if ds in ("humanevalplus", "mbppplus"):
        ep = spec["entry_point"]
        code = completion if f"def {ep}" in completion else f"{spec.get('prompt','')}\n{completion}"
        return {"kind": "assert", "code": code, "test": spec["test"], "entry_point": ep}
    else:  # livecodebench: stdin/stdout
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
    global BENCHMARKS, SEEDS, N
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="code_eval_results.json")
    ap.add_argument("--benchmarks", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    if args.benchmarks: BENCHMARKS = [b.strip() for b in args.benchmarks.split(",")]
    if args.seeds: SEEDS = [int(s) for s in args.seeds.split(",")]
    if args.n: N = args.n

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
    # Per-prompt max_tokens: keep MAXTOK, shrink only for prompts that wouldn't fit the 4096 context.
    per_max = [max(256, min(MAXTOK, MAXLEN - pl - 8)) for pl in prompt_lens]
    n_capped = sum(1 for pl, m in zip(prompt_lens, per_max) if m < MAXTOK)
    print(f"loaded {len(flat_prompts)} prompts across {len(BENCHMARKS)} code benchmarks; "
          f"{n_capped} prompts capped below {MAXTOK} gen tokens to fit {MAXLEN} context")

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=MAXLEN)
    pool = ThreadPoolExecutor(max_workers=(os.cpu_count() or 16))

    results = {b: {"avg@1": [], "pass@8": [], "n_q": spans[b][1]-spans[b][0]} for b in BENCHMARKS}
    for seed in SEEDS:
        sps = [SamplingParams(n=N, temperature=0.6, top_p=0.95, max_tokens=m, seed=seed) for m in per_max]
        outs = llm.generate(flat_prompts, sps)
        # Flatten all (sample, meta) pairs and score in parallel.
        jobs = []
        for o, (ds, gt, ei) in zip(outs, meta):
            for s in o.outputs:
                jobs.append((ds, s.text, gt, ei))
        flags = list(pool.map(lambda a: score_one(*a), jobs))
        # Reduce back: each question has N consecutive flags.
        pc = np.array([sum(flags[i*N:(i+1)*N]) for i in range(len(meta))])
        for b in BENCHMARKS:
            s, e = spans[b]
            seg = pc[s:e]
            avg1 = (seg / N).mean()
            p8 = np.mean([passk(c, N, 8) for c in seg])
            results[b]["avg@1"].append(float(avg1))
            results[b]["pass@8"].append(float(p8))
            print(f"[seed {seed}] {b:14} avg@1={avg1:.4f} pass@8={p8:.4f}")

    print("\n================ CODE BENCHMARKS — 3 seeds (n=8) ================")
    print(f"model: {args.model}")
    summary = {}; a_means, p_means = [], []
    for b in BENCHMARKS:
        a = np.array(results[b]["avg@1"]); p = np.array(results[b]["pass@8"])
        summary[b] = {"n_q": results[b]["n_q"], "avg@1_mean": float(a.mean()), "avg@1_std": float(a.std()),
                      "pass@8_mean": float(p.mean()), "pass@8_std": float(p.std()),
                      "avg@1_seeds": results[b]["avg@1"], "pass@8_seeds": results[b]["pass@8"]}
        a_means.append(a.mean()); p_means.append(p.mean())
        print(f"{b:14} {results[b]['n_q']:>4}  avg@1={a.mean():.4f}±{a.std():.4f}  pass@8={p.mean():.4f}±{p.std():.4f}")
    print(f"{'MACRO-AVG':14} {'':>4}  avg@1={np.mean(a_means):.4f}         pass@8={np.mean(p_means):.4f}")
    json.dump({"model": args.model, "seeds": SEEDS, "n": N, "max_tok": MAXTOK,
               "per_benchmark": summary, "macro_avg@1": float(np.mean(a_means)),
               "macro_pass@8": float(np.mean(p_means))}, open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")

if __name__ == "__main__":
    main()
