import argparse, json
import numpy as np, pandas as pd
from math import comb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from verl.utils.reward_score import default_compute_score

EVAL_DIR = "/workspace/jepa-grpo-cache/eval_data"
BENCHMARKS = ["aime24", "aime25", "aime26", "amc23"]
SEEDS = [1234, 2025, 7777]
N = 8
MAXTOK = 3072

def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def main():
    global BENCHMARKS, SEEDS, N
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--benchmarks", default=None, help="comma-separated benchmark names")
    ap.add_argument("--seeds", default=None, help="comma-separated integer seeds")
    ap.add_argument("--n", type=int, default=None, help="samples per problem")
    args = ap.parse_args()
    if args.benchmarks: BENCHMARKS = [b.strip() for b in args.benchmarks.split(",")]
    if args.seeds: SEEDS = [int(s) for s in args.seeds.split(",")]
    if args.n: N = args.n

    tok = AutoTokenizer.from_pretrained(args.model)
    # Build a flat prompt list across all benchmarks, remember offsets.
    flat_prompts, meta, spans = [], [], {}
    for b in BENCHMARKS:
        df = pd.read_parquet(f"{EVAL_DIR}/{b}.parquet")
        start = len(flat_prompts)
        for _, row in df.iterrows():
            flat_prompts.append(tok.apply_chat_template(list(row["prompt"]), tokenize=False,
                                                        add_generation_prompt=True))
            meta.append((row["data_source"], row["reward_model"]["ground_truth"]))
        spans[b] = (start, len(flat_prompts))
    print(f"loaded {len(flat_prompts)} prompts across {len(BENCHMARKS)} benchmarks")

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)

    results = {b: {"avg@1": [], "pass@8": [], "pass@16": [], "n_q": spans[b][1]-spans[b][0]} for b in BENCHMARKS}
    for seed in SEEDS:
        sp = SamplingParams(n=N, temperature=0.6, top_p=0.95, max_tokens=MAXTOK, seed=seed)
        outs = llm.generate(flat_prompts, sp)
        pc = np.array([sum(1 for s in o.outputs
                           if default_compute_score(ds, s.text, gt)["acc"])
                       for o, (ds, gt) in zip(outs, meta)])
        for b in BENCHMARKS:
            s, e = spans[b]
            seg = pc[s:e]
            avg1 = (seg / N).mean()
            p8 = np.mean([passk(c, N, 8) for c in seg])
            p16 = np.mean([passk(c, N, 16) for c in seg])
            results[b]["avg@1"].append(float(avg1))
            results[b]["pass@8"].append(float(p8))
            results[b]["pass@16"].append(float(p16))
            print(f"[seed {seed}] {b:14} avg@1={avg1:.4f} pass@8={p8:.4f} pass@16={p16:.4f}")

    print("\n================ MATH BENCHMARKS — 3 seeds (n=8) ================")
    print(f"model: {args.model}")
    print(f"seeds: {SEEDS}  temp=0.6 top_p=0.95 max_tok={MAXTOK}")
    print(f"{'benchmark':14} {'n':>4} {'avg@1(mean±std)':>22} {'pass@8(mean±std)':>22}")
    summary = {}
    a_means, p8_means, p16_means = [], [], []
    for b in BENCHMARKS:
        a = np.array(results[b]["avg@1"]); p = np.array(results[b]["pass@8"]); p16 = np.array(results[b]["pass@16"])
        summary[b] = {"n_q": results[b]["n_q"], "avg@1_mean": float(a.mean()), "avg@1_std": float(a.std()),
                      "pass@8_mean": float(p.mean()), "pass@8_std": float(p.std()),
                      "pass@16_mean": float(p16.mean()), "pass@16_std": float(p16.std()),
                      "avg@1_seeds": results[b]["avg@1"], "pass@8_seeds": results[b]["pass@8"],
                      "pass@16_seeds": results[b]["pass@16"]}
        a_means.append(a.mean()); p8_means.append(p.mean()); p16_means.append(p16.mean())
        print(f"{b:14} {results[b]['n_q']:>4}  avg@1={a.mean():.4f}±{a.std():.4f}  "
              f"pass@8={p.mean():.4f}±{p.std():.4f}  pass@16={p16.mean():.4f}±{p16.std():.4f}")
    print(f"{'MACRO-AVG':14} {'':>4}  avg@1={np.mean(a_means):.4f}         "
          f"pass@8={np.mean(p8_means):.4f}         pass@16={np.mean(p16_means):.4f}")

    json.dump({"model": args.model, "seeds": SEEDS, "n": N, "max_tok": MAXTOK,
               "per_benchmark": summary, "macro_avg@1": float(np.mean(a_means)),
               "macro_pass@8": float(np.mean(p8_means)), "macro_pass@16": float(np.mean(p16_means))},
              open(args.out, "w"), indent=2)
    print(f"\nsaved -> {args.out}")

if __name__ == "__main__":
    main()
