import numpy as np, pandas as pd
from math import comb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from verl.experimental.fepo.math_parser import compute_math_reward

MODEL = "/workspace/exploration/best_final_hf"
DATA  = "/workspace/jepa-grpo-cache/eval_data/aime25.parquet"
N = 16
MAXTOK = 3072
SEEDS = [1234, 2025, 7777]

def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def main():
    df = pd.read_parquet(DATA)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True)
               for p in df["prompt"]]
    gts = [rm["ground_truth"] for rm in df["reward_model"]]
    n_q = len(gts)

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)

    results = []
    for seed in SEEDS:
        sp = SamplingParams(n=N, temperature=0.6, top_p=0.95, max_tokens=MAXTOK, seed=seed)
        outs = llm.generate(prompts, sp)
        pc = np.array([sum(1 for s in o.outputs
                           if compute_math_reward(s.text, gt, dataset_kind="aime25").is_correct)
                       for o, gt in zip(outs, gts)])
        avg1 = (pc / N).mean()
        p8 = np.mean([passk(c, N, 8) for c in pc])
        p16 = np.mean([passk(c, N, 16) for c in pc])
        results.append((seed, avg1, p8, p16, (pc > 0).sum()))
        print(f"[seed {seed}] avg@1={avg1:.4f} pass@8={p8:.4f} pass@16={p16:.4f} solved={int((pc>0).sum())}/{n_q}")

    arr = np.array([[r[1], r[2], r[3]] for r in results])
    mean, std = arr.mean(0), arr.std(0)
    print("\n================ AIME25 — 3 seeds ================")
    print(f"seeds: {SEEDS}   n={N}/q  temp=0.6 top_p=0.95 max_tok={MAXTOK}")
    print(f"{'seed':>8} {'avg@1':>8} {'pass@8':>8} {'pass@16':>8}")
    for s, a, p8, p16, _ in results:
        print(f"{s:>8} {a:>8.4f} {p8:>8.4f} {p16:>8.4f}")
    print(f"{'mean':>8} {mean[0]:>8.4f} {mean[1]:>8.4f} {mean[2]:>8.4f}")
    print(f"{'std':>8} {std[0]:>8.4f} {std[1]:>8.4f} {std[2]:>8.4f}")

if __name__ == "__main__":
    main()
