import numpy as np, pandas as pd
from math import comb
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from verl.experimental.fepo.math_parser import compute_math_reward

MODEL = "/workspace/exploration/best_final_hf"
DATA  = "/workspace/jepa-grpo-cache/eval_data/aime25.parquet"
N = 16
MAXTOK = 3072

def passk(c, n, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def main():
    df = pd.read_parquet(DATA)
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = [tok.apply_chat_template(list(p), tokenize=False, add_generation_prompt=True)
               for p in df["prompt"]]
    gts = [rm["ground_truth"] for rm in df["reward_model"]]

    llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=4096)
    sp = SamplingParams(n=N, temperature=0.6, top_p=0.95, max_tokens=MAXTOK, seed=1234)
    outs = llm.generate(prompts, sp)

    per_q_correct = []
    for o, gt in zip(outs, gts):
        corr = sum(1 for s in o.outputs if compute_math_reward(s.text, gt, dataset_kind="aime25").is_correct)
        per_q_correct.append(corr)

    per_q_correct = np.array(per_q_correct)
    n_q = len(per_q_correct)
    avg_at_1 = (per_q_correct / N).mean()
    p8  = np.mean([passk(c, N, 8)  for c in per_q_correct])
    p16 = np.mean([passk(c, N, 16) for c in per_q_correct])

    print("\n================ AIME25 EVAL ================")
    print(f"model: {MODEL}")
    print(f"questions: {n_q}   samples/q: {N}   temp=0.6 top_p=0.95 max_tok={MAXTOK}")
    print(f"avg@1 (mean accuracy): {avg_at_1:.4f}")
    print(f"pass@8 : {p8:.4f}")
    print(f"pass@16: {p16:.4f}")
    print(f"solved >=1: {(per_q_correct>0).sum()}/{n_q}")
    print("per-question correct/16:", per_q_correct.tolist())

if __name__ == "__main__":
    main()
