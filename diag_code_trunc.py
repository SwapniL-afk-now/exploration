"""Diagnose why 'ours' scores lower on code: measure truncation + fence-extraction
failure rates per model on humanevalplus + livecodebench. Generation only (no GPU
needed for scoring)."""
import argparse, re, os
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

EVAL_DIR = "/workspace/jepa-grpo-cache/eval_data"
MAXTOK, MAXLEN = 3072, 4096
_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

def run(model, benchmarks, n, seed):
    tok = AutoTokenizer.from_pretrained(model)
    prompts, per_max, spans = [], [], {}
    for b in benchmarks:
        df = pd.read_parquet(f"{EVAL_DIR}/{b}.parquet")
        start = len(prompts)
        for _, row in df.iterrows():
            t = tok.apply_chat_template(list(row["prompt"]), tokenize=False, add_generation_prompt=True)
            prompts.append(t); per_max.append(max(256, min(MAXTOK, MAXLEN - len(tok(t)["input_ids"]) - 8)))
        spans[b] = (start, len(prompts))
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=MAXLEN)
    sps = [SamplingParams(n=n, temperature=0.6, top_p=0.95, max_tokens=m, seed=seed) for m in per_max]
    outs = llm.generate(prompts, sps)
    print(f"\n==== {model} ====")
    for b in benchmarks:
        s, e = spans[b]
        tot = trunc = nofence = ngen = 0
        for o in outs[s:e]:
            for samp in o.outputs:
                ngen += 1
                if samp.finish_reason == "length": trunc += 1
                if not _FENCE.findall(samp.text): nofence += 1
            tot += 1
        print(f"{b:14} q={tot} gen={ngen}  truncated={trunc/ngen:.3f}  no_closed_fence={nofence/ngen:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--benchmarks", default="humanevalplus,livecodebench")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    run(a.model, [b.strip() for b in a.benchmarks.split(",")], a.n, a.seed)
