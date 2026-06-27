"""Compute unbiased pass@1 / pass@N from an EvalPlus `*.eval_results.json`.

EvalPlus's own `evaluate` prints pass@1 (and a couple of fixed k's), but not an
arbitrary pass@N. This reuses EvalPlus's `estimate_pass_at_k` over the per-sample
base/plus statuses so HumanEval+/MBPP+ get a faithful pass@N. A "plus" pass requires
the sample to pass BOTH the base and the plus (augmented) test suites.

Usage: python eval_passk_from_evalplus.py <eval_results.json> <N>
Emits JSON: {"base": {"pass@1":..,"pass@N":..}, "plus": {...}, "n_tasks":..}
"""
import json, sys
import numpy as np
from evalplus.eval import estimate_pass_at_k

PASS = "pass"


def main():
    path, N = sys.argv[1], int(sys.argv[2])
    data = json.load(open(path))["eval"]
    total, base_correct, plus_correct = [], [], []
    for _task, res in data.items():
        total.append(len(res))
        base_correct.append(sum(r["base_status"] == PASS for r in res))
        plus_correct.append(sum(r["base_status"] == PASS and r["plus_status"] == PASS for r in res))
    total = np.array(total)
    out = {"n_tasks": int(len(total)), "base": {}, "plus": {}}
    for k in (1, N):
        out["base"][f"pass@{k}"] = float(estimate_pass_at_k(total, np.array(base_correct), k).mean())
        out["plus"][f"pass@{k}"] = float(estimate_pass_at_k(total, np.array(plus_correct), k).mean())
    print(json.dumps(out))


if __name__ == "__main__":
    main()
