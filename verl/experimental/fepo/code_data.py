# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Convert code-generation benchmarks (HumanEval+, MBPP+, LiveCodeBench) to verl
parquet format, matching the reward-function expectations in
`verl.utils.reward_score.codegen_plus` (humanevalplus/mbppplus) and
`verl.utils.reward_score.stdio_code` (livecodebench).

LiveCodeBench note: `livecodebench/code_generation_lite` ships per-release jsonl
snapshots (test.jsonl..test6.jsonl) that are *incremental*, not cumulative, so all
of them are downloaded and deduplicated by `question_id` here. Only
`testtype: stdin` problems are kept (the majority); `functional` (LeetCode-style
class-method) problems are skipped, since that calling convention isn't handled by
`stdio_code`. Use `--cutoff-date` to filter to contamination-free problems released
after your base model's pretraining cutoff.
"""

from __future__ import annotations

import argparse
import json
import os

CODE_SYSTEM_PROMPTS = {
    "humanevalplus": (
        "You are an expert Python programmer. Complete the function below. "
        "Respond with the complete function implementation in a ```python code block."
    ),
    "mbppplus": (
        "You are an expert Python programmer. Write a Python function that solves the task "
        "below, named exactly `{entry_point}`. Respond with the complete function "
        "implementation in a ```python code block."
    ),
    "livecodebench": (
        "You are an expert competitive programmer. Solve the problem by writing a complete "
        "Python program that reads from standard input and writes the answer to standard "
        "output. Respond with the program in a ```python code block."
    ),
}


def convert_humanevalplus(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("evalplus/humanevalplus", split=split)
    rows = []
    for i, row in enumerate(ds):
        gt = json.dumps({"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]})
        rows.append({
            "data_source": "humanevalplus",
            "prompt": [
                {"role": "system", "content": CODE_SYSTEM_PROMPTS["humanevalplus"]},
                {"role": "user", "content": row["prompt"]},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": split, "index": i, "task_id": row["task_id"], "entry_point": row["entry_point"],
            },
        })
    return rows


def convert_mbppplus(split: str = "test") -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("evalplus/mbppplus", split=split)
    rows = []
    for i, row in enumerate(ds):
        # entry_point isn't a separate column; derive it from the canonical solution's
        # first `def name(` line.
        first_def = next(line for line in row["code"].splitlines() if line.strip().startswith("def "))
        entry_point = first_def.split("def ", 1)[1].split("(", 1)[0].strip()
        gt = json.dumps({"prompt": "", "test": row["test"], "entry_point": entry_point})
        rows.append({
            "data_source": "mbppplus",
            "prompt": [
                {
                    "role": "system",
                    "content": CODE_SYSTEM_PROMPTS["mbppplus"].format(entry_point=entry_point),
                },
                {"role": "user", "content": row["prompt"]},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": split, "index": i, "task_id": str(row["task_id"]), "entry_point": entry_point,
            },
        })
    return rows


def convert_livecodebench(cutoff_date: str | None = None) -> list[dict]:
    from huggingface_hub import hf_hub_download

    files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
    by_id: dict[str, dict] = {}
    for fn in files:
        path = hf_hub_download("livecodebench/code_generation_lite", filename=fn, repo_type="dataset")
        for line in open(path):
            row = json.loads(line)
            by_id[row["question_id"]] = row

    rows = []
    for i, row in enumerate(sorted(by_id.values(), key=lambda r: r["contest_date"])):
        if cutoff_date and row["contest_date"] < cutoff_date:
            continue
        pub = json.loads(row["public_test_cases"])
        if not pub or pub[0]["testtype"] != "stdin":
            continue  # functional/class-method problems not supported by stdio_code
        gt = json.dumps({"inputs": [t["input"] for t in pub], "outputs": [t["output"] for t in pub]})
        rows.append({
            "data_source": "livecodebench",
            "prompt": [
                {"role": "system", "content": CODE_SYSTEM_PROMPTS["livecodebench"]},
                {"role": "user", "content": row["question_content"]},
            ],
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": gt},
            "extra_info": {
                "split": "test", "index": i, "question_id": row["question_id"],
                "platform": row["platform"], "difficulty": row["difficulty"],
                "contest_date": row["contest_date"],
            },
        })
    return rows


def write_parquet(rows: list[dict], output_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert code benchmarks to VERL parquet format.")
    parser.add_argument("--benchmark", required=True, choices=["humanevalplus", "mbppplus", "livecodebench"])
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--cutoff-date", default=None,
        help="livecodebench only: ISO date (YYYY-MM-DD); keep problems with contest_date >= this.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.benchmark == "humanevalplus":
        rows = convert_humanevalplus(split=args.split)
    elif args.benchmark == "mbppplus":
        rows = convert_mbppplus(split=args.split)
    else:
        rows = convert_livecodebench(cutoff_date=args.cutoff_date)

    write_parquet(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
