# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

import numpy as np

from verl.experimental.fepo.math_parser import normalize_ground_truth_answer

SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "Reason step by step and put the final answer in \\boxed{}."
)
USER_SUFFIX = "Solve the problem. Show your reasoning and put the final answer in \\boxed{}."

TRAIN_DATASET_ALIASES = {
    "dapo": "zhuzilin/dapo-math-17k",
    "dapo_math_17k": "zhuzilin/dapo-math-17k",
    "skywork": "sungyub/skywork-or1-math-verl",
    "skywork_or1_math_verl": "sungyub/skywork-or1-math-verl",
}

EVAL_DATASETS = [
    "math-ai/math500",
    "math-ai/olympiadbench",
    "math-ai/amc23",
    "math-ai/aime24",
    "math-ai/aime25",
    "math-ai/aime26",
]

DAPO_PROMPT_INTRO = "Solve the following math problem step by step."
DAPO_PROMPT_REMINDER_RE = re.compile(
    r'\n\nRemember to put your answer on its own line after "Answer:"\.?\s*$',
    flags=re.IGNORECASE,
)


def dataset_short_name(dataset_id: str) -> str:
    return dataset_id.split("/")[-1].replace("_", "-").lower()


def resolve_dataset_id(dataset_id: str) -> str:
    return TRAIN_DATASET_ALIASES.get(dataset_id, dataset_id)


def make_messages(problem: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{problem}\n{USER_SUFFIX}"},
    ]


def _messages_to_problem(prompt: Any) -> str:
    if isinstance(prompt, (list, tuple)):
        messages = list(prompt)
        user_messages = [
            message for message in messages if isinstance(message, Mapping) and message.get("role") == "user"
        ]
        message = user_messages[0] if user_messages else (messages[0] if messages else {})
        content = message.get("content", "") if isinstance(message, Mapping) else str(message)
    elif isinstance(prompt, Mapping):
        content = prompt.get("content", "")
    else:
        content = str(prompt)
    return str(content).strip()


def extract_dapo_problem(prompt: Any) -> str:
    problem = _messages_to_problem(prompt)
    if problem.startswith(DAPO_PROMPT_INTRO) and "\n\n" in problem:
        problem = problem.split("\n\n", 1)[1].strip()
    problem = DAPO_PROMPT_REMINDER_RE.sub("", problem).strip()
    if not problem:
        raise ValueError(f"Could not extract DAPO-Math-17K problem from prompt: {prompt!r}")
    return problem


def _pick(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _extract_problem(row: Mapping[str, Any], dataset_id: str) -> str:
    short = dataset_short_name(dataset_id)
    if short in {"dapo-math-17k"} or dataset_id == "zhuzilin/dapo-math-17k":
        return extract_dapo_problem(row["prompt"])

    prompt = _pick(row, ["prompt", "messages"])
    if prompt is not None:
        return _messages_to_problem(prompt)

    problem = _pick(row, ["problem", "question", "Problem", "Question", "prompt_text"])
    if problem is None:
        raise KeyError(f"Could not find a problem/question column for {dataset_id}. Found columns: {list(row.keys())}")
    return str(problem).strip()


def _extract_ground_truth(row: Mapping[str, Any], dataset_id: str) -> Any:
    reward_model = row.get("reward_model")
    if isinstance(reward_model, Mapping) and reward_model.get("ground_truth") is not None:
        return reward_model["ground_truth"]
    answer = _pick(row, ["label", "answer", "final_answer", "ground_truth", "target"])
    if answer is None:
        solution = _pick(row, ["solution", "solutions"])
        if solution is not None:
            answer = solution
    if answer is None:
        raise KeyError(f"Could not find an answer column for {dataset_id}. Found columns: {list(row.keys())}")
    return answer


def adapt_row_to_verl(row: Mapping[str, Any], dataset_id: str, split: str, index: int) -> dict[str, Any]:
    dataset_id = resolve_dataset_id(dataset_id)
    short = dataset_short_name(dataset_id)
    problem = _extract_problem(row, dataset_id)
    raw_answer = _extract_ground_truth(row, dataset_id)
    target = normalize_ground_truth_answer(raw_answer, dataset_kind=short)
    if target is None:
        raise ValueError(f"Could not normalize answer for {dataset_id} row {index}: {raw_answer!r}")

    raw_prompt = row.get("prompt")
    if short == "skywork-or1-math-verl" and isinstance(raw_prompt, list):
        prompt = raw_prompt
    else:
        prompt = make_messages(problem)

    ground_truth_raw = raw_answer
    if isinstance(ground_truth_raw, np.ndarray):
        ground_truth_raw = ground_truth_raw.tolist()
    if isinstance(ground_truth_raw, (list, tuple)):
        ground_truth_raw = ground_truth_raw[0] if len(ground_truth_raw) > 0 else ""
    if not isinstance(ground_truth_raw, str):
        ground_truth_raw = str(ground_truth_raw)

    return {
        "data_source": short,
        "prompt": prompt,
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": target},
        "extra_info": {
            "split": split,
            "index": int(index),
            "source_dataset": dataset_id,
            "problem": problem,
            "ground_truth_raw": ground_truth_raw,
        },
    }


def convert_split_to_verl_rows(split_data, dataset_id: str, split: str, max_samples: Optional[int] = None):
    rows = []
    limit = len(split_data) if max_samples is None or max_samples < 0 else min(len(split_data), max_samples)
    for idx in range(limit):
        rows.append(adapt_row_to_verl(split_data[idx], dataset_id=dataset_id, split=split, index=idx))
    return rows


def load_hf_split(dataset_id: str, split: str, config: Optional[str] = None, streaming: bool = False):
    from datasets import load_dataset

    dataset_id = resolve_dataset_id(dataset_id)
    try:
        if config:
            return load_dataset(dataset_id, config, split=split, streaming=streaming)
        return load_dataset(dataset_id, split=split, streaming=streaming)
    except Exception as exc:
        if dataset_short_name(dataset_id) != "skywork-or1-math-verl":
            raise
        return _load_hf_parquet_fallback(load_dataset, dataset_id, split, streaming, exc)


def _load_hf_parquet_fallback(load_dataset_fn, dataset_id: str, split: str, streaming: bool, original_exc: Exception):
    """Load a dataset repo's parquet files directly when HF feature casting fails."""

    try:
        from huggingface_hub import HfApi, hf_hub_url

        api = HfApi()
        parquet_files = [
            path
            for path in api.list_repo_files(dataset_id, repo_type="dataset")
            if path.endswith((".parquet", ".parquet.gz"))
        ]
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in dataset repo {dataset_id}")
        parquet_urls = [
            hf_hub_url(repo_id=dataset_id, filename=path, repo_type="dataset") for path in sorted(parquet_files)
        ]
        return load_dataset_fn("parquet", data_files={split: parquet_urls}, split=split, streaming=streaming)
    except Exception as fallback_exc:
        raise RuntimeError(
            f"Failed to load {dataset_id!r} split {split!r} normally and through parquet fallback. "
            f"Original load error: {original_exc}"
        ) from fallback_exc


def write_parquet(rows: list[dict[str, Any]], output_path: str) -> None:
    import pandas as pd

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FEPO math datasets to VERL parquet format.")
    parser.add_argument("--dataset", required=True, help="HF dataset id or alias.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    split_data = load_hf_split(args.dataset, split=args.split, config=args.config)
    rows = convert_split_to_verl_rows(
        split_data,
        dataset_id=args.dataset,
        split=args.split,
        max_samples=args.max_samples,
    )
    write_parquet(rows, args.output)


if __name__ == "__main__":
    main()
