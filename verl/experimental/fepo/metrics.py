# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from verl.experimental.fepo.math_parser import compute_math_reward


def summarize_training_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize per-generation train records with prompt-level pass@N."""

    if not records:
        return {
            "train/response_acc": 0.0,
            "train/prompt_pass_at_n": 0.0,
            "train/parse_rate": 0.0,
            "train/response_count": 0,
            "train/prompt_count": 0,
            "train/avg_response_length": 0.0,
            "train/valid_token_count": 0.0,
            "train/ignored_token_count": 0.0,
            "train/active_response_count_mean": 0.0,
            "perf/generation_seconds": 0.0,
            "perf/avg_generation_seconds": 0.0,
            "fepo/solver_loss": 0.0,
            "fepo/solver_scaled_loss": 0.0,
            "fepo/failure_sft_loss": 0.0,
            "fepo/failure_scaled_loss": 0.0,
            "fepo/failure_active_response_count_mean": 0.0,
            "fepo/failure_escape_kl_failed": 0.0,
            "fepo/reference_kl_failed": 0.0,
        }

    total = len(records)
    prompt_success: dict[str, bool] = {}
    for idx, record in enumerate(records):
        prompt_uid = str(record.get("prompt_uid", f"ungrouped-{idx}"))
        prompt_success[prompt_uid] = prompt_success.get(prompt_uid, False) or float(record.get("reward", 0.0)) > 0.0

    def average(key: str) -> float:
        return float(sum(float(record.get(key, 0.0)) for record in records) / total)

    def total_count(key: str) -> float:
        return float(sum(float(record.get(key, 0.0)) for record in records))

    return {
        "train/response_acc": average("reward"),
        "train/prompt_pass_at_n": float(sum(prompt_success.values()) / len(prompt_success)) if prompt_success else 0.0,
        "train/parse_rate": average("has_parseable_answer"),
        "train/response_count": int(total),
        "train/prompt_count": int(len(prompt_success)),
        "train/avg_response_length": average("response_length"),
        "train/valid_token_count": total_count("valid_token_count"),
        "train/ignored_token_count": total_count("ignored_token_count"),
        "train/active_response_count_mean": average("active_response_count"),
        "perf/generation_seconds": total_count("generation_seconds"),
        "perf/avg_generation_seconds": average("generation_seconds"),
        "fepo/solver_loss": average("solver_total_loss"),
        "fepo/solver_scaled_loss": average("solver_scaled_loss"),
        "fepo/solver_pg_loss": average("solver_pg_loss"),
        "fepo/solver_clip_fraction": average("solver_clip_fraction"),
        "fepo/failure_sft_loss": average("failure_sft_loss"),
        "fepo/failure_scaled_loss": average("failure_scaled_loss"),
        "fepo/failure_sft_active_rate": average("failure_is_sft_active"),
        "fepo/failure_active_response_count_mean": average("failure_active_response_count"),
        "fepo/failure_escape_kl_failed": average("failure_escape_kl_failed"),
        "fepo/reference_kl_failed": average("reference_kl_failed"),
        "fepo/token_ratio_mean": average("token_ratio_mean"),
        "fepo/token_log_ratio_abs_mean": average("token_log_ratio_abs_mean"),
        "fepo/loss_token_count": total_count("loss_token_count"),
        "fepo/failed_token_count": total_count("failed_token_count"),
    }


def evaluate_responses_by_prompt(
    examples: Sequence[dict[str, Any]],
    responses_by_prompt: Sequence[Sequence[str]],
    dataset_name: str,
    k: int,
) -> dict[str, float | int | str]:
    """Compute pass@1/pass@k/avg@k/parse_rate for grouped generations."""

    prompt_total = len(examples)
    prompt_first_correct = 0
    prompt_any_correct = 0
    generation_correct = 0
    generation_total = 0
    parseable_total = 0

    for example, responses in zip(examples, responses_by_prompt, strict=False):
        target = example.get("ground_truth_normalized") or example.get("reward_model", {}).get("ground_truth")
        generation_results = []
        for response in list(responses)[:k]:
            result = compute_math_reward(response, target, dataset_kind=dataset_name)
            parseable_total += int(result.has_parseable_answer)
            generation_correct += int(result.is_correct)
            generation_total += 1
            generation_results.append(result.is_correct)

        prompt_first_correct += int(bool(generation_results and generation_results[0]))
        prompt_any_correct += int(any(generation_results))

    return {
        "dataset": dataset_name,
        "total_prompts": int(prompt_total),
        "total_generations": int(generation_total),
        "k": int(k),
        "pass_at_1": float(prompt_first_correct / prompt_total) if prompt_total else 0.0,
        "pass_at_k": float(prompt_any_correct / prompt_total) if prompt_total else 0.0,
        "avg_at_k": float(generation_correct / generation_total) if generation_total else 0.0,
        "parse_rate": float(parseable_total / generation_total) if generation_total else 0.0,
    }
