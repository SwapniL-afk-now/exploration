# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
import math
import re
from typing import Any

from verl.experimental.fepo.math_parser import compute_math_reward


def _prediction_key(record: dict[str, Any]) -> str:
    pred = record.get("prediction_normalized")
    if pred is None:
        pred = record.get("pred")
    if pred is None:
        pred = "<unparsed>"
    return str(pred)




def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5)


def _entropy_from_counts(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)


def _repetition_rate(text: Any, ngram: int = 4) -> float:
    tokens = re.findall(r"\S+", str(text).lower())
    if len(tokens) < ngram * 2:
        return 0.0
    ngrams = [tuple(tokens[i : i + ngram]) for i in range(len(tokens) - ngram + 1)]
    if not ngrams:
        return 0.0
    return float(1.0 - (len(set(ngrams)) / len(ngrams)))


def _grouped_generation_metrics(groups: Sequence[Sequence[dict[str, Any]]], prefix: str, cutoffs: Sequence[int] = (1, 4, 8, 16)) -> dict[str, float | int]:
    prompt_total = len(groups)
    generation_total = sum(len(group) for group in groups)
    correct_total = sum(1 for group in groups for item in group if float(item.get("reward", 0.0)) > 0.0)
    length_values = [float(item.get("response_length", len(str(item.get("response", "")).split()))) for group in groups for item in group]
    repetition_values = [_repetition_rate(item.get("response", "")) for group in groups for item in group]
    unique_response_rates = []
    unique_answer_rates = []
    answer_entropies = []
    majority_correct = {int(k): 0.0 for k in cutoffs}
    pass_counts = {int(k): 0.0 for k in cutoffs}
    all_wrong = 0.0
    all_correct = 0.0
    mixed = 0.0

    for group in groups:
        rewards = [float(item.get("reward", 0.0)) > 0.0 for item in group]
        answers = [_prediction_key(item) for item in group]
        responses = [str(item.get("response", "")) for item in group]
        if rewards:
            all_wrong += float(not any(rewards))
            all_correct += float(all(rewards))
            mixed += float(any(rewards) and not all(rewards))
        if responses:
            unique_response_rates.append(len(set(responses)) / len(responses))
        if answers:
            counts = dict(Counter(answers))
            unique_answer_rates.append(len(counts) / len(answers))
            answer_entropies.append(_entropy_from_counts(counts))
        for cutoff in cutoffs:
            subset_rewards = rewards[:cutoff]
            subset_answers = answers[:cutoff]
            pass_counts[cutoff] += float(any(subset_rewards)) if subset_rewards else 0.0
            if subset_rewards and subset_answers:
                answer_counts = Counter(subset_answers)
                majority_answer = sorted(answer_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
                majority_is_correct = any(answer == majority_answer and reward for answer, reward in zip(subset_answers, subset_rewards, strict=False))
                majority_correct[cutoff] += float(majority_is_correct)

    metrics: dict[str, float | int] = {
        f"{prefix}/avg_at_k": float(correct_total / generation_total) if generation_total else 0.0,
        f"{prefix}/total_prompts": int(prompt_total),
        f"{prefix}/total_generations": int(generation_total),
        f"{prefix}/unique_response_ratio_at_k": _mean(unique_response_rates),
        f"{prefix}/unique_answer_ratio_at_k": _mean(unique_answer_rates),
        f"{prefix}/answer_entropy_at_k": _mean(answer_entropies),
        f"{prefix}/all_wrong_group_rate": float(all_wrong / prompt_total) if prompt_total else 0.0,
        f"{prefix}/all_correct_group_rate": float(all_correct / prompt_total) if prompt_total else 0.0,
        f"{prefix}/mixed_group_rate": float(mixed / prompt_total) if prompt_total else 0.0,
        f"{prefix}/response_length_mean": _mean(length_values),
        f"{prefix}/response_length_std": _std(length_values),
        f"{prefix}/repetition_rate": _mean(repetition_values),
    }
    for cutoff in cutoffs:
        metrics[f"{prefix}/pass_at_{cutoff}"] = float(pass_counts[cutoff] / prompt_total) if prompt_total else 0.0
        metrics[f"{prefix}/maj_at_{cutoff}"] = float(majority_correct[cutoff] / prompt_total) if prompt_total else 0.0
    metrics[f"{prefix}/pass_at_k"] = metrics[f"{prefix}/pass_at_{max(cutoffs)}"] if cutoffs else 0.0
    metrics[f"{prefix}/maj_at_k"] = metrics[f"{prefix}/maj_at_{max(cutoffs)}"] if cutoffs else 0.0
    return metrics

def summarize_exploration_records(records: Sequence[dict[str, Any]], prefix: str = "train") -> dict[str, float]:
    """Summarize grouped generation diversity for collapse/exploration monitoring."""

    groups: dict[str, list[dict[str, Any]]] = {}
    for idx, record in enumerate(records):
        prompt_uid = str(record.get("prompt_uid", f"ungrouped-{idx}"))
        groups.setdefault(prompt_uid, []).append(record)

    if not groups:
        return {
            f"{prefix}/exploration/unique_prediction_rate": 0.0,
            f"{prefix}/exploration/majority_prediction_share": 0.0,
            f"{prefix}/exploration/collapse_rate": 0.0,
            f"{prefix}/exploration/reward_std_mean": 0.0,
        }

    unique_rates = []
    majority_shares = []
    collapse_flags = []
    reward_stds = []
    for items in groups.values():
        preds = [_prediction_key(item) for item in items]
        counts: dict[str, int] = {}
        for pred in preds:
            counts[pred] = counts.get(pred, 0) + 1
        n = max(len(items), 1)
        unique_count = len(counts)
        unique_rates.append(unique_count / n)
        majority_shares.append(max(counts.values()) / n if counts else 0.0)
        collapse_flags.append(float(unique_count <= 1))
        rewards = [float(item.get("reward", 0.0)) for item in items]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        reward_stds.append((sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5 if rewards else 0.0)

    def mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        f"{prefix}/exploration/unique_prediction_rate": mean(unique_rates),
        f"{prefix}/exploration/majority_prediction_share": mean(majority_shares),
        f"{prefix}/exploration/collapse_rate": mean(collapse_flags),
        f"{prefix}/exploration/reward_std_mean": mean(reward_stds),
    }


def summarize_training_records(records: Sequence[dict[str, Any]]) -> dict[str, float | int]:
    """Summarize per-generation train records with prompt-level pass@N."""

    if not records:
        return {
            "train/accuracy": 0.0,
            "train/failure_rate": 0.0,
            "train/correct_count": 0,
            "train/response_acc": 0.0,
            "train/prompt_pass_at_n": 0.0,
            "train/pass_at_k": 0.0,
            "train/avg_at_k": 0.0,
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
            "fepo/reference_kl_all": 0.0,
        }

    total = len(records)
    correct_count = sum(1 for record in records if float(record.get("reward", 0.0)) > 0.0)
    prompt_success: dict[str, bool] = {}
    for idx, record in enumerate(records):
        prompt_uid = str(record.get("prompt_uid", f"ungrouped-{idx}"))
        prompt_success[prompt_uid] = prompt_success.get(prompt_uid, False) or float(record.get("reward", 0.0)) > 0.0

    def average(key: str) -> float:
        return float(sum(float(record.get(key, 0.0)) for record in records) / total)

    def total_count(key: str) -> float:
        return float(sum(float(record.get(key, 0.0)) for record in records))

    metrics = {
        "train/accuracy": float(correct_count / total) if total else 0.0,
        "train/failure_rate": float(1.0 - (correct_count / total)) if total else 0.0,
        "train/correct_count": int(correct_count),
        "train/response_acc": float(correct_count / total) if total else 0.0,
        "train/prompt_pass_at_n": float(sum(prompt_success.values()) / len(prompt_success)) if prompt_success else 0.0,
        "train/pass_at_k": float(sum(prompt_success.values()) / len(prompt_success)) if prompt_success else 0.0,
        "train/avg_at_k": float(correct_count / total) if total else 0.0,
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
        "fepo/reference_kl_all": average("reference_kl_all"),
        "fepo/token_ratio_mean": average("token_ratio_mean"),
        "fepo/token_log_ratio_abs_mean": average("token_log_ratio_abs_mean"),
        "fepo/loss_token_count": total_count("loss_token_count"),
        "fepo/failed_token_count": total_count("failed_token_count"),
    }
    train_groups: dict[str, list[dict[str, Any]]] = {}
    for idx, record in enumerate(records):
        prompt_uid = str(record.get("prompt_uid", f"ungrouped-{idx}"))
        train_groups.setdefault(prompt_uid, []).append(record)
    metrics.update(_grouped_generation_metrics(list(train_groups.values()), prefix="train"))
    metrics.update(summarize_exploration_records(records, prefix="train"))
    return metrics


def evaluate_responses_by_prompt(
    examples: Sequence[dict[str, Any]],
    responses_by_prompt: Sequence[Sequence[str]],
    dataset_name: str,
    k: int,
) -> dict[str, float | int | str]:
    """Compute pass@K/avg@K/maj@K and diversity metrics for grouped generations."""

    groups = []
    parseable_total = 0
    generation_total = 0
    for example, responses in zip(examples, responses_by_prompt, strict=False):
        target = example.get("ground_truth_normalized") or example.get("reward_model", {}).get("ground_truth")
        items = []
        for response in list(responses)[:k]:
            result = compute_math_reward(response, target, dataset_kind=dataset_name)
            parseable_total += int(result.has_parseable_answer)
            generation_total += 1
            items.append(
                {
                    "reward": float(result.is_correct),
                    "prediction_normalized": result.prediction_normalized,
                    "response": response,
                    "response_length": len(str(response).split()),
                }
            )
        groups.append(items)

    cutoffs = tuple(cutoff for cutoff in (1, 4, 8, 16) if cutoff <= max(k, 1))
    metrics = _grouped_generation_metrics(groups, prefix="", cutoffs=cutoffs or (1,))
    metrics = {key.removeprefix("/"): value for key, value in metrics.items()}
    metrics.update(
        {
            "dataset": dataset_name,
            "total_prompts": int(len(examples)),
            "total_generations": int(generation_total),
            "k": int(k),
            "parse_rate": float(parseable_total / generation_total) if generation_total else 0.0,
            "exploration_unique_prediction_rate": metrics.get("unique_answer_ratio_at_k", 0.0),
            "exploration_majority_prediction_share": 1.0 - float(metrics.get("unique_answer_ratio_at_k", 0.0)),
            "exploration_collapse_rate": metrics.get("all_wrong_group_rate", 0.0) + metrics.get("all_correct_group_rate", 0.0),
        }
    )
    return metrics
