# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from verl.experimental.tafr_grpo.failure_data_collector import FailureExample


@dataclass
class FailureSFTStepResult:
    loss: float
    token_count: int
    batch_size: int


def failure_sft_loss(log_prob: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    """Wrong-response SFT objective with response-token-only masking.

    L_fail-SFT(phi) = - mean_examples [ mean_response_tokens log pi_phi(y^-|x) ]
    """

    mask = response_mask.to(device=log_prob.device, dtype=log_prob.dtype)
    token_count = mask.sum(dim=-1).clamp_min(1.0)
    seq_loss = -(log_prob * mask).sum(dim=-1) / token_count
    valid_seq = (mask.sum(dim=-1) > 0).to(dtype=log_prob.dtype)
    return (seq_loss * valid_seq).sum() / valid_seq.sum().clamp_min(1.0)


class FailureSFTTrainer:
    """Minimal separate-optimizer trainer for pi_phi.

    The class is intentionally framework-light so it can be used in unit tests
    and wrapped by a verl worker. The model and optimizer passed here must be
    distinct from the GRPO actor model and optimizer.
    """

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        self.optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}

    def assert_separate_from(self, other_optimizer: torch.optim.Optimizer) -> None:
        other_param_ids = {id(p) for group in other_optimizer.param_groups for p in group["params"]}
        overlap = self.optimizer_param_ids & other_param_ids
        if overlap:
            raise AssertionError("Failure-SFT optimizer shares parameters with the GRPO optimizer.")

    def step_tensor_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, response_mask: torch.Tensor):
        self.model.train()
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits[:, :-1]
        labels = input_ids[:, 1:]
        shifted_mask = response_mask[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        loss = failure_sft_loss(log_probs, shifted_mask)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return FailureSFTStepResult(
            loss=float(loss.detach().cpu()),
            token_count=int(shifted_mask.sum().detach().cpu()),
            batch_size=int(input_ids.shape[0]),
        )

    def build_text_batch(self, examples: Iterable[FailureExample], tokenizer, max_length: int):
        messages = []
        for example in examples:
            messages.append(
                [
                    {"role": "user", "content": example.prompt},
                    {"role": "assistant", "content": example.wrong_response},
                ]
            )
        if not messages:
            raise ValueError("Cannot build a failure-SFT batch from zero examples.")

        encoded_rows = []
        response_masks = []
        for row in messages:
            prompt_ids = tokenizer.apply_chat_template([row[0]], add_generation_prompt=True, tokenize=True)
            full_ids = tokenizer.apply_chat_template(row, add_generation_prompt=False, tokenize=True)
            ids = torch.as_tensor(full_ids[:max_length], dtype=torch.long)
            mask = torch.zeros_like(ids)
            prompt_len = min(len(prompt_ids), ids.numel())
            mask[prompt_len:] = 1
            encoded_rows.append(ids)
            response_masks.append(mask)

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        max_len = max(row.numel() for row in encoded_rows)
        input_ids = torch.full((len(encoded_rows), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        response_mask = torch.zeros_like(input_ids)
        for i, ids in enumerate(encoded_rows):
            n = ids.numel()
            input_ids[i, :n] = ids
            attention_mask[i, :n] = 1
            response_mask[i, :n] = response_masks[i]
        return input_ids, attention_mask, response_mask
