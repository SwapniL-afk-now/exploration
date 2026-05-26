# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from dataclasses import dataclass

import torch

from verl.utils.torch_functional import get_response_mask


@dataclass
class ReplaySampleBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    replay_log_probs: torch.Tensor


def gather_token_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1).gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def sample_replay_responses(
    *,
    replay_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eos_token_id: int | list[int],
    pad_token_id: int,
    max_new_tokens: int,
    num_samples: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> ReplaySampleBatch:
    """Generate replay samples from frozen pi_replay and return frozen logprobs."""

    replay_model.eval()
    repeated_input_ids = input_ids.repeat_interleave(num_samples, dim=0)
    repeated_attention_mask = attention_mask.repeat_interleave(num_samples, dim=0)
    generated = replay_model.generate(
        input_ids=repeated_input_ids,
        attention_mask=repeated_attention_mask,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    prompt_len = repeated_input_ids.shape[1]
    response_ids = generated[:, prompt_len:]
    response_mask = get_response_mask(response_ids, eos_token=eos_token_id, dtype=repeated_attention_mask.dtype)
    full_attention_mask = torch.cat([repeated_attention_mask, response_mask], dim=-1)
    output = replay_model(input_ids=generated, attention_mask=full_attention_mask)
    token_log_probs = gather_token_log_probs(output.logits[:, :-1], generated[:, 1:])
    replay_response_log_probs = token_log_probs[:, -response_ids.shape[1] :].detach()
    return ReplaySampleBatch(
        input_ids=generated,
        attention_mask=full_attention_mask,
        response_mask=response_mask,
        replay_log_probs=replay_response_log_probs,
    )
