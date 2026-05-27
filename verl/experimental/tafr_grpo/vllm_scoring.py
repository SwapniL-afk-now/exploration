from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TAFRVLLMAdapterSpec:
    name: str
    int_id: int
    path: str


TAFR_VLLM_ADAPTERS: dict[str, TAFRVLLMAdapterSpec] = {
    "anchor": TAFRVLLMAdapterSpec(name="tafr_anchor", int_id=124, path="tafr_anchor_lora_path"),
    "replay": TAFRVLLMAdapterSpec(name="tafr_replay", int_id=125, path="tafr_replay_lora_path"),
}


def tafr_vllm_adapter_spec(adapter: str) -> TAFRVLLMAdapterSpec:
    try:
        return TAFR_VLLM_ADAPTERS[adapter]
    except KeyError as exc:
        raise ValueError(f"Unknown TAFR vLLM adapter: {adapter}") from exc


def extract_response_logprobs_from_prompt_logprobs(
    *,
    prompt_logprobs: Sequence[Sequence[float]],
    prompt_len: int,
    response_len: int,
) -> list[float]:
    """Slice response-token logprobs from verl's prompt_logprobs contract.

    ``extract_prompt_logprobs`` stores the logprob for sequence token position j at
    index j - 1 because the first token has no causal logprob. The final dummy row
    keeps the array length equal to the sequence length.
    """

    if prompt_len <= 0:
        raise ValueError("prompt_len must be positive.")
    if response_len < 0:
        raise ValueError("response_len must be non-negative.")
    if response_len == 0:
        return []

    start = prompt_len - 1
    end = start + response_len
    if end > len(prompt_logprobs):
        raise ValueError(
            f"Cannot slice {response_len} response logprobs from prompt_len={prompt_len}; "
            f"prompt_logprobs has length {len(prompt_logprobs)}."
        )

    response_logprobs: list[float] = []
    for row in prompt_logprobs[start:end]:
        if not row:
            raise ValueError("prompt_logprobs rows must contain at least one logprob.")
        response_logprobs.append(float(row[0]))
    return response_logprobs
