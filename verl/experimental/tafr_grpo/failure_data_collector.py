# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

import pandas as pd


@dataclass
class FailureExample:
    """One wrong-response SFT example.

    The replay signal is not trained directly from this stream. These examples
    only train pi_phi with response-token-only SFT; pi_replay is later built
    from lagged failure checkpoints.
    """

    prompt: Any
    wrong_response: Any
    reward: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.reward != 0:
            raise ValueError("FailureExample only accepts reward=0 examples.")


class FailureDataCollector:
    """Bounded wrong-response stream for failure-SFT.

    This is intentionally not a replay-buffer loss. GRPO never optimizes
    directly against this collector; only the separate failure-SFT optimizer
    consumes it to update pi_phi.
    """

    def __init__(
        self,
        max_size: Optional[int] = None,
        sampling: str = "recent",
        seed: Optional[int] = None,
    ):
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None.")
        if sampling not in {"recent", "uniform"}:
            raise ValueError("sampling must be 'recent' or 'uniform'.")
        self.max_size = max_size
        self.sampling = sampling
        self._rng = random.Random(seed)
        self._examples: deque[FailureExample] | list[FailureExample]
        self._examples = deque(maxlen=max_size) if sampling == "recent" else []
        self.total_seen = 0

    def __len__(self) -> int:
        return len(self._examples)

    def add(
        self, *, prompt: Any, response: Any, reward: float | int, metadata: Optional[dict[str, Any]] = None
    ) -> bool:
        """Append a reward-0 response and reject every reward-1 response."""

        if float(reward) != 0.0:
            return False
        example = FailureExample(prompt=prompt, wrong_response=response, reward=0, metadata=metadata or {})
        self.total_seen += 1
        if self.sampling == "recent":
            self._examples.append(example)
            return True

        examples = self._examples
        assert isinstance(examples, list)
        if self.max_size is None or len(examples) < self.max_size:
            examples.append(example)
        else:
            # Reservoir replacement keeps a bounded uniform stream.
            j = self._rng.randrange(self.total_seen)
            if j < self.max_size:
                examples[j] = example
        return True

    def add_many(self, rows: Iterable[dict[str, Any]]) -> int:
        count = 0
        for row in rows:
            count += int(
                self.add(
                    prompt=row.get("prompt"),
                    response=row.get("response", row.get("wrong_response")),
                    reward=row.get("reward", 1),
                    metadata=row.get("metadata"),
                )
            )
        return count

    def sample(self, batch_size: int) -> list[FailureExample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        examples = list(self._examples)
        if len(examples) <= batch_size:
            return examples
        if self.sampling == "recent":
            return examples[-batch_size:]
        return self._rng.sample(examples, batch_size)

    def clear(self) -> None:
        """Discard all buffered examples. Call after each SFT update."""
        if self.sampling == "recent":
            self._examples.clear()
        else:
            self._examples = []

    def to_records(self) -> list[dict[str, Any]]:
        return [asdict(example) for example in self._examples]

    def to_parquet(self, path: str) -> None:
        records = []
        for example in self._examples:
            record = asdict(example)
            record.update(record.pop("metadata") or {})
            records.append(record)
        pd.DataFrame(records).to_parquet(path, index=False)
