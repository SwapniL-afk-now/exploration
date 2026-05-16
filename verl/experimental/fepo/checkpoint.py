# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


@dataclass(frozen=True)
class FEPOCheckpoint:
    path: Path
    global_step: int
    payload: dict[str, Any]


class OptimizerStepCheckpointManager:
    """Small optimizer-step checkpoint manager for experimental FEPO runs."""

    def __init__(self, root_dir: str | os.PathLike[str], keep_last: Optional[int] = 3):
        self.root_dir = Path(root_dir)
        self.keep_last = keep_last
        self.latest_path = self.root_dir / "latest_checkpoint.json"

    def _step_dir(self, global_step: int) -> Path:
        return self.root_dir / f"global_step_{int(global_step)}"

    def save(
        self,
        *,
        global_step: int,
        payload: dict[str, Any],
        model_state: Optional[dict[str, Any]] = None,
        optimizer_state: Optional[dict[str, Any]] = None,
        dataloader_state: Optional[dict[str, Any]] = None,
    ) -> FEPOCheckpoint:
        step_dir = self._step_dir(global_step)
        step_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "global_step": int(global_step),
            "payload": payload,
            "model_state": model_state or {},
            "optimizer_state": optimizer_state or {},
            "dataloader_state": dataloader_state or {},
            "rng_state": capture_rng_state(),
        }
        state_path = step_dir / "state.pt"
        torch.save(state, state_path)

        pointer = {
            "global_step": int(global_step),
            "checkpoint_path": str(step_dir),
            "state_path": str(state_path),
        }
        tmp_path = self.latest_path.with_suffix(".tmp")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
        tmp_path.replace(self.latest_path)

        self._cleanup_old_checkpoints()
        return FEPOCheckpoint(path=step_dir, global_step=int(global_step), payload=payload)

    def latest_checkpoint_dir(self) -> Optional[Path]:
        if not self.latest_path.exists():
            return None
        payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        path = Path(payload["checkpoint_path"])
        return path if path.exists() else None

    def load(self, checkpoint_dir: Optional[str | os.PathLike[str]] = None) -> dict[str, Any]:
        path = Path(checkpoint_dir) if checkpoint_dir is not None else self.latest_checkpoint_dir()
        if path is None:
            raise FileNotFoundError(f"No FEPO checkpoint found under {self.root_dir}")
        state_path = path / "state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing FEPO checkpoint state: {state_path}")
        state = torch.load(state_path, weights_only=False)
        if "rng_state" in state:
            restore_rng_state(state["rng_state"])
        return state

    def _cleanup_old_checkpoints(self) -> None:
        if self.keep_last is None or self.keep_last <= 0:
            return
        checkpoints = sorted(
            [p for p in self.root_dir.glob("global_step_*") if p.is_dir()],
            key=lambda p: int(p.name.rsplit("_", 1)[-1]),
            reverse=True,
        )
        for stale in checkpoints[self.keep_last :]:
            for child in stale.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted([p for p in stale.rglob("*") if p.is_dir()], reverse=True):
                child.rmdir()
            stale.rmdir()
