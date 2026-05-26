# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Trajectory-Anchored Failure-Replay GRPO utilities."""

from verl.experimental.tafr_grpo.config import (
    TAFRGRPOConfig,
    should_checkpoint_and_refresh,
    should_run_failure_sft,
    validate_tafr_config,
)
from verl.experimental.tafr_grpo.failure_data_collector import FailureDataCollector, FailureExample
from verl.experimental.tafr_grpo.tafr_loss import compute_tafr_grpo_auxiliary_loss

__all__ = [
    "FailureDataCollector",
    "FailureExample",
    "TAFRGRPOConfig",
    "compute_tafr_grpo_auxiliary_loss",
    "should_checkpoint_and_refresh",
    "should_run_failure_sft",
    "validate_tafr_config",
]
