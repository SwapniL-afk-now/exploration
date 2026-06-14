# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""

import datetime
import logging
import numbers
import os
import pprint

import torch


def concat_dict_to_str(dict: dict, step):
    if os.environ.get("VERL_CONSOLE_FULL_METRICS", "0") == "1":
        output = [f"step:{step}"]
        for k, v in dict.items():
            if isinstance(v, numbers.Number):
                output.append(f"{k}:{pprint.pformat(v)}")
        return " - ".join(output)

    preferred = [
        "actor/loss",
        "actor/pg_loss",
        "actor/kl_loss",
        "actor/grad_norm",
        "actor/entropy",
        "fepo/solver_loss",
        "fepo/solver_pg_loss",
        "fepo/failure_sft_loss",
        "solver_grad_norm",
        "failure_grad_norm",
        "train/accuracy",
        "train/failure_rate",
        "train/pass_at_1",
        "train/pass_at_8",
        "train/unique_answer_ratio_at_k",
        "train/exploration_collapse_rate",
        "tafr_grpo/loss_total",
        "tafr_grpo/loss_grpo",
        "tafr_grpo/kl_anchor",
        "tafr_grpo/kl_replay",
        "tafr_grpo/replay_gate_mean",
        "tafr_grpo/mean_group_reward",
        "tafr_grpo/failure_dataset_size",
        "tafr_grpo/did_sft_update_this_step",
        "tafr_grpo/did_checkpoint_save_this_step",
        "tafr_grpo/failure_sft_loss",
        "val/amc23/pass_at_1",
        "val/amc23/pass_at_8",
        "val/amc23/avg_at_k",
        "val/amc23/avg_at_8",
        "perf/time_per_step",
        "perf/step_seconds",
        "perf/responses_per_second",
        "timing_s/step",
        # JEPA-GRPO
        "grpo_pg_loss",
        "grpo_clip_fraction",
        "grad_norm",
        "jepa/lejepa_loss",
        "jepa/align_loss",
        "jepa/sigreg_loss",
        "jepa/n_valid_pairs",
        "jepa/frac_questions_with_both_correct",
        "cot/pass_at_1",
        "cot/avg_reward",
        "code/pass_at_1",
        "code/avg_reward",
    ]
    output = [f"step:{step}"]
    seen = set()
    for k in preferred:
        if k in dict and isinstance(dict[k], numbers.Number):
            output.append(f"{k}:{float(dict[k]):.4g}")
            seen.add(k)

    val_suffixes = {"pass_at_1", "pass_at_8", "avg_at_k", "avg_at_8"}
    for k in sorted(dict):
        parts = k.split("/")
        if len(parts) == 3 and parts[0] == "val" and parts[2] in val_suffixes and k not in seen:
            v = dict[k]
            if isinstance(v, numbers.Number):
                output.append(f"{k}:{float(v):.4g}")
                seen.add(k)

    if len(output) == 1:
        for k, v in dict.items():
            if isinstance(v, numbers.Number):
                output.append(f"{k}:{float(v):.4g}")
                if len(output) >= 12:
                    break
    return " - ".join(output)


class LocalLogger:
    """
    A local logger that logs messages to the console.

    Args:
        print_to_console (bool): Whether to print to the console.
    """

    def __init__(self, print_to_console=True):
        self.print_to_console = print_to_console

    def flush(self):
        pass

    def log(self, data, step):
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)


class DecoratorLoggerBase:
    """
    Base class for all decorators that log messages.

    Args:
        role (str): The role (the name) of the logger.
        logger (logging.Logger): The logger instance to use for logging.
        level (int): The logging level.
        rank (int): The rank of the process.
        log_only_rank_0 (bool): If True, only log for rank 0.
    """

    def __init__(
        self, role: str, logger: logging.Logger = None, level=logging.DEBUG, rank: int = 0, log_only_rank_0: bool = True
    ):
        self.role = role
        self.logger = logger
        self.level = level
        self.rank = rank
        self.log_only_rank_0 = log_only_rank_0
        self.logging_function = self.log_by_logging
        if logger is None:
            self.logging_function = self.log_by_print

    def log_by_print(self, log_str):
        if not self.log_only_rank_0 or self.rank == 0:
            print(f"{self.role} {log_str}", flush=True)

    def log_by_logging(self, log_str):
        if self.logger is None:
            raise ValueError("Logger is not initialized")
        if not self.log_only_rank_0 or self.rank == 0:
            self.logger.log(self.level, f"{self.role} {log_str}")


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def print_with_rank(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        print(f"[Rank {rank}] {message}", flush=True)


def print_with_rank_and_timer(message: str, rank: int = 0, log_only_rank_0: bool = False):
    """_summary_
    Print a message with rank information and a timestamp.
    This function prints the message only if `log_only_rank_0` is False or if the rank is 0.

    Args:
        message (str): _description_
        rank (int, optional): _description_. Defaults to 0.
        log_only_rank_0 (bool, optional): _description_. Defaults to False.
    """
    now = datetime.datetime.now()
    message = f"[{now.strftime('%Y-%m-%d %H:%M:%S')}] [Rank {rank}] {message}"
    if not log_only_rank_0 or rank == 0:
        print(message, flush=True)


def log_with_rank(message: str, rank, logger: logging.Logger, level=logging.INFO, log_only_rank_0: bool = False):
    """_summary_
    Log a message with rank information using a logger.
    This function logs the message only if `log_only_rank_0` is False or if the rank is 0.
    Args:
        message (str): The message to log.
        rank (int): The rank of the process.
        logger (logging.Logger): The logger instance to use for logging.
        level (int, optional): The logging level. Defaults to logging.INFO.
        log_only_rank_0 (bool, optional): If True, only log for rank 0. Defaults to False.
    """
    if not log_only_rank_0 or rank == 0:
        logger.log(level, f"[Rank {rank}] {message}")
