# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""JEPA-GRPO single-GPU experimental trainer.

Trains an LLM on two views of each math problem:
  - CoT (Chain-of-Thought) — Dr.GRPO policy optimization
  - Code (Python solution)  — reward-only scoring for JEPA targets

The LeJEPA loss aligns CoT prompt embeddings with correct Code response
embeddings on the unit sphere, while SIGReg enforces N(0, I_d) as the
joint embedding distribution.

Usage:
    python -m verl.experimental.jepa_grpo.main \\
        model_id=Qwen/Qwen2.5-7B-Instruct \\
        alpha=0.1
"""

from __future__ import annotations

import contextlib
import gc
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.experimental.fepo.checkpoint import OptimizerStepCheckpointManager
from verl.experimental.fepo.core_algos import compute_group_advantages
from verl.experimental.fepo.data import EVAL_DATASETS, convert_split_to_verl_rows, load_hf_split
from verl.experimental.fepo.math_parser import compute_math_reward
from verl.experimental.fepo.metrics import _grouped_generation_metrics, evaluate_responses_by_prompt
from verl.experimental.jepa_grpo.core_algos import dr_grpo_loss, lejepa_loss
from verl.utils.tracking import Tracking


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JEPAGRPOConfig:
    # ---- Dataset ----
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    train_dataset: str = "zhuzilin/dapo-math-17k"
    train_dataset_config: Optional[str] = "default"
    train_split: str = "train"
    train_max_samples: int = -1
    eval_datasets: list[str] = field(default_factory=lambda: list(EVAL_DATASETS))
    eval_split: str = "test"
    eval_max_samples: int = -1

    # ---- Output / device ----
    output_dir: str = "checkpoints/jepa_grpo/qwen25_1_5b"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    reference_mode: str = "disabled_adapter"   # disable LoRA to get ref log-probs

    # ---- LoRA (single adapter — not two like FEPO) ----
    lora_adapter_name: str = "policy"
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    # ---- Generation ----
    num_generations: int = 8          # G — same for both CoT and Code views
    train_prompt_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_optimizer_steps: int = 3000
    max_model_len: int = 4096
    max_response_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95

    # ---- Optimizer ----
    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 1.0

    # ---- Dr.GRPO ----
    clip_eps: float = 0.2
    token_log_ratio_clip: float = 8.0

    # ---- JEPA (the research contribution) ----
    alpha: float = 0.1              # JEPA weight in combined loss — THE only tuning knob
    lambda_: float = 0.05           # SIGReg vs align mixing (LeJEPA default arXiv:2511.08544 §6.1)
    M: int = 1024                   # SIGReg random projection directions (LeJEPA default)
    n_freq: int = 17                # Epps-Pulley quadrature points (LeJEPA default)
    t_min: float = -5.0             # Frequency range min (LeJEPA default)
    t_max: float = 5.0              # Frequency range max (LeJEPA default)
    epps_pulley_s: float = 1.0      # Gaussian kernel bandwidth (LeJEPA default)
    embed_micro_batch_size: int = 16
    min_pool_size_for_sigreg: int = 2    # Skip JEPA only if no valid pairs at all
    # EMA target encoder — Code view goes through this (no backward, faster + more stable)
    ema_decay: float = 0.99
    ema_adapter_name: str = "ema_policy"

    # ---- View prompts ----
    cot_system_prompt: str = (
        "You are a careful mathematical problem solver. "
        "Reason step by step and put the final answer in \\boxed{}."
    )
    cot_user_suffix: str = "Solve the problem. Show your reasoning and put the final answer in \\boxed{}."
    code_system_prompt: str = (
        "You are an expert programmer. "
        "Write Python code to solve the math problem. "
        "Print the final answer. Wrap your code in ```python\\n...\\n```."
    )
    code_user_suffix: str = (
        "Solve the problem by writing Python code. "
        "The code should print the final numeric answer. "
        "Wrap your solution in ```python\\n...\\n```."
    )

    # ---- vLLM ----
    generation_backend: str = "vllm_openai"
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_base_model_name: str = "qwen-base"
    vllm_lora_name: str = "train_lora"
    launch_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.45
    vllm_max_num_seqs: int = 32
    vllm_client_concurrency: int = 1
    vllm_attention_backend: str = "FLASHINFER"
    vllm_sleep_enabled: bool = True
    vllm_sleep_level: int = 1
    vllm_sleep_timeout_s: int = 120
    vllm_wake_timeout_s: int = 180
    request_timeout_s: int = 240
    actor_attention_impl: str = "flash_attention_2"
    keep_last_adapters: int = 1

    # ---- Training loop ----
    loss_micro_batch_size: int = 8
    max_token_per_micro_batch: int = 8192
    use_token_based_batching: bool = False
    console_sample_count: int = 3

    # ---- Eval ----
    eval_every_steps: int = 20
    eval_generations: int = 8
    eval_batch_size: int = 4

    # ---- Checkpointing ----
    checkpoint_every_steps: int = 1
    keep_last_checkpoints: int = 3
    resume_mode: str = "auto"
    resume_from_path: Optional[str] = None

    # ---- Tracking ----
    project_name: str = "jepa-grpo"
    experiment_name: str = "qwen25-jepa-grpo-single-gpu"
    logger: list[str] = field(default_factory=lambda: ["console", "wandb"])


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _torch_dtype(dtype: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype.lower()]


def _attach_dr_grpo_advantages(items: list[dict[str, Any]]) -> None:
    """Attach group-centered Dr.GRPO advantages (no std normalization) to CoT items."""
    if not items:
        return
    rewards = [float(item.get("reward", 0.0)) for item in items]
    result = compute_group_advantages(rewards, len(items))

    for idx, item in enumerate(items):
        adv = float(result.solver_advantages[idx].item())
        item["grpo_advantage"] = adv
        item["group_reward_mean"] = float(result.reward_mean[idx].item())
        item["group_reward_std"] = float(result.reward_std[idx].item())

        length = int(item.get("completion_length", 0))
        if length > 0:
            item["token_advantages"] = torch.full((length,), adv, dtype=torch.float32)
            item["token_loss_mask"] = torch.ones((length,), dtype=torch.bool)
            item["valid_token_count"] = length
        else:
            item["token_advantages"] = torch.zeros((0,), dtype=torch.float32)
            item["token_loss_mask"] = torch.zeros((0,), dtype=torch.bool)
            item["valid_token_count"] = 0


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class JEPAGRPOTrainer:
    """Single-GPU Multi-View JEPA + GRPO trainer.

    Generates two views per math problem (CoT + Code), runs Dr.GRPO on the
    CoT view, and adds a LeJEPA representation loss that aligns CoT prompt
    embeddings with correct Code response embeddings on the unit sphere.
    """

    def __init__(self, config: JEPAGRPOConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.adapter_root = self.output_dir / "adapters"
        self.checkpoint_manager = OptimizerStepCheckpointManager(
            self.output_dir / "checkpoints", keep_last=config.keep_last_checkpoints
        )
        self.global_step = 0
        self.train_cursor = 0
        self.vllm_process: Optional[subprocess.Popen] = None
        self.vllm_sleeping = False
        self.vllm_sleep_lock = Lock()

        self.tokenizer = None
        self.policy_model = None
        self.optimizer = None
        self.train_rows: list[dict[str, Any]] = []
        self.eval_rows_by_dataset: dict[str, list[dict[str, Any]]] = {}

        tracking_config = {"trainer": asdict(config)}
        self.logger = Tracking(
            project_name=config.project_name,
            experiment_name=config.experiment_name,
            default_backend=config.logger,
            config=tracking_config,
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "JEPAGRPOTrainer":
        return cls(JEPAGRPOConfig(**mapping))

    # ------------------------------------------------------------------
    # URLs
    # ------------------------------------------------------------------

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.config.vllm_port}/v1"

    @property
    def vllm_server_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.config.vllm_port}"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        self._load_datasets()
        self._load_models()
        if self.config.generation_backend == "vllm_openai" and self.config.launch_vllm:
            self.launch_vllm_server()
        self._maybe_resume()

    def _load_datasets(self) -> None:
        train_split = load_hf_split(
            self.config.train_dataset,
            split=self.config.train_split,
            config=self.config.train_dataset_config,
        )
        self.train_rows = convert_split_to_verl_rows(
            train_split,
            dataset_id=self.config.train_dataset,
            split=self.config.train_split,
            max_samples=self.config.train_max_samples,
        )
        for dataset_id in self.config.eval_datasets:
            split = load_hf_split(dataset_id, split=self.config.eval_split)
            rows = convert_split_to_verl_rows(
                split,
                dataset_id=dataset_id,
                split=self.config.eval_split,
                max_samples=self.config.eval_max_samples,
            )
            self.eval_rows_by_dataset[dataset_id.split("/")[-1].lower()] = rows

    def _load_models(self) -> None:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.config.reference_mode not in {"disabled_adapter", "separate_model"}:
            raise ValueError(
                f"reference_mode must be 'disabled_adapter' or 'separate_model', "
                f"got {self.config.reference_mode!r}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=False,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = _torch_dtype(self.config.dtype)
        self.policy_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            attn_implementation=self.config.actor_attention_impl,
        ).to(self.config.device)
        self.policy_model.config.use_cache = False
        self.policy_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.lora_target_modules),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # Online (policy) adapter — trained via GRPO + JEPA
        self.policy_model = get_peft_model(
            self.policy_model, peft_config, adapter_name=self.config.lora_adapter_name
        )
        if hasattr(self.policy_model, "enable_input_require_grads"):
            self.policy_model.enable_input_require_grads()

        # EMA weights stored as a plain dict on GPU — NOT a second PEFT adapter so vLLM
        # never sees them when we save/load the policy checkpoint.
        policy_params = self.adapter_named_parameters(self.config.lora_adapter_name)
        self.ema_weights: dict[str, torch.Tensor] = {
            n: p.data.clone().detach() for n, p in policy_params.items()
        }

        lora_params = list(self.adapter_named_parameters(self.config.lora_adapter_name).values())
        if not lora_params:
            raise RuntimeError("No LoRA parameters found.")

        optimizer_cls = self._optimizer_cls()
        self.optimizer = optimizer_cls(
            lora_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    def _optimizer_cls(self):
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit
        except Exception:
            return torch.optim.AdamW

    # ------------------------------------------------------------------
    # Adapter management
    # ------------------------------------------------------------------

    def _adapter_marker(self) -> str:
        return f".{self.config.lora_adapter_name}."

    def adapter_named_parameters(self, adapter_name: Optional[str] = None) -> dict[str, torch.nn.Parameter]:
        marker = f".{adapter_name or self.config.lora_adapter_name}."
        return {name: param for name, param in self.policy_model.named_parameters() if marker in name}

    def set_active_adapter(self, train: bool) -> None:
        self.policy_model.set_adapter(self.config.lora_adapter_name)
        self.policy_model.train(mode=train)
        marker = self._adapter_marker()
        for name, param in self.policy_model.named_parameters():
            if marker in name:
                param.requires_grad_(train)

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def prompt_token_ids(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        return ids.tolist() if hasattr(ids, "tolist") else list(ids)

    def encode_prompt_response(
        self, messages: list[dict[str, str]], response_text: str
    ) -> Optional[dict[str, Any]]:
        prompt_ids = self.prompt_token_ids(messages)
        response_ids = self.tokenizer(response_text, add_special_tokens=False)["input_ids"]
        if len(response_ids) == 0:
            return None
        available = self.config.max_model_len - len(prompt_ids)
        if available <= 0:
            return None
        response_ids = response_ids[:available]
        input_ids_list = list(prompt_ids) + list(response_ids)
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.config.device)
        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "completion_start": len(prompt_ids),
            "completion_length": len(response_ids),
        }

    def completion_logprobs(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        completion_start: int,
    ) -> torch.Tensor:
        if input_ids.shape[1] <= completion_start:
            return input_ids.new_zeros((0,), dtype=torch.float32)
        outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        logits = outputs.logits
        shifted_logits = logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        start = max(0, completion_start - 1)
        return token_logps[0, start:].float()

    def batch_completion_logprobs(
        self,
        model: torch.nn.Module,
        encoded_items: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if not encoded_items:
            device = torch.device(self.config.device)
            return (
                torch.zeros((0,), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.bool, device=device),
                [],
            )

        pad_id = int(self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        lengths = [int(item["input_ids"].shape[1]) for item in encoded_items]
        max_len = max(lengths)
        batch_size = len(encoded_items)
        device = torch.device(self.config.device)

        input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        completion_starts = [int(item["completion_start"]) for item in encoded_items]
        completion_lengths = [int(item["completion_length"]) for item in encoded_items]

        for row, item in enumerate(encoded_items):
            row_ids = item["input_ids"].to(device=device).view(-1)
            input_ids[row, : row_ids.numel()] = row_ids
            attention_mask[row, : row_ids.numel()] = 1

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        shifted_logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_logps = F.log_softmax(shifted_logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1).float()

        shifted_positions = torch.arange(max_len - 1, device=device).unsqueeze(0)
        starts = torch.tensor([max(0, s - 1) for s in completion_starts], device=device).unsqueeze(1)
        ends = torch.tensor([max(0, l - 1) for l in lengths], device=device).unsqueeze(1)
        completion_mask = (shifted_positions >= starts) & (shifted_positions < ends)
        completion_mask &= attention_mask[:, 1:].bool()
        return token_logps[completion_mask], completion_mask, completion_lengths

    def _detached_logps(self, encoded: dict[str, Any]) -> torch.Tensor:
        self.set_active_adapter(train=False)
        with torch.no_grad():
            return self.completion_logprobs(
                self.policy_model, encoded["input_ids"], encoded["completion_start"]
            ).cpu()

    def _concat_item_tensors(
        self,
        items: list[dict[str, Any]],
        key: str,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if device is None:
            device = torch.device(self.config.device)
        tensors = [item[key].to(device=device, dtype=dtype) if dtype else item[key].to(device=device) for item in items]
        return torch.cat(tensors, dim=0)

    # ------------------------------------------------------------------
    # Micro-batch helpers
    # ------------------------------------------------------------------

    def _micro_batches(self, items: list[Any]) -> list[list[Any]]:
        if self.config.use_token_based_batching:
            return self._token_budget_micro_batches(items)
        size = max(1, int(self.config.loss_micro_batch_size))
        return [items[i: i + size] for i in range(0, len(items), size)]

    def _token_budget_micro_batches(self, items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        max_tokens = self.config.max_token_per_micro_batch
        batches: list[list] = []
        current: list = []
        current_tokens = 0
        for item in items:
            count = int(item.get("valid_token_count", 0))
            if not count:
                continue
            if current_tokens + count > max_tokens and current:
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += count
        if current:
            batches.append(current)
        return batches or [items[:1]]

    # ------------------------------------------------------------------
    # View message builders
    # ------------------------------------------------------------------

    def cot_messages(self, problem: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.config.cot_system_prompt},
            {"role": "user", "content": f"{problem}\n{self.config.cot_user_suffix}"},
        ]

    def code_messages(self, problem: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.config.code_system_prompt},
            {"role": "user", "content": f"{problem}\n{self.config.code_user_suffix}"},
        ]

    # ------------------------------------------------------------------
    # vLLM management (verbatim from FEPOSingleGPUTrainer)
    # ------------------------------------------------------------------

    def launch_vllm_server(self) -> None:
        if self._vllm_is_ready():
            return
        env = os.environ.copy()
        env["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] = "True"
        if self.config.vllm_sleep_enabled:
            env["VLLM_SERVER_DEV_MODE"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        if self.config.vllm_attention_backend:
            env["VLLM_ATTENTION_BACKEND"] = self.config.vllm_attention_backend

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.config.model_id,
            "--served-model-name", self.config.vllm_base_model_name,
            "--host", self.config.vllm_host,
            "--port", str(self.config.vllm_port),
            "--dtype", "bfloat16" if self.config.dtype in {"bf16", "bfloat16"} else "float16",
            "--max-model-len", str(self.config.max_model_len),
            "--max-num-seqs", str(self.config.vllm_max_num_seqs),
            "--gpu-memory-utilization", str(self.config.vllm_gpu_memory_utilization),
            "--enable-lora",
            "--max-lora-rank", str(self.config.lora_r),
            "--max-loras", "1",
            "--no-enable-log-requests",
        ]
        if self.config.vllm_attention_backend:
            cmd.extend(["--attention-backend", self.config.vllm_attention_backend])
        if self.config.vllm_sleep_enabled:
            cmd.append("--enable-sleep-mode")
        def _pipe_filtered(src, dst):
            for line in src:
                # Suppress uvicorn HTTP access log lines (contain 'HTTP/1.1' with status code).
                # These flood the terminal with every /is_sleeping poll and /v1/completions call.
                if "HTTP/1.1" not in line:
                    dst.write(line)
                    dst.flush()

        self.vllm_process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
        )
        Thread(target=_pipe_filtered, args=(self.vllm_process.stdout, sys.stdout), daemon=True).start()
        Thread(target=_pipe_filtered, args=(self.vllm_process.stderr, sys.stderr), daemon=True).start()
        deadline = time.time() + 600
        while time.time() < deadline:
            if self._vllm_is_ready():
                return
            if self.vllm_process.poll() is not None:
                raise RuntimeError("vLLM exited early — check output above.")
            time.sleep(5)
        raise TimeoutError("vLLM did not become ready within 600s — check output above.")

    def _vllm_is_ready(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.vllm_base_url}/models", timeout=5).status_code == 200
        except Exception:
            return False

    def _post_vllm(self, path: str, payload: dict[str, Any], timeout_s: int):
        import requests
        r = requests.post(f"{self.vllm_base_url}{path}", json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r

    def _post_vllm_server(self, path: str, timeout_s: int, params: Optional[dict[str, Any]] = None):
        import requests
        r = requests.post(f"{self.vllm_server_url}{path}", params=params, timeout=timeout_s)
        r.raise_for_status()
        return r

    def _vllm_is_sleeping(self) -> bool:
        import requests
        r = requests.get(f"{self.vllm_server_url}/is_sleeping", timeout=10)
        r.raise_for_status()
        return bool(r.json().get("is_sleeping", False))

    def _wait_for_vllm_sleep_state(self, sleeping: bool, timeout_s: int) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._vllm_is_sleeping() == sleeping:
                return
            time.sleep(1)
        state = "sleep" if sleeping else "wake"
        raise TimeoutError(f"Timed out waiting for vLLM to {state}.")

    def sleep_vllm_if_enabled(self) -> None:
        if self.config.generation_backend != "vllm_openai" or not self.config.vllm_sleep_enabled:
            return
        with self.vllm_sleep_lock:
            if self.vllm_process is not None and self.vllm_process.poll() is not None:
                return
            if self.vllm_sleeping:
                return
            self._post_vllm_server("/sleep", timeout_s=self.config.vllm_sleep_timeout_s,
                                   params={"level": int(self.config.vllm_sleep_level)})
            self._wait_for_vllm_sleep_state(True, self.config.vllm_sleep_timeout_s)
            self.vllm_sleeping = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def wake_vllm_if_needed(self) -> None:
        if self.config.generation_backend != "vllm_openai" or not self.config.vllm_sleep_enabled:
            return
        with self.vllm_sleep_lock:
            if self.vllm_process is not None and self.vllm_process.poll() is not None:
                raise RuntimeError("vLLM process exited before wake_up.")
            if not self.vllm_sleeping:
                try:
                    self.vllm_sleeping = self._vllm_is_sleeping()
                except Exception:
                    self.vllm_sleeping = False
            if not self.vllm_sleeping:
                return
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._post_vllm_server("/wake_up", timeout_s=self.config.vllm_wake_timeout_s)
            self._wait_for_vllm_sleep_state(False, self.config.vllm_wake_timeout_s)
            self.vllm_sleeping = False

    # ------------------------------------------------------------------
    # Adapter save/reload
    # ------------------------------------------------------------------

    def save_named_lora_adapter(self, adapter_dir: Path) -> Path:
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.set_active_adapter(train=False)
        self.policy_model.save_pretrained(adapter_dir, selected_adapters=[self.config.lora_adapter_name])
        # Flatten nested adapter dir if present (vLLM requires flat layout)
        nested = adapter_dir / self.config.lora_adapter_name
        if nested.exists() and (nested / "adapter_config.json").exists():
            for child in nested.iterdir():
                dest = adapter_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, dest)
                else:
                    shutil.copy2(child, dest)
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Saved adapter is not vLLM-compatible at {adapter_dir}")
        self.tokenizer.save_pretrained(adapter_dir)
        return adapter_dir

    def reload_vllm_lora(self, adapter_dir: Path) -> None:
        self.wake_vllm_if_needed()
        try:
            self._post_vllm("/unload_lora_adapter", {"lora_name": self.config.vllm_lora_name}, timeout_s=120)
        except Exception:
            pass
        self._post_vllm(
            "/load_lora_adapter",
            {"lora_name": self.config.vllm_lora_name, "lora_path": str(adapter_dir)},
            timeout_s=180,
        )

    def save_reload_and_cleanup(self, step_label: Any) -> Path:
        adapter_dir = self.adapter_root / f"adapter_step_{step_label}"
        self.save_named_lora_adapter(adapter_dir)
        if self.config.generation_backend == "vllm_openai":
            self.reload_vllm_lora(adapter_dir)
        self._cleanup_adapters(adapter_dir)
        return adapter_dir

    def _cleanup_adapters(self, current: Path) -> None:
        adapters = sorted(
            [p for p in self.adapter_root.glob("adapter_step_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in adapters[self.config.keep_last_adapters:]:
            if stale.resolve() != current.resolve():
                shutil.rmtree(stale, ignore_errors=True)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict[str, str]],
        n: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        if self.config.generation_backend == "vllm_openai":
            self.wake_vllm_if_needed()
            payload = {
                "model": self.config.vllm_lora_name,
                "messages": messages,
                "n": n,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": self.config.max_response_tokens,
            }
            response = self._post_vllm("/chat/completions", payload, timeout_s=self.config.request_timeout_s).json()
            return [choice["message"]["content"] for choice in response.get("choices", [])]
        return self._generate_with_transformers(messages, n=n, temperature=temperature, top_p=top_p)

    def _generate_with_transformers(
        self,
        messages: list[dict[str, str]],
        n: int,
        temperature: float,
        top_p: float,
    ) -> list[str]:
        self.set_active_adapter(train=False)
        input_ids = torch.tensor(
            [self.prompt_token_ids(messages)], dtype=torch.long, device=self.config.device
        )
        outputs = self.policy_model.generate(
            input_ids=input_ids,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            max_new_tokens=self.config.max_response_tokens,
            num_return_sequences=n,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = input_ids.shape[1]
        return [self.tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]

    # ------------------------------------------------------------------
    # Two-view rollout
    # ------------------------------------------------------------------

    def rollout_prompt_cot(
        self, example: dict[str, Any], prompt_uid: str
    ) -> list[dict[str, Any]]:
        """Generate G CoT responses, score, compute old_logps, attach Dr.GRPO advantages."""
        problem = self._problem_from_example(example)
        messages = self.cot_messages(problem)
        ground_truth = example["reward_model"]["ground_truth"]
        dataset_kind = example["data_source"]

        t0 = time.perf_counter()
        responses = self.generate(messages, self.config.num_generations,
                                  self.config.temperature, self.config.top_p)
        gen_secs = (time.perf_counter() - t0) / max(len(responses), 1)

        items: list[dict[str, Any]] = []
        for gen_idx, response in enumerate(responses):
            encoded = self.encode_prompt_response(messages, response)
            reward_result = compute_math_reward(response, ground_truth, dataset_kind=dataset_kind)
            item: dict[str, Any] = {
                "response": response,
                "generation_index": gen_idx,
                "prompt_uid": prompt_uid,
                "view": "cot",
                "problem": problem,
                "encoded": encoded,
                "completion_length": int(encoded["completion_length"]) if encoded else 0,
                "generation_seconds": gen_secs,
                **reward_result.__dict__,
            }
            if encoded is not None:
                item["old_logps"] = self._detached_logps(encoded)
            else:
                item["old_logps"] = torch.zeros((0,), dtype=torch.float32)
            items.append(item)

        _attach_dr_grpo_advantages(items)
        return items

    def rollout_prompt_code(
        self, example: dict[str, Any], prompt_uid: str
    ) -> list[dict[str, Any]]:
        """Generate G Code responses and score. No GRPO — reward-only for JEPA targets."""
        problem = self._problem_from_example(example)
        messages = self.code_messages(problem)
        ground_truth = example["reward_model"]["ground_truth"]
        dataset_kind = example["data_source"]

        t0 = time.perf_counter()
        responses = self.generate(messages, self.config.num_generations,
                                  self.config.temperature, self.config.top_p)
        gen_secs = (time.perf_counter() - t0) / max(len(responses), 1)

        items: list[dict[str, Any]] = []
        for gen_idx, response in enumerate(responses):
            reward_result = compute_math_reward(response, ground_truth, dataset_kind=dataset_kind)
            items.append({
                "response": response,
                "generation_index": gen_idx,
                "prompt_uid": prompt_uid,
                "view": "code",
                "problem": problem,
                "completion_length": 0,
                "generation_seconds": gen_secs,
                **reward_result.__dict__,
            })
        return items

    def _problem_from_example(self, example: dict[str, Any]) -> str:
        extra = example.get("extra_info", {})
        if "problem" in extra:
            return str(extra["problem"])
        # Fallback: extract from prompt messages
        prompt = example.get("prompt", [])
        if isinstance(prompt, (list, tuple)):
            for msg in reversed(prompt):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", ""))
        return str(prompt)

    # ------------------------------------------------------------------
    # Rollout batch (sequential or concurrent)
    # ------------------------------------------------------------------

    def rollout_prompt_batch(
        self,
        batch: list[dict[str, Any]],
        batch_start: int,
        accumulation_index: int,
    ) -> list[tuple[str, str, list[dict], list[dict]]]:
        """Return list of (prompt_uid, problem, cot_items, code_items) tuples."""
        jobs = []
        for prompt_index, example in enumerate(batch):
            extra = example.get("extra_info", {})
            row_index = extra.get("index", (batch_start + prompt_index) % len(self.train_rows))
            prompt_uid = (
                f"step{self.global_step + 1}:accum{accumulation_index}:"
                f"prompt{prompt_index}:row{row_index}"
            )
            jobs.append((prompt_uid, prompt_index, example, self._problem_from_example(example)))

        def _rollout_one(args):
            prompt_uid, _, example, problem = args
            cot_items = self.rollout_prompt_cot(example, prompt_uid=f"{prompt_uid}:cot")
            code_items = self.rollout_prompt_code(example, prompt_uid=f"{prompt_uid}:code")
            return prompt_uid, problem, cot_items, code_items

        concurrency = max(1, int(self.config.vllm_client_concurrency))
        if self.config.generation_backend != "vllm_openai" or concurrency <= 1 or len(jobs) <= 1:
            return [_rollout_one(j) for j in jobs]

        def _rollout_view(args):
            prompt_uid, prompt_index, example, _, view = args
            if view == "cot":
                return prompt_index, view, self.rollout_prompt_cot(example, prompt_uid=f"{prompt_uid}:cot")
            return prompt_index, view, self.rollout_prompt_code(example, prompt_uid=f"{prompt_uid}:code")

        view_jobs = [(*job, "cot") for job in jobs] + [(*job, "code") for job in jobs]
        cot_by_index: dict[int, list[dict]] = {}
        code_by_index: dict[int, list[dict]] = {}
        max_workers = min(concurrency, len(view_jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for prompt_index, view, items in executor.map(_rollout_view, view_jobs):
                if view == "cot":
                    cot_by_index[prompt_index] = items
                else:
                    code_by_index[prompt_index] = items

        return [
            (
                prompt_uid,
                problem,
                cot_by_index.get(prompt_index, []),
                code_by_index.get(prompt_index, []),
            )
            for prompt_uid, prompt_index, _, problem in jobs
        ]

    # ------------------------------------------------------------------
    # Embedding extraction (core to JEPA)
    # ------------------------------------------------------------------

    def build_cot_prompt_encoded(self, problem: str) -> dict[str, Any]:
        """Encode (question + CoT prompt) with NO response — prompt-only.

        enc_q_cot: the hidden state at the last prompt token, representing
        'what kind of reasoning does the CoT prompt elicit?'
        """
        messages = self.cot_messages(problem)
        prompt_ids = self.prompt_token_ids(messages)
        prompt_ids = prompt_ids[: self.config.max_model_len]
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.config.device)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "seq_length": len(prompt_ids),
        }

    def build_code_response_encoded(
        self, problem: str, response: str
    ) -> Optional[dict[str, Any]]:
        """Encode (question + Code prompt + correct code response) — full sequence.

        enc_a_code: last-token hidden state of the complete correct code solution.
        Returns None if the sequence would exceed max_model_len.
        """
        messages = self.code_messages(problem)
        encoded = self.encode_prompt_response(messages, response)
        if encoded is None:
            return None
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "seq_length": int(encoded["input_ids"].shape[1]),
        }

    @contextlib.contextmanager
    def _ema_forward_ctx(self):
        """Temporarily swap policy LoRA weights with EMA weights for a no-grad forward."""
        policy_params = self.adapter_named_parameters(self.config.lora_adapter_name)
        saved = {n: p.data for n, p in policy_params.items()}
        try:
            for n, p in policy_params.items():
                p.data = self.ema_weights[n]
            yield
        finally:
            for n, p in policy_params.items():
                p.data = saved[n]

    def _last_layer(self) -> torch.nn.Module:
        """Return the last transformer decoder layer for forward-hook attachment."""
        # PeftModel → CausalLM (e.g. Qwen2ForCausalLM) → backbone (e.g. Qwen2Model) → layers
        return self.policy_model.model.model.layers[-1]

    def extract_embeddings_with_hook(
        self,
        encoded_list: list[dict[str, Any]],
        adapter_name: str,
        requires_grad: bool,
        micro_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass → L2-normalized last-token hidden states via last-layer hook.

        Hooks the final decoder layer instead of requesting output_hidden_states=True,
        avoiding materializing all intermediate layer outputs.

        For the online (policy) encoder: requires_grad=True, gradients flow into LoRA.
        For the EMA (target) encoder: requires_grad=False, no backward, no graph built.

        Returns:
            (N, d) float32 tensor on config.device, L2-normalized.
        """
        if not encoded_list:
            d = self.policy_model.config.hidden_size
            return torch.zeros((0, d), dtype=torch.float32, device=self.config.device)

        is_ema = (adapter_name == self.config.ema_adapter_name)
        bs = micro_batch_size or self.config.embed_micro_batch_size
        device = torch.device(self.config.device)
        pad_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        all_embeddings: list[torch.Tensor] = []

        self.set_active_adapter(train=requires_grad)
        last_layer = self._last_layer()

        for start in range(0, len(encoded_list), bs):
            chunk = encoded_list[start: start + bs]
            lengths = [int(item["seq_length"]) for item in chunk]
            max_len = max(lengths)
            B = len(chunk)

            input_ids = torch.full((B, max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
            for row, item in enumerate(chunk):
                row_ids = item["input_ids"].to(device=device).view(-1)
                n = row_ids.numel()
                input_ids[row, :n] = row_ids
                attention_mask[row, :n] = 1

            captured: dict[str, Optional[torch.Tensor]] = {"hidden": None}

            def _hook(module, inp, out, _cap=captured):
                _cap["hidden"] = out[0]  # (B, T, d)

            handle = last_layer.register_forward_hook(_hook)
            try:
                weight_ctx = self._ema_forward_ctx() if is_ema else contextlib.nullcontext()
                grad_ctx = torch.no_grad() if is_ema else contextlib.nullcontext()
                with weight_ctx, grad_ctx:
                    self.policy_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
            finally:
                handle.remove()

            last_hidden = captured["hidden"]  # (B, T, d)
            last_positions = torch.tensor([l - 1 for l in lengths], dtype=torch.long, device=device)
            batch_embs = last_hidden[torch.arange(B, device=device), last_positions, :]  # (B, d)
            all_embeddings.append(batch_embs.float())

        embeddings = torch.cat(all_embeddings, dim=0)  # (N, d)
        return F.normalize(embeddings, dim=-1)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        cot_items_by_prompt: list[list[dict[str, Any]]],    # [prompt][gen]
        code_items_by_prompt: list[list[dict[str, Any]]],   # [prompt][gen]
        problems_by_prompt: list[str],
        grpo_scale: float,
    ) -> dict[str, float]:
        """Accumulate GRPO + JEPA gradients. Does NOT call optimizer.step()."""
        all_metrics: dict[str, float] = {}

        # ================================================================
        # PHASE 1: Dr.GRPO loss on CoT responses (micro-batched)
        # ================================================================
        self.set_active_adapter(train=True)

        flat_cot = [
            item for items in cot_items_by_prompt for item in items
            if item.get("encoded") is not None and item.get("valid_token_count", 0) > 0
        ]

        grpo_accum: dict[str, list[float]] = {}
        for chunk in self._micro_batches(flat_cot):
            encoded_items = [item["encoded"] for item in chunk]
            current_logps, _, _ = self.batch_completion_logprobs(self.policy_model, encoded_items)
            old_logps = self._concat_item_tensors(
                chunk, "old_logps", dtype=current_logps.dtype, device=current_logps.device)
            advantages = self._concat_item_tensors(
                chunk, "token_advantages", dtype=current_logps.dtype, device=current_logps.device)
            loss_mask = self._concat_item_tensors(
                chunk, "token_loss_mask", dtype=torch.bool, device=current_logps.device)

            grpo_loss_val, chunk_metrics = dr_grpo_loss(
                current_logps=current_logps,
                old_logps=old_logps,
                advantages=advantages,
                loss_mask=loss_mask,
                clip_eps=self.config.clip_eps,
                token_log_ratio_clip=self.config.token_log_ratio_clip,
            )
            (grpo_loss_val * grpo_scale).backward()

            for k, v in chunk_metrics.items():
                grpo_accum.setdefault(k, []).append(v)

        all_metrics.update({k: float(sum(vs) / len(vs)) for k, vs in grpo_accum.items() if vs})

        # ================================================================
        # PHASE 2: JEPA loss (embedding alignment + SIGReg)
        # ================================================================
        B = len(problems_by_prompt)

        cot_has_correct = [
            any(float(item.get("reward", 0.0)) >= 1.0 for item in items)
            for items in cot_items_by_prompt
        ]
        code_has_correct = [
            any(float(item.get("reward", 0.0)) >= 1.0 for item in items)
            for items in code_items_by_prompt
        ]
        joint_indices = [i for i in range(B) if cot_has_correct[i] and code_has_correct[i]]

        all_metrics["jepa/frac_cot_has_correct"] = sum(cot_has_correct) / max(B, 1)
        all_metrics["jepa/frac_code_has_correct"] = sum(code_has_correct) / max(B, 1)
        all_metrics["jepa/frac_questions_with_both_correct"] = len(joint_indices) / max(B, 1)

        if not joint_indices:
            # Curriculum: no question has correct in both views → pure GRPO this step
            all_metrics.update({"jepa/skipped": 1.0, "jepa/align_loss": 0.0,
                                 "jepa/sigreg_loss": 0.0, "jepa/lejepa_loss": 0.0,
                                 "jepa/n_valid_pairs": 0.0, "jepa/pool_size": 0.0})
            return all_metrics

        all_metrics["jepa/skipped"] = 0.0

        # Build all encoding inputs.
        # pair_cot_encoded / pair_code_encoded: one per valid joint prompt (aligned).
        # extra_code_encoded: additional correct code responses beyond the first (SIGReg pool extras).
        pair_cot_encoded: list[dict] = []
        pair_code_encoded: list[dict] = []
        extra_code_encoded: list[dict] = []
        for i in joint_indices:
            first_code_enc: Optional[dict] = None
            for item in code_items_by_prompt[i]:
                if float(item.get("reward", 0.0)) >= 1.0:
                    enc = self.build_code_response_encoded(problems_by_prompt[i], item["response"])
                    if enc is None:
                        continue
                    if first_code_enc is None:
                        first_code_enc = enc
                    else:
                        extra_code_encoded.append(enc)
            if first_code_enc is not None:
                pair_cot_encoded.append(self.build_cot_prompt_encoded(problems_by_prompt[i]))
                pair_code_encoded.append(first_code_enc)

        # pool = pair CoT (online) + pair Code + extra Code (all Code via EMA)
        pool_size = len(pair_cot_encoded) + len(pair_code_encoded) + len(extra_code_encoded)
        all_metrics["jepa/pool_size"] = float(pool_size)

        if pool_size < self.config.min_pool_size_for_sigreg or not pair_cot_encoded:
            all_metrics.update({"jepa/skipped": 1.0, "jepa/align_loss": 0.0,
                                 "jepa/sigreg_loss": 0.0, "jepa/lejepa_loss": 0.0,
                                 "jepa/n_valid_pairs": float(len(pair_cot_encoded))})
            return all_metrics

        # Pass 1: CoT prompts — online (policy) adapter, gradients flow for JEPA backprop.
        enc_q_cot = self.extract_embeddings_with_hook(
            pair_cot_encoded,
            adapter_name=self.config.lora_adapter_name,
            requires_grad=True,
        )  # (B_pairs, d)

        # Pass 2: ALL Code sequences — EMA (target) adapter, no grad, no backward.
        # Combining pair + extra in one pass avoids a third forward.
        all_code_list = pair_code_encoded + extra_code_encoded
        all_code_embs = self.extract_embeddings_with_hook(
            all_code_list,
            adapter_name=self.config.ema_adapter_name,
            requires_grad=False,
        )  # (B_code_total, d)

        enc_a_code = all_code_embs[:len(pair_code_encoded)]  # already no-grad from EMA
        # Pool: online CoT (grad flows) + all EMA Code (no grad — target side)
        all_pool_embs = torch.cat([enc_q_cot, all_code_embs], dim=0)  # (pool_size, d)

        jepa_loss_val, jepa_metrics = lejepa_loss(
            enc_q_cot=enc_q_cot,
            enc_a_code=enc_a_code,
            all_embeddings=all_pool_embs,
            lambda_=self.config.lambda_,
            M=self.config.M,
            n_freq=self.config.n_freq,
            t_min=self.config.t_min,
            t_max=self.config.t_max,
            s=self.config.epps_pulley_s,
        )
        (self.config.alpha * jepa_loss_val).backward()

        all_metrics.update(jepa_metrics)
        all_metrics["jepa/n_valid_pairs"] = float(len(pair_cot_encoded))

        return all_metrics

    # ------------------------------------------------------------------
    # Optimizer step
    # ------------------------------------------------------------------

    def optimizer_step(self) -> dict[str, float]:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(self.adapter_named_parameters().values()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # EMA update: ema = decay * ema + (1 - decay) * policy
        decay = self.config.ema_decay
        with torch.no_grad():
            for n, p in self.adapter_named_parameters(self.config.lora_adapter_name).items():
                self.ema_weights[n].mul_(decay).add_(p.data, alpha=1.0 - decay)

        return {"grad_norm": float(grad_norm)}

    # ------------------------------------------------------------------
    # Metrics aggregation
    # ------------------------------------------------------------------

    def _summarize_step(
        self,
        cot_records_by_prompt: list[list[dict[str, Any]]],
        code_records_by_prompt: list[list[dict[str, Any]]],
        step_metrics_list: list[dict[str, float]],
    ) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        # CoT pass@k / avg@k
        cot_groups = [
            [{"reward": float(item.get("reward", 0.0)),
              "response": item.get("response", ""),
              "prediction_normalized": item.get("prediction_normalized"),
              "response_length": float(item.get("completion_length", 0))}
             for item in prompt_items]
            for prompt_items in cot_records_by_prompt
        ]
        if cot_groups:
            metrics.update(_grouped_generation_metrics(cot_groups, prefix="cot"))

        # Code pass@k / avg@k
        code_groups = [
            [{"reward": float(item.get("reward", 0.0)),
              "response": item.get("response", ""),
              "prediction_normalized": item.get("prediction_normalized"),
              "response_length": float(len(str(item.get("response", "")).split()))}
             for item in prompt_items]
            for prompt_items in code_records_by_prompt
        ]
        if code_groups:
            metrics.update(_grouped_generation_metrics(code_groups, prefix="code"))

        # Average GRPO + JEPA metrics over accumulation steps
        all_keys = set(k for d in step_metrics_list for k in d)
        for key in all_keys:
            vals = [d[key] for d in step_metrics_list if key in d]
            if vals:
                metrics[key] = float(sum(vals) / len(vals))

        return metrics

    # ------------------------------------------------------------------
    # Print samples
    # ------------------------------------------------------------------

    def _response_preview(self, text: Any, limit: int = 180) -> str:
        preview = " ".join(str(text).split())
        return preview[:limit - 3] + "..." if len(preview) > limit else preview

    def _print_step_samples(
        self,
        step: int,
        rollouts: list[tuple[str, str, list[dict], list[dict]]],
    ) -> None:
        limit = max(0, int(self.config.console_sample_count))
        if limit <= 0:
            return
        printed = 0
        for prompt_uid, problem, cot_items, code_items in rollouts:
            if not cot_items:
                continue
            first_cot = cot_items[0]
            cot_acc = sum(float(i.get("reward", 0.0)) for i in cot_items) / max(len(cot_items), 1)
            code_acc = sum(float(i.get("reward", 0.0)) for i in code_items) / max(len(code_items), 1)
            print(
                f"sample step={step} "
                f"cot_acc={cot_acc:.3f} code_acc={code_acc:.3f} "
                f"gt={first_cot.get('ground_truth_normalized')} "
                f"pred={first_cot.get('prediction_normalized')} "
                f"response={self._response_preview(first_cot.get('response', ''))}",
                flush=True,
            )
            printed += 1
            if printed >= limit:
                break

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        if not self.train_rows:
            raise RuntimeError("Call setup() before train().")

        self.save_reload_and_cleanup(self.global_step if self.global_step > 0 else "initial")
        cursor = self.train_cursor

        while self.global_step < self.config.max_optimizer_steps:
            step_start = time.perf_counter()
            rollout_wall = 0.0
            loss_wall = 0.0

            # Collect rollouts for all gradient accumulation steps
            accumulation_rollouts: list[list[tuple]] = []

            for accum_idx in range(self.config.gradient_accumulation_steps):
                batch_start = cursor
                batch = [
                    self.train_rows[(cursor + offset) % len(self.train_rows)]
                    for offset in range(self.config.train_prompt_batch_size)
                ]
                cursor += self.config.train_prompt_batch_size

                t0 = time.perf_counter()
                rollouts = self.rollout_prompt_batch(batch, batch_start, accum_idx)
                rollout_wall += time.perf_counter() - t0

                accumulation_rollouts.append(rollouts)
                self._print_step_samples(self.global_step + 1, rollouts)

            self.sleep_vllm_if_enabled()

            # Count active CoT responses for GRPO loss scaling
            total_active = sum(
                item.get("valid_token_count", 0) > 0 and item.get("encoded") is not None
                for rollouts in accumulation_rollouts
                for _, _, cot_items, _ in rollouts
                for item in cot_items
            )
            grpo_scale = 1.0 / float(self.config.gradient_accumulation_steps * max(total_active, 1))

            # Training forward passes (gradient accumulation)
            all_cot_records: list[list[dict]] = []
            all_code_records: list[list[dict]] = []
            step_metrics_list: list[dict[str, float]] = []

            t0 = time.perf_counter()
            for rollouts in accumulation_rollouts:
                cot_by_prompt = [cot_items for _, _, cot_items, _ in rollouts]
                code_by_prompt = [code_items for _, _, _, code_items in rollouts]
                problems = [problem for _, problem, _, _ in rollouts]

                step_metrics = self.train_step(
                    cot_items_by_prompt=cot_by_prompt,
                    code_items_by_prompt=code_by_prompt,
                    problems_by_prompt=problems,
                    grpo_scale=grpo_scale,
                )
                step_metrics_list.append(step_metrics)
                all_cot_records.extend(cot_by_prompt)
                all_code_records.extend(code_by_prompt)
            loss_wall = time.perf_counter() - t0

            self.global_step += 1
            self.train_cursor = cursor

            grad_metrics = self.optimizer_step()
            self.wake_vllm_if_needed()
            adapter_dir = self.save_reload_and_cleanup(self.global_step)

            # Aggregate and log
            metrics = self._summarize_step(all_cot_records, all_code_records, step_metrics_list)
            metrics.update(grad_metrics)
            step_secs = time.perf_counter() - step_start
            total_responses = sum(len(items) for items in all_cot_records) + sum(len(items) for items in all_code_records)
            metrics.update({
                "checkpoint/adapter": str(adapter_dir),
                "perf/step_seconds": float(step_secs),
                "perf/rollout_wall_seconds": float(rollout_wall),
                "perf/loss_wall_seconds": float(loss_wall),
                "perf/responses_per_second": float(total_responses / step_secs) if step_secs > 0 else 0.0,
            })
            self.logger.log(metrics, step=self.global_step)

            if self.config.checkpoint_every_steps > 0 and self.global_step % self.config.checkpoint_every_steps == 0:
                self._save_checkpoint(cursor, adapter_dir)

            if self.config.eval_every_steps > 0 and self.global_step % self.config.eval_every_steps == 0:
                self.evaluate()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict[str, dict[str, Any]]:
        all_metrics = {}
        for dataset_name, rows in self.eval_rows_by_dataset.items():
            responses_by_prompt = []
            for start in range(0, len(rows), self.config.eval_batch_size):
                chunk = rows[start: start + self.config.eval_batch_size]

                def _generate_eval(example):
                    problem = self._problem_from_example(example)
                    messages = self.cot_messages(problem)
                    return self.generate(
                        messages,
                        n=self.config.eval_generations,
                        temperature=self.config.eval_temperature,
                        top_p=self.config.eval_top_p,
                    )

                concurrency = max(1, int(self.config.vllm_client_concurrency))
                if self.config.generation_backend == "vllm_openai" and concurrency > 1 and len(chunk) > 1:
                    max_workers = min(concurrency, len(chunk))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        responses_by_prompt.extend(executor.map(_generate_eval, chunk))
                else:
                    for example in chunk:
                        responses_by_prompt.append(_generate_eval(example))

            metrics = evaluate_responses_by_prompt(
                rows, responses_by_prompt, dataset_name, self.config.eval_generations
            )
            self.logger.log(
                {f"val/{dataset_name}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))},
                step=self.global_step,
            )
            all_metrics[dataset_name] = metrics
        return all_metrics

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, cursor: int, adapter_dir: Path) -> None:
        self.checkpoint_manager.save(
            global_step=self.global_step,
            payload={"cursor": cursor, "adapter_path": str(adapter_dir)},
            model_state={
                "policy": {k: v.detach().cpu() for k, v in self.adapter_named_parameters().items()}
            },
            optimizer_state={"policy": self.optimizer.state_dict()},
            dataloader_state={"cursor": cursor},
        )

    def _maybe_resume(self) -> None:
        if self.config.resume_mode == "disable":
            return
        checkpoint_dir = self.config.resume_from_path
        if checkpoint_dir is None and self.config.resume_mode == "auto":
            latest = self.checkpoint_manager.latest_checkpoint_dir()
            checkpoint_dir = str(latest) if latest is not None else None
        if checkpoint_dir is None:
            return
        state = self.checkpoint_manager.load(checkpoint_dir)
        self.global_step = int(state["global_step"])
        self.train_cursor = int(
            state.get("dataloader_state", {}).get("cursor",
            state.get("payload", {}).get("cursor", 0))
        )
        for name, tensor in state.get("model_state", {}).get("policy", {}).items():
            self.adapter_named_parameters()[name].data.copy_(tensor.to(self.config.device))
        if state.get("optimizer_state", {}).get("policy"):
            self.optimizer.load_state_dict(state["optimizer_state"]["policy"])

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self.vllm_process is not None and self.vllm_process.poll() is None:
            try:
                os.killpg(self.vllm_process.pid, 15)
            except ProcessLookupError:
                return
            try:
                self.vllm_process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.vllm_process.pid, 9)
                self.vllm_process.wait(timeout=20)
