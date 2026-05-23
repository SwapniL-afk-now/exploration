# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from verl.experimental.fepo.checkpoint import OptimizerStepCheckpointManager
from verl.experimental.fepo.core_algos import (
    attach_fepo_dr_grpo_group_advantages,
    token_failure_sft_loss,
    token_solver_fepo_loss,
)
from verl.experimental.fepo.data import EVAL_DATASETS, convert_split_to_verl_rows, load_hf_split
from verl.experimental.fepo.math_parser import compute_math_reward
from verl.experimental.fepo.metrics import evaluate_responses_by_prompt, summarize_training_records
from verl.utils.tracking import Tracking


@dataclass
class FEPOSingleGPUConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    train_dataset: str = "zhuzilin/dapo-math-17k"
    train_dataset_config: Optional[str] = "default"
    train_split: str = "train"
    train_max_samples: int = -1
    eval_datasets: list[str] = field(default_factory=lambda: list(EVAL_DATASETS))
    eval_split: str = "test"
    eval_max_samples: int = -1

    output_dir: str = "checkpoints/fepo/qwen25_1_5b"
    seed: int = 42
    device: str = "cuda"
    dtype: str = "bfloat16"
    reference_mode: str = "disabled_adapter"

    solver_adapter_name: str = "solver"
    failure_adapter_name: str = "failure"
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    num_generations: int = 8
    train_prompt_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_optimizer_steps: int = 3000
    max_model_len: int = 4096
    max_response_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 0.95
    eval_temperature: float = 1.0
    eval_top_p: float = 0.95

    learning_rate: float = 5e-6
    failure_learning_rate: float = 1.25e-6
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.99
    max_grad_norm: float = 1.0
    clip_eps: float = 0.2
    fepo_escape_alpha: float = 0.01
    fepo_reference_beta: float = 0.03
    token_log_ratio_clip: float = 8.0

    eval_every_steps: int = 20
    eval_generations: int = 8
    eval_batch_size: int = 4

    generation_backend: str = "vllm_openai"
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_base_model_name: str = "qwen-base"
    vllm_lora_name: str = "train_lora"
    launch_vllm: bool = True
    vllm_gpu_memory_utilization: float = 0.35
    vllm_max_num_seqs: int = 16
    vllm_client_concurrency: int = 1
    loss_micro_batch_size: int = 8
    max_token_per_micro_batch: int = 8192  # Token budget for dynamic batching
    use_token_based_batching: bool = False  # Toggle between fixed sample vs token-based batching
    console_sample_count: int = 3
    actor_attention_impl: str = "flash_attention_2"
    vllm_attention_backend: str = "FLASHINFER"
    vllm_sleep_enabled: bool = True
    vllm_sleep_level: int = 1
    vllm_sleep_timeout_s: int = 120
    vllm_wake_timeout_s: int = 180
    request_timeout_s: int = 240
    keep_last_adapters: int = 1

    checkpoint_every_steps: int = 1
    keep_last_checkpoints: int = 3
    resume_mode: str = "auto"
    resume_from_path: Optional[str] = None

    project_name: str = "failure-escape"
    experiment_name: str = "qwen25-fepo-single-gpu"
    logger: list[str] = field(default_factory=lambda: ["console", "wandb"])


def _torch_dtype(dtype: str):
    return {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }[dtype.lower()]


class FEPOSingleGPUTrainer:
    """Single-GPU experimental FEPO trainer.

    This trainer is deliberately separate from the production Ray PPO trainer.
    It ports the Kaggle notebook behavior with named solver/failure LoRA
    adapters, vLLM OpenAI-compatible generation, W&B via ``Tracking``, and
    optimizer-step checkpointing.
    """

    def __init__(self, config: FEPOSingleGPUConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.adapter_root = self.output_dir / "solver_adapters"
        self.failure_adapter_root = self.output_dir / "failure_adapters"
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
        self.reference_model = None
        self.solver_optimizer = None
        self.failure_optimizer = None
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
    def from_mapping(cls, mapping: dict[str, Any]) -> FEPOSingleGPUTrainer:
        return cls(FEPOSingleGPUConfig(**mapping))

    @property
    def vllm_base_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.config.vllm_port}/v1"

    @property
    def vllm_server_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.config.vllm_port}"

    def setup(self) -> None:
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_root.mkdir(parents=True, exist_ok=True)
        self.failure_adapter_root.mkdir(parents=True, exist_ok=True)
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
                "reference_mode must be 'disabled_adapter' or 'separate_model', "
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
        self.policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=list(self.config.lora_target_modules),
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.policy_model = get_peft_model(
            self.policy_model, peft_config, adapter_name=self.config.solver_adapter_name
        )
        self.policy_model.add_adapter(self.config.failure_adapter_name, peft_config)
        if hasattr(self.policy_model, "enable_input_require_grads"):
            self.policy_model.enable_input_require_grads()

        if self.config.reference_mode == "separate_model":
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                attn_implementation=self.config.actor_attention_impl,
            ).to(self.config.device)
            self.reference_model.config.use_cache = False
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad_(False)

        solver_params = list(self.adapter_named_parameters(self.config.solver_adapter_name).values())
        failure_params = list(self.adapter_named_parameters(self.config.failure_adapter_name).values())
        if not solver_params or not failure_params:
            raise RuntimeError("FEPO requires both solver and failure LoRA adapter parameters.")

        optimizer_cls = self._optimizer_cls()
        self.solver_optimizer = optimizer_cls(
            solver_params,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )
        self.failure_optimizer = optimizer_cls(
            failure_params,
            lr=self.config.failure_learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            weight_decay=self.config.weight_decay,
        )

    def _optimizer_cls(self):
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit
        except Exception:
            return torch.optim.AdamW

    def _adapter_marker(self, adapter_name: str) -> str:
        return f".{adapter_name}."

    def adapter_named_parameters(self, adapter_name: str) -> dict[str, torch.nn.Parameter]:
        marker = self._adapter_marker(adapter_name)
        return {name: param for name, param in self.policy_model.named_parameters() if marker in name}

    def set_active_adapter(self, adapter_name: str, train: bool) -> None:
        self.policy_model.set_adapter(adapter_name)
        self.policy_model.train(mode=train)
        for name, param in self.policy_model.named_parameters():
            is_selected = self._adapter_marker(adapter_name) in name
            is_any_lora = (
                self._adapter_marker(self.config.solver_adapter_name) in name
                or self._adapter_marker(self.config.failure_adapter_name) in name
            )
            if is_any_lora:
                param.requires_grad_(bool(train and is_selected))

    def prompt_token_ids(self, messages: list[dict[str, str]]) -> list[int]:
        ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors=None,
        )
        return ids.tolist() if hasattr(ids, "tolist") else list(ids)

    def encode_prompt_response(self, messages: list[dict[str, str]], response_text: str) -> Optional[dict[str, Any]]:
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
            "attention_mask": torch.ones_like(input_ids, device=self.config.device),
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
            return torch.zeros((0,), dtype=torch.float32, device=device), torch.zeros((0,), dtype=torch.bool, device=device), []

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
        starts = torch.tensor([max(0, start - 1) for start in completion_starts], device=device).unsqueeze(1)
        ends = torch.tensor([max(0, length - 1) for length in lengths], device=device).unsqueeze(1)
        completion_mask = (shifted_positions >= starts) & (shifted_positions < ends)
        completion_mask &= attention_mask[:, 1:].bool()
        return token_logps[completion_mask], completion_mask, completion_lengths

    def _concat_item_tensors(
        self,
        items: list[dict[str, Any]],
        key: str,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        tensors = [item[key] for item in items]
        if device is None:
            device = torch.device(self.config.device)
        if dtype is None:
            return torch.cat([tensor.to(device=device) for tensor in tensors], dim=0)
        return torch.cat([tensor.to(device=device, dtype=dtype) for tensor in tensors], dim=0)

    def _compute_token_budget_batches(
        self,
        items: list[dict[str, Any]],
        valid_indices: list[int],
    ) -> list[list[int]]:
        """Split items by token budget instead of fixed sample count.
        
        Returns list of index lists, each batch under max_token_per_micro_batch tokens.
        """
        if not self.config.use_token_based_batching:
            # Fall back to fixed sample count batching
            micro_batch_size = max(1, int(self.config.loss_micro_batch_size))
            return [valid_indices[i:i + micro_batch_size] for i in range(0, len(valid_indices), micro_batch_size)]
        
        max_tokens = self.config.max_token_per_micro_batch
        batches = []
        current_batch = []
        current_tokens = 0
        
        for idx in valid_indices:
            item = items[idx]
            token_count = int(item.get("valid_token_count", 0))
            if not token_count:
                continue
                
            # If adding this item exceeds budget and we have items, start new batch
            if current_tokens + token_count > max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            
            current_batch.append(idx)
            current_tokens += token_count
        
        if current_batch:
            batches.append(current_batch)
        
        return batches if batches else [valid_indices[:1]]  # Ensure at least one batch

    def detached_adapter_completion_logprobs(self, adapter_name: str, encoded: dict[str, Any]) -> torch.Tensor:
        self.set_active_adapter(adapter_name, train=False)
        with torch.no_grad():
            return self.completion_logprobs(self.policy_model, encoded["input_ids"], encoded["completion_start"]).cpu()

    def detached_reference_completion_logprobs(self, encoded: dict[str, Any]) -> torch.Tensor:
        if self.config.reference_mode == "disabled_adapter":
            self.policy_model.eval()
            with torch.no_grad(), self.policy_model.disable_adapter():
                return self.completion_logprobs(
                    self.policy_model,
                    encoded["input_ids"],
                    encoded["completion_start"],
                ).cpu()
        if self.reference_model is None:
            raise RuntimeError("reference_model is not initialized; set reference_mode='disabled_adapter' or load one.")
        self.reference_model.eval()
        with torch.no_grad():
            return self.completion_logprobs(
                self.reference_model, encoded["input_ids"], encoded["completion_start"]
            ).cpu()

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
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model_id,
            "--served-model-name",
            self.config.vllm_base_model_name,
            "--host",
            self.config.vllm_host,
            "--port",
            str(self.config.vllm_port),
            "--dtype",
            "bfloat16" if self.config.dtype in {"bf16", "bfloat16"} else "float16",
            "--max-model-len",
            str(self.config.max_model_len),
            "--max-num-seqs",
            str(self.config.vllm_max_num_seqs),
            "--gpu-memory-utilization",
            str(self.config.vllm_gpu_memory_utilization),
            "--enable-lora",
            "--max-lora-rank",
            str(self.config.lora_r),
            "--max-loras",
            "1",
            "--disable-log-requests",
        ]
        if self.config.vllm_sleep_enabled:
            cmd.append("--enable-sleep-mode")
        log_path = self.output_dir / "vllm_server.log"
        log_file = open(log_path, "a", buffering=1)
        self.vllm_process = subprocess.Popen(
            cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT, text=True, start_new_session=True
        )
        deadline = time.time() + 600
        while time.time() < deadline:
            if self._vllm_is_ready():
                return
            if self.vllm_process.poll() is not None:
                raise RuntimeError(f"vLLM exited early. Log tail:\n{log_path.read_text(errors='ignore')[-4000:]}")
            time.sleep(5)
        raise TimeoutError(f"vLLM did not become ready. Log tail:\n{log_path.read_text(errors='ignore')[-4000:]}")

    def _vllm_is_ready(self) -> bool:
        try:
            import requests

            response = requests.get(f"{self.vllm_base_url}/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _post_vllm(self, path: str, payload: dict[str, Any], timeout_s: int):
        import requests

        response = requests.post(f"{self.vllm_base_url}{path}", json=payload, timeout=timeout_s)
        response.raise_for_status()
        return response

    def _post_vllm_server(self, path: str, timeout_s: int, params: Optional[dict[str, Any]] = None):
        import requests

        response = requests.post(f"{self.vllm_server_url}{path}", params=params, timeout=timeout_s)
        response.raise_for_status()
        return response

    def _vllm_is_sleeping(self) -> bool:
        import requests

        response = requests.get(f"{self.vllm_server_url}/is_sleeping", timeout=10)
        response.raise_for_status()
        return bool(response.json().get("is_sleeping", False))

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
            self._post_vllm_server(
                "/sleep",
                timeout_s=self.config.vllm_sleep_timeout_s,
                params={"level": int(self.config.vllm_sleep_level)},
            )
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

    def save_named_lora_adapter(self, adapter_name: str, adapter_dir: Path) -> Path:
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        self.set_active_adapter(adapter_name, train=False)
        self.policy_model.save_pretrained(adapter_dir, selected_adapters=[adapter_name])

        nested_adapter_dir = adapter_dir / adapter_name
        if nested_adapter_dir.exists() and (nested_adapter_dir / "adapter_config.json").exists():
            for child in nested_adapter_dir.iterdir():
                destination = adapter_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, destination)
                else:
                    shutil.copy2(child, destination)
        if not (adapter_dir / "adapter_config.json").exists():
            raise FileNotFoundError(f"Saved adapter {adapter_name!r} is not vLLM-compatible at {adapter_dir}")
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

    def save_reload_and_cleanup(self, step_label: Any) -> tuple[Path, Path]:
        solver_dir = self.save_named_lora_adapter(
            self.config.solver_adapter_name,
            self.adapter_root / f"adapter_step_{step_label}",
        )
        failure_dir = self.save_named_lora_adapter(
            self.config.failure_adapter_name, self.failure_adapter_root / f"adapter_step_{step_label}"
        )
        if self.config.generation_backend == "vllm_openai":
            self.reload_vllm_lora(solver_dir)
        self._cleanup_adapters(self.adapter_root, solver_dir)
        self._cleanup_adapters(self.failure_adapter_root, failure_dir)
        return solver_dir, failure_dir

    def _cleanup_adapters(self, root: Path, current: Path) -> None:
        adapters = sorted(
            [p for p in root.glob("adapter_step_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for stale in adapters[self.config.keep_last_adapters :]:
            if stale.resolve() != current.resolve():
                shutil.rmtree(stale, ignore_errors=True)

    def generate(self, messages: list[dict[str, str]], n: int, temperature: float, top_p: float) -> list[str]:
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
        self.set_active_adapter(self.config.solver_adapter_name, train=False)
        input_ids = torch.tensor([self.prompt_token_ids(messages)], dtype=torch.long, device=self.config.device)
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
        return [self.tokenizer.decode(output[prompt_len:], skip_special_tokens=True) for output in outputs]

    def rollout_prompt(self, example: dict[str, Any], prompt_uid: Optional[str] = None) -> list[dict[str, Any]]:
        messages = example["prompt"]
        generation_start = time.perf_counter()
        outputs = self.generate(messages, self.config.num_generations, self.config.temperature, self.config.top_p)
        generation_seconds = time.perf_counter() - generation_start
        generation_seconds_per_response = generation_seconds / max(len(outputs), 1)
        items: list[dict[str, Any]] = []
        for generation_index, response in enumerate(outputs):
            encoded = self.encode_prompt_response(messages, response)
            reward = compute_math_reward(
                response,
                example["reward_model"]["ground_truth"],
                dataset_kind=example["data_source"],
            )
            item: dict[str, Any] = {
                "response": response,
                "generation_index": generation_index,
                "prompt_uid": prompt_uid,
                "encoded": encoded,
                "completion_length": int(encoded["completion_length"]) if encoded is not None else 0,
                "generation_seconds": generation_seconds_per_response,
                **reward.__dict__,
            }
            if encoded is not None:
                item["old_solver_logps"] = self.detached_adapter_completion_logprobs(
                    self.config.solver_adapter_name, encoded
                )
                item["old_failure_logps"] = self.detached_adapter_completion_logprobs(
                    self.config.failure_adapter_name, encoded
                )
                item["failure_logps_for_escape"] = item["old_failure_logps"].clone()
                item["ref_logps"] = self.detached_reference_completion_logprobs(encoded)
            else:
                item["old_solver_logps"] = torch.zeros((0,), dtype=torch.float32)
                item["old_failure_logps"] = torch.zeros((0,), dtype=torch.float32)
                item["failure_logps_for_escape"] = torch.zeros((0,), dtype=torch.float32)
                item["ref_logps"] = torch.zeros((0,), dtype=torch.float32)
            items.append(item)
        attach_fepo_dr_grpo_group_advantages(items)
        for item in items:
            token_loss_mask = item.get("token_loss_mask")
            valid_token_count = int(token_loss_mask.sum().item()) if token_loss_mask is not None else 0
            item["valid_token_count"] = valid_token_count
            item["ignored_token_count"] = max(int(item.get("completion_length", 0)) - valid_token_count, 0)
        return items

    def _active_response_count(self, items: list[dict[str, Any]]) -> int:
        return sum(
            int(item.get("encoded") is not None and int(item.get("valid_token_count", 0)) > 0) for item in items
        )

    def _failure_active_response_count(self, items: list[dict[str, Any]]) -> int:
        return sum(
            int(
                item.get("encoded") is not None
                and int(item.get("valid_token_count", 0)) > 0
                and float(item.get("reward", 0.0)) == 0.0
            )
            for item in items
        )

    def _train_record_base(self, item: dict[str, Any], prompt_uid: str) -> dict[str, float | int | str]:
        return {
            "prompt_uid": str(item.get("prompt_uid") or prompt_uid),
            "generation_index": int(item.get("generation_index", 0)),
            "reward": float(item.get("reward", 0.0)),
            "has_parseable_answer": float(bool(item.get("has_parseable_answer", False))),
            "is_correct": float(bool(item.get("is_correct", False))),
            "response_length": float(item.get("completion_length", 0)),
            "valid_token_count": float(item.get("valid_token_count", 0)),
            "ignored_token_count": float(item.get("ignored_token_count", 0)),
            "generation_seconds": float(item.get("generation_seconds", 0.0)),
            "grpo_advantage": float(item.get("grpo_advantage", 0.0)),
            "failure_advantage": float(item.get("failure_advantage", 0.0)),
            "group_reward_mean": float(item.get("group_reward_mean", 0.0)),
            "group_anti_reward_mean": float(item.get("group_anti_reward_mean", 0.0)),
        }

    def train_on_items(
        self,
        items: list[dict[str, Any]],
        solver_scale: float,
        failure_scale: float,
        prompt_uid: str,
        active_response_count: int,
        failure_active_response_count: int,
        prompt_batch_index: int,
        train_prompt_batch_size: int,
    ) -> list[dict[str, float | int | str]]:
        records = []
        batch_total = len(items)
        batch_correct = sum(float(item.get("reward", 0.0)) for item in items)
        batch_parseable = sum(float(bool(item.get("has_parseable_answer", False))) for item in items)
        for item in items:
            record = self._train_record_base(item, prompt_uid)
            record.update(
                {
                    "active_response_count": float(active_response_count),
                    "failure_active_response_count": float(failure_active_response_count),
                    "train_prompt_batch_index": int(prompt_batch_index),
                    "train_prompt_batch_size": int(train_prompt_batch_size),
                    "batch_correct": float(batch_correct),
                    "batch_total": float(batch_total),
                    "batch_accuracy": float(batch_correct / batch_total) if batch_total else 0.0,
                    "batch_parse_rate": float(batch_parseable / batch_total) if batch_total else 0.0,
                    "solver_scaled_loss": 0.0,
                    "failure_scaled_loss": 0.0,
                }
            )
            records.append(record)

        active_indices = [
            idx
            for idx, item in enumerate(items)
            if item.get("encoded") is not None and int(item.get("valid_token_count", 0)) > 0
        ]
        if not active_indices:
            return records

        # Use token-budget batching if enabled, otherwise fixed sample count
        batch_chunks = self._compute_token_budget_batches(items, active_indices)
        for chunk_indices in batch_chunks:
            chunk_items = [items[idx] for idx in chunk_indices]
            encoded_items = [item["encoded"] for item in chunk_items]

            self.set_active_adapter(self.config.solver_adapter_name, train=True)
            current, _, _ = self.batch_completion_logprobs(self.policy_model, encoded_items)
            old_logps = self._concat_item_tensors(chunk_items, "old_solver_logps", dtype=current.dtype, device=current.device)
            ref_logps = self._concat_item_tensors(chunk_items, "ref_logps", dtype=current.dtype, device=current.device)
            failure_logps = self._concat_item_tensors(
                chunk_items, "failure_logps_for_escape", dtype=current.dtype, device=current.device
            )
            advantages = self._concat_item_tensors(chunk_items, "token_advantages", dtype=current.dtype, device=current.device)
            failed_token_mask = self._concat_item_tensors(
                chunk_items, "failed_token_mask", dtype=torch.bool, device=current.device
            )
            loss_mask = self._concat_item_tensors(chunk_items, "token_loss_mask", dtype=torch.bool, device=current.device)
            solver_loss, solver_metrics = token_solver_fepo_loss(
                current,
                old_logps,
                ref_logps,
                failure_logps,
                advantages,
                failed_token_mask,
                clip_eps=self.config.clip_eps,
                reference_beta=self.config.fepo_reference_beta,
                escape_alpha=self.config.fepo_escape_alpha,
                loss_mask=loss_mask,
                token_log_ratio_clip=self.config.token_log_ratio_clip,
            )
            (solver_loss * solver_scale).backward()

            self.set_active_adapter(self.config.failure_adapter_name, train=True)
            failure_current, _, _ = self.batch_completion_logprobs(self.policy_model, encoded_items)
            failed_response_mask = self._concat_item_tensors(
                chunk_items, "failed_token_mask", dtype=torch.bool, device=failure_current.device
            )
            failure_loss_mask = loss_mask.to(device=failure_current.device) & failed_response_mask
            failure_loss, failure_metrics = token_failure_sft_loss(
                failure_current,
                is_failed=bool(failure_loss_mask.any().detach().cpu()),
                loss_mask=failure_loss_mask,
            )
            (failure_loss * failure_scale).backward()

            for record_index in chunk_indices:
                item = items[record_index]
                valid_token_count = float(item.get("valid_token_count", 0.0))
                failed_token_count = valid_token_count if float(item.get("reward", 0.0)) == 0.0 else 0.0
                records[record_index].update(solver_metrics)
                records[record_index].update(failure_metrics)
                records[record_index]["loss_token_count"] = valid_token_count
                records[record_index]["failed_token_count"] = failed_token_count
                records[record_index]["failure_sft_token_count"] = failed_token_count
                records[record_index]["failure_is_sft_active"] = float(failed_token_count > 0.0)
                records[record_index]["solver_scaled_loss"] = float((solver_loss.detach() * float(solver_scale)).cpu())
                records[record_index]["failure_scaled_loss"] = float((failure_loss.detach() * float(failure_scale)).cpu())
        return records

    def rollout_prompt_batch(
        self,
        batch: list[dict[str, Any]],
        batch_start: int,
        accumulation_index: int,
    ) -> list[tuple[str, int, list[dict[str, Any]]]]:
        jobs: list[tuple[str, int, dict[str, Any]]] = []
        for prompt_index, example in enumerate(batch):
            extra = example.get("extra_info", {})
            row_index = extra.get("index", (batch_start + prompt_index) % len(self.train_rows))
            prompt_uid = (
                f"step{self.global_step + 1}:accum{accumulation_index}:"
                f"prompt{prompt_index}:row{row_index}"
            )
            jobs.append((prompt_uid, prompt_index, example))

        concurrency = max(1, int(self.config.vllm_client_concurrency))
        if self.config.generation_backend != "vllm_openai" or concurrency <= 1 or len(jobs) <= 1:
            return [
                (prompt_uid, prompt_index, self.rollout_prompt(example, prompt_uid=prompt_uid))
                for prompt_uid, prompt_index, example in jobs
            ]

        max_workers = min(concurrency, len(jobs))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self.rollout_prompt, example, prompt_uid=prompt_uid)
                for prompt_uid, _, example in jobs
            ]
            return [
                (prompt_uid, prompt_index, future.result())
                for (prompt_uid, prompt_index, _), future in zip(jobs, futures, strict=True)
            ]

    def optimizer_step(self) -> dict[str, float]:
        solver_norm = torch.nn.utils.clip_grad_norm_(
            self.adapter_named_parameters(self.config.solver_adapter_name).values(), self.config.max_grad_norm
        )
        failure_norm = torch.nn.utils.clip_grad_norm_(
            self.adapter_named_parameters(self.config.failure_adapter_name).values(), self.config.max_grad_norm
        )
        self.solver_optimizer.step()
        self.failure_optimizer.step()
        self.solver_optimizer.zero_grad(set_to_none=True)
        self.failure_optimizer.zero_grad(set_to_none=True)
        return {"solver_grad_norm": float(solver_norm), "failure_grad_norm": float(failure_norm)}

    def train(self) -> None:
        if not self.train_rows:
            raise RuntimeError("Call setup() before train().")
        self.save_reload_and_cleanup(self.global_step if self.global_step > 0 else "initial")
        cursor = self.train_cursor
        while self.global_step < self.config.max_optimizer_steps:
            step_start = time.perf_counter()
            rollout_wall_seconds = 0.0
            loss_wall_seconds = 0.0
            step_records: list[dict[str, float | int | str]] = []
            accumulation_rollouts: list[tuple[int, int, list[tuple[str, int, list[dict[str, Any]]]]]] = []

            for accumulation_index in range(self.config.gradient_accumulation_steps):
                batch_start = cursor
                batch = [
                    self.train_rows[(cursor + offset) % len(self.train_rows)]
                    for offset in range(self.config.train_prompt_batch_size)
                ]
                cursor += self.config.train_prompt_batch_size
                rollout_start = time.perf_counter()
                rollouts = self.rollout_prompt_batch(
                    batch=batch,
                    batch_start=batch_start,
                    accumulation_index=accumulation_index,
                )
                rollout_wall_seconds += time.perf_counter() - rollout_start
                accumulation_rollouts.append((accumulation_index, len(batch), rollouts))
                self._print_step_samples(self.global_step + 1, rollouts)

            self.sleep_vllm_if_enabled()

            for _, train_prompt_batch_size, rollouts in accumulation_rollouts:
                active_response_count = sum(self._active_response_count(items) for _, _, items in rollouts)
                failure_active_response_count = sum(
                    self._failure_active_response_count(items) for _, _, items in rollouts
                )
                solver_scale = (
                    1.0 / float(self.config.gradient_accumulation_steps * active_response_count)
                    if active_response_count > 0
                    else 0.0
                )
                failure_scale = (
                    1.0 / float(self.config.gradient_accumulation_steps * failure_active_response_count)
                    if failure_active_response_count > 0
                    else 0.0
                )
                loss_start = time.perf_counter()
                for prompt_uid, prompt_index, items in rollouts:
                    step_records.extend(
                        self.train_on_items(
                            items,
                            solver_scale=solver_scale,
                            failure_scale=failure_scale,
                            prompt_uid=prompt_uid,
                            active_response_count=active_response_count,
                            failure_active_response_count=failure_active_response_count,
                            prompt_batch_index=prompt_index,
                            train_prompt_batch_size=train_prompt_batch_size,
                        )
                    )
                loss_wall_seconds += time.perf_counter() - loss_start
            self.global_step += 1
            self.train_cursor = cursor
            grad_metrics = self.optimizer_step()
            self.wake_vllm_if_needed()
            solver_dir, failure_dir = self.save_reload_and_cleanup(self.global_step)
            metrics = self._summarize_step(step_records)
            metrics.update(grad_metrics)
            step_seconds = time.perf_counter() - step_start
            metrics.update(
                {
                    "checkpoint/solver_adapter": str(solver_dir),
                    "checkpoint/failure_adapter": str(failure_dir),
                    "perf/step_seconds": float(step_seconds),
                    "perf/rollout_wall_seconds": float(rollout_wall_seconds),
                    "perf/loss_wall_seconds": float(loss_wall_seconds),
                    "perf/responses_per_second": float(len(step_records) / step_seconds) if step_seconds > 0 else 0.0,
                }
            )
            self.logger.log(metrics, step=self.global_step)
            if self.config.checkpoint_every_steps > 0 and self.global_step % self.config.checkpoint_every_steps == 0:
                self._save_checkpoint(cursor, solver_dir, failure_dir)
            if self.config.eval_every_steps > 0 and self.global_step % self.config.eval_every_steps == 0:
                self.evaluate()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _response_preview(self, text: Any, limit: int = 180) -> str:
        preview = " ".join(str(text).split())
        if len(preview) > limit:
            preview = preview[: limit - 3] + "..."
        return preview

    def _print_step_samples(self, step: int, rollouts: list[tuple[str, int, list[dict[str, Any]]]]) -> None:
        sample_limit = max(0, int(self.config.console_sample_count))
        if sample_limit <= 0:
            return
        printed = 0
        for prompt_uid, prompt_index, items in rollouts:
            if not items:
                continue
            first = items[0]
            print(
                "sample "
                f"step={step} prompt_index={prompt_index} "
                f"batch_acc={sum(float(item.get('reward', 0.0)) for item in items) / max(len(items), 1):.3f} "
                f"gt={first.get('ground_truth_normalized')} "
                f"pred={first.get('prediction_normalized')} "
                f"correct={bool(first.get('is_correct', False))} "
                f"response={self._response_preview(first.get('response', ''))}",
                flush=True,
            )
            printed += 1
            if printed >= sample_limit:
                break

    def _summarize_step(self, records: list[dict[str, float | int | str]]) -> dict[str, float | int]:
        return summarize_training_records(records)

    def evaluate(self) -> dict[str, dict[str, float | int | str]]:
        all_metrics = {}
        for dataset_name, rows in self.eval_rows_by_dataset.items():
            responses_by_prompt = []
            for start in range(0, len(rows), self.config.eval_batch_size):
                for example in rows[start : start + self.config.eval_batch_size]:
                    responses_by_prompt.append(
                        self.generate(
                            example["prompt"],
                            n=self.config.eval_generations,
                            temperature=self.config.eval_temperature,
                            top_p=self.config.eval_top_p,
                        )
                    )
            metrics = evaluate_responses_by_prompt(
                rows,
                responses_by_prompt,
                dataset_name,
                self.config.eval_generations,
            )
            self.logger.log(
                {
                    f"val/{dataset_name}/{key}": value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                },
                step=self.global_step,
            )
            all_metrics[dataset_name] = metrics
        return all_metrics

    def _save_checkpoint(self, cursor: int, solver_dir: Path, failure_dir: Path) -> None:
        self.checkpoint_manager.save(
            global_step=self.global_step,
            payload={
                "cursor": cursor,
                "solver_adapter_path": str(solver_dir),
                "failure_adapter_path": str(failure_dir),
            },
            model_state={
                "solver": {
                    k: v.detach().cpu()
                    for k, v in self.adapter_named_parameters(self.config.solver_adapter_name).items()
                },
                "failure": {
                    k: v.detach().cpu()
                    for k, v in self.adapter_named_parameters(self.config.failure_adapter_name).items()
                },
            },
            optimizer_state={
                "solver": self.solver_optimizer.state_dict(),
                "failure": self.failure_optimizer.state_dict(),
            },
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
            state.get("dataloader_state", {}).get("cursor", state.get("payload", {}).get("cursor", 0))
        )
        for name, tensor in state.get("model_state", {}).get("solver", {}).items():
            self.adapter_named_parameters(self.config.solver_adapter_name)[name].data.copy_(
                tensor.to(self.config.device)
            )
        for name, tensor in state.get("model_state", {}).get("failure", {}).items():
            self.adapter_named_parameters(self.config.failure_adapter_name)[name].data.copy_(
                tensor.to(self.config.device)
            )
        if state.get("optimizer_state", {}).get("solver"):
            self.solver_optimizer.load_state_dict(state["optimizer_state"]["solver"])
        if state.get("optimizer_state", {}).get("failure"):
            self.failure_optimizer.load_state_dict(state["optimizer_state"]["failure"])

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
