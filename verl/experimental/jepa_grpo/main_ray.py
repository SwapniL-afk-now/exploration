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
"""Entry point for Ray-based JEPA-GRPO training.

Usage:
    python -m verl.experimental.jepa_grpo.main_ray \\
        trainer.n_gpus_per_node=1 \\
        data.train_files=... \\
        actor_rollout_ref.model.path=...

Config is loaded in two steps:
  1. Load verl's ppo_trainer base config (from verl/trainer/config)
  2. Merge JEPA overrides on top (from jepa_grpo_ray_trainer.yaml)
  3. Apply CLI overrides

This uses verl's production infrastructure:
  - Ray for distribution
  - FSDP / FSDP2 for actor training
  - Hybrid engine (FSDP↔vLLM) for zero-copy weight sync
  - ActorRolloutRefWorker extended with JEPA EMA encoder
  - RayPPOTrainer extended with Code-view rollout and LeJEPA loss step
"""

import os
import socket
import sys
from pathlib import Path

import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


_VERL_TRAINER_CONFIG_DIR = str(Path(__file__).parent.parent.parent / "trainer" / "config")


def _coerce(value: str):
    """Convert a CLI string value to its native Python type."""
    value = value.strip()
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    if value.lower() in ("null", "none", "~"):
        return None
    # List: [a,b,c]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        return [_coerce(v.strip()) for v in inner.split(",") if v.strip()]
    # Strip surrounding quotes
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
_JEPA_CONFIG_DIR = str(Path(__file__).parent / "config")


def _load_config(overrides: list[str], config_name: str = "jepa_grpo_ray_trainer") -> DictConfig:
    """Load ppo_trainer base, merge JEPA overrides, then apply CLI overrides."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_VERL_TRAINER_CONFIG_DIR, version_base=None):
        base_cfg = compose(
            config_name="ppo_trainer",
            overrides=["model_engine=dp"],
        )

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=_JEPA_CONFIG_DIR, version_base=None):
        jepa_cfg = compose(config_name=config_name)

    GlobalHydra.instance().clear()

    # Convert base_cfg from struct to open (allows new keys like 'jepa')
    base_open = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=False))

    # Merge: base_cfg (ppo_trainer defaults) ← jepa_cfg (JEPA overrides) ← CLI
    cfg = OmegaConf.merge(base_open, jepa_cfg)

    # Apply CLI overrides (key=value pairs from sys.argv)
    for override in overrides:
        key, _, raw = override.partition("=")
        # A leading "+" is Hydra's "add a new key" marker (e.g. for a key under an
        # existing-but-empty struct dict like `override_config: {}`). OmegaConf.update
        # needs force_add=True to actually add it instead of raising "not in struct".
        force_add = key.startswith("+")
        key = key.lstrip("+~")
        # Coerce value to native Python type (Hydra normally does this; we're doing it manually)
        value = _coerce(raw)
        OmegaConf.update(cfg, key, value, merge=True, force_add=force_add)

    return cfg


def main():
    # Parse --config-name if provided (default: jepa_grpo_ray_trainer)
    config_name = "jepa_grpo_ray_trainer"
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        if arg in ("--config-name", "--config_name") and i + 1 < len(args):
            config_name = args[i + 1]
        elif arg.startswith("--config-name="):
            config_name = arg.split("=", 1)[1]

    # Split args into overrides (key=value) vs flags (--config-name etc.)
    cli_overrides = [a for a in args if "=" in a and not a.startswith("-")]

    config = _load_config(cli_overrides, config_name=config_name)
    auto_set_device(config)
    _run(config)


def _run(config):
    if not ray.is_initialized():
        # Clear stale RAY_ADDRESS that may point to a dead cluster (e.g. left by standalone trainer)
        os.environ.pop("RAY_ADDRESS", None)

        default_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env = OmegaConf.merge(default_env, ray_init_kwargs.get("runtime_env", {}))
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    task_runner_cls = ray.remote(num_cpus=1)(JEPATaskRunner)
    runner = task_runner_cls.remote()
    ray.get(runner.run.remote(config))


class JEPATaskRunner:
    """Task runner that wires in JEPAActorRolloutRefWorker and JEPARayPPOTrainer."""

    def run(self, config):
        from pprint import pprint

        print(f"JEPATaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        if os.environ.get("VERL_PRINT_CONFIG", "0") == "1":
            pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        from verl.experimental.jepa_grpo.ray_trainer import JEPARayPPOTrainer
        from verl.experimental.jepa_grpo.worker import JEPAActorRolloutRefWorker

        # ── Workers ─────────────────────────────────────────────────────
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0

        if need_reference_policy(config) and not ref_in_actor:
            actor_role = Role.ActorRolloutRef
        else:
            actor_role = Role.ActorRollout

        role_worker_mapping = {actor_role: ray.remote(JEPAActorRolloutRefWorker)}
        mapping = {actor_role: "global_pool"}

        # ── Resource pool ───────────────────────────────────────────────
        resource_pool_spec = {
            "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )

        # ── Model & tokenizer ───────────────────────────────────────────
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote)
        processor = hf_processor(local_path, trust_remote_code=trust_remote, use_fast=True)

        # ── Datasets ────────────────────────────────────────────────────
        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor, is_train=True
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, is_train=False
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # ── Trainer ─────────────────────────────────────────────────────
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )
        trainer = JEPARayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
