# Environment Setup

This file documents the exact library versions used in this project so reinstallation is reproducible.

## Hardware

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (94.97 GiB VRAM)
- CUDA Driver: 595.71.05
- Compute Capability: 12.0

## System

- Python: 3.12.13
- CUDA Toolkit: 12.8 (nvcc release 12.8, V12.8.93)

## Key Library Versions

| Library | Version |
|---|---|
| torch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| triton | 3.3.1 |
| flash-attn | 2.8.3 |
| vllm | 0.10.0 |
| ray | 2.55.1 |
| transformers | 4.52.4 |
| peft | 0.19.1 |
| datasets | 4.8.5 |
| numpy | 1.26.4 |
| hydra-core | 1.3.2 |
| wandb | 0.27.0 |
| einops | 0.8.2 |
| codetiming | 1.4.0 |

## Python Environment

Uses `uv` with Python 3.12:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12
source .venv/bin/activate
```

## Installation Order

**Install PyTorch first** (must match CUDA version before vllm or flash-attn):

```bash
uv pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

**Install flash-attn** (build from source or use prebuilt wheel matching torch+CUDA):

```bash
uv pip install flash-attn==2.8.3 --no-build-isolation
```

**Install vllm** (must match torch version — vllm 0.10.0 requires torch 2.7.x):

```bash
uv pip install vllm==0.10.0
```

**Install Ray:**

```bash
uv pip install ray==2.55.1
```

**Install remaining dependencies:**

```bash
uv pip install \
  transformers==4.52.4 \
  peft==0.19.1 \
  datasets==4.8.5 \
  numpy==1.26.4 \
  hydra-core==1.3.2 \
  wandb==0.27.0 \
  einops==0.8.2 \
  codetiming==1.4.0
```

**Install the project in editable mode:**

```bash
uv pip install -e ".[dev]"
```

## Known Memory Configuration

This setup runs on a single 95 GiB GPU with both vllm (rollout) and the FSDP actor sharing the same device. The following defaults in `examples/tafr_grpo/run_tafr_grpo_fsdp.sh` were tuned to avoid OOM:

| Parameter | Value | Why |
|---|---|---|
| `ROLLOUT_GPU_MEM_UTIL` | 0.45 | vllm at 0.5 leaves too little room for the actor backward pass |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | 16384 | Halves activation memory during PPO update; 32768 causes OOM |
| `actor.fsdp_config.optimizer_offload` | True | Offloads optimizer states to CPU to save GPU memory |

If you have a GPU with less than 95 GiB, reduce `ROLLOUT_GPU_MEM_UTIL` and `PPO_MAX_TOKEN_LEN_PER_GPU` further.

## Credentials

`WANDB_API_KEY` and other secrets live in `.env` at the repo root. The run script sources it automatically — no manual `wandb login` needed.
