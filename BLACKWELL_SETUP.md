# Blackwell RTX PRO 6000 Setup Guide for verl / TAFR-GRPO

This document describes how to set up the verl training framework with Flash Attention on NVIDIA Blackwell (sm_120) GPUs, specifically the RTX PRO 6000 Blackwell Workstation Edition.

## Hardware & Software

| Component | Version |
|-----------|---------|
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition (98 GB VRAM) |
| Compute Capability | 12.0 (sm_120) |
| Driver | 595.71.05 |
| CUDA | 13.0 |
| PyTorch | 2.11.0+cu130 |
| Python | 3.12.13 |
| flash-attn | 2.8.3 (built from source) |
| vLLM | 0.21.0 |
| verl | 0.8.0.dev |
| NCCL | 2.28.9+cuda13.0 |

## 1. Flash Attention 2 (sm_120)

Pre-built `flash-attn` wheels on PyPI do **not** include sm_120 kernels. You must build from source.

### Why FA2, not FA4?

- `flash-attn-4` (4.0.0b16) uses TCGEN05/TMEM instructions that are absent on sm_120.
- The `feat/sm120-support` branch from `blake-snc` fork works for FA4 CuTe DSL but is beta.
- FA2 2.8.3 compiles cleanly for sm_120 from source and is production-stable.

### Build from source

```bash
# Clone FA2
cd /tmp
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.8.3

# Build for sm_120
FLASH_ATTN_CUDA_ARCHS=120 \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
MAX_JOBS=4 \
pip install --no-build-isolation .
```

- `FLASH_ATTN_CUDA_ARCHS=120` -- targets only sm_120 (avoids compiling for all architectures).
- `MAX_JOBS=4` -- limits parallel compilation to avoid OOM during build.

### Verification

```python
import flash_attn
print(flash_attn.__version__)  # 2.8.3

# Smoke tests (all should pass)
from flash_attn import flash_attn_func
import torch
q = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(1, 8, 32, 64, dtype=torch.bfloat16, device='cuda')
out = flash_attn_func(q, k, v, causal=True)
print(f"FA2 output shape: {out.shape}")
```

## 2. verl Installation

```bash
pip install -e ".[gpu]"
```

This installs verl with GPU extras including vLLM 0.21.0.

## 3. Known Issues & Fixes

### 3.1 Ray workers can't find installed packages

**Problem:** Ray workers use `/usr/bin/python3` (system Python) instead of the venv Python, so they can't find packages installed in the venv.

**Fix:** Add the venv site-packages to `PYTHONPATH` in the Ray runtime env.

**File:** `verl/trainer/constants_ppo.py`

```python
_venv_site_packages = "/venv/main/lib/python3.12/site-packages"
_extra_pythonpath = f"{_venv_site_packages}:{_verl_repo}"

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        ...
        "PYTHONPATH": _extra_pythonpath,
    },
}
```

### 3.2 Ray workers don't inherit shell environment variables

**Problem:** Ray workers only get env vars from the explicit `runtime_env.env_vars` dict. They do **not** inherit the driver process's environment.

**Fix:** Pass critical env vars (like `WANDB_API_KEY`, `HF_TOKEN`) explicitly to the runtime env.

**File:** `verl/trainer/constants_ppo.py`

```python
# After the existing filter loop, explicitly forward vars that Ray workers need
for key in ("WANDB_API_KEY", "HF_TOKEN"):
    if os.environ.get(key) is not None:
        runtime_env["env_vars"][key] = os.environ[key]
```

### 3.3 Config fields missing for TAFR-GRPO and Wasserstein guidance

**Problem:** The bash script references config fields (`anchor_beta`, `embed_model`, `decode_model`) that weren't defined in the Python dataclasses or YAML defaults.

**Fixes:**

1. `verl/workers/config/actor.py` -- Added `embed_model` and `decode_model` to `WassersteinGuidanceConfig`.
2. `verl/trainer/config/actor/actor.yaml` -- Added default values for those fields.
3. `verl/experimental/tafr_grpo/config.py` -- Added `anchor_beta` to `TAFRGRPOConfig`.
4. `verl/trainer/config/ppo_trainer.yaml` -- Added `anchor_beta` to the TAFR-GRPO section.

### 3.4 Hydra override syntax

**Problem:** The bash script uses `+custom_tafr_grpo.anchor_beta=0.0` but the `+` prefix is for adding new keys, not overriding existing ones.

**Fix:** Remove the `+` prefix:

```bash
# Before (broken)
'+custom_tafr_grpo.anchor_beta=0.0'

# After (correct)
'custom_tafr_grpo.anchor_beta=0.0'
```

**File:** `examples/tafr_grpo/run_tafr_grpo_fsdp.sh`

### 3.5 Python binary path

**Problem:** The bash script defaults to `.venv/bin/python` which may not exist.

**Fix:** Override with `PYTHON_BIN`:

```bash
export PYTHON_BIN=/venv/main/bin/python3
```

### 3.6 NCCL on Blackwell

**Problem:** NCCL_NVLS (NVLink SHARP) and MNNVL (Multi-Node NVLink) can cause issues on Blackwell consumer GPUs.

**Fix:** Disable both in the Ray runtime env:

```python
_gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}
```

### 3.7 FSDP single-GPU warning

When running on a single GPU, FSDP falls back to `NO_SHARD`:

```
FSDP is switching to use NO_SHARD instead of ShardingStrategy.FULL_SHARD since the world size is 1.
```

This is expected and harmless.

## 4. Running TAFR-GRPO Training

```bash
# Start Ray
ray start --head --num-gpus=1

# Source environment (for wandb, HF tokens, etc.)
source /workspace/exploration/.env

# Set Python binary
export PYTHON_BIN=/venv/main/bin/python3

# Clear stale RAY_ADDRESS
unset RAY_ADDRESS

# Run training (default hyperparameters, 400 steps)
cd /workspace/exploration
bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh

# For quick testing, override max steps:
MAX_OPTIMIZER_STEPS=10 bash examples/tafr_grpo/run_tafr_grpo_fsdp.sh
```

## 5. Benchmark Results (Baseline, step 0)

| Benchmark | pass@1 | pass@8 | maj@8 | avg@k |
|-----------|--------|--------|-------|-------|
| AMC23 | 0.225 | 0.55 | 0.325 | 0.2047 |
| AIME24 | 0.033 | 0.067 | 0.033 | 0.015 |
| AIME25 | 0.000 | 0.100 | 0.000 | 0.017 |

## 6. Training Performance

- **~84s/step** for pure GRPO steps (rollout + training)
- **~159s/step** for steps with SFT updates (includes failure-data SFT)
- Validation on 3 benchmarks adds ~90s per evaluation (every 10 steps)

Estimated full 400-step run: ~10-12 hours.
