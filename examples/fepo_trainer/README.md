# FEPO Trainer

FEPO/failure-escape is an experimental single-GPU trainer that builds on Dr.GRPO-style group advantages but adds a failure-aware training path. It runs through `verl.experimental.fepo.main_fepo`, not `verl.trainer.main_ppo`, because it trains two named LoRA adapters and uses FEPO-specific checkpointing and generation flow.

## Algorithm Shape

The trainer keeps two adapters active in one policy model:

- `solver`: synced into vLLM for generation and trained with the FEPO solver loss.
- `failure`: trained with wrong-only SFT and used as the solver escape target.

The solver uses group-centered Dr.GRPO advantages without std normalization, plus FEPO reference and failure-escape terms. The failure adapter learns only from failed responses.

## Key Knobs

Each launcher exposes standard env vars and maps them to FEPO Hydra overrides:

```bash
MODEL_PATH -> model_id
TRAIN_DATASET -> train_dataset
TRAIN_DATASET_CONFIG -> train_dataset_config
TRAIN_SPLIT -> train_split
EVAL_DATASETS -> eval_datasets
EVAL_SPLIT -> eval_split
OUTPUT_DIR -> output_dir
PROJECT_NAME -> project_name
EXPERIMENT_NAME -> experiment_name
LOGGER -> logger
MAX_OPTIMIZER_STEPS -> max_optimizer_steps
EVAL_EVERY_STEPS -> eval_every_steps
EVAL_BATCH_SIZE -> eval_batch_size
EVAL_GENERATIONS -> eval_generations
TRAIN_PROMPT_BATCH_SIZE -> train_prompt_batch_size
GRADIENT_ACCUMULATION_STEPS -> gradient_accumulation_steps
NUM_GENERATIONS -> num_generations
MAX_MODEL_LEN -> max_model_len
MAX_RESPONSE_TOKENS -> max_response_tokens
LEARNING_RATE -> learning_rate
FAILURE_LEARNING_RATE -> failure_learning_rate
FEPO_ESCAPE_ALPHA -> fepo_escape_alpha
FEPO_REFERENCE_BETA -> fepo_reference_beta
VLLM_GPU_MEM_UTIL -> vllm_gpu_memory_utilization
VLLM_MAX_NUM_SEQS -> vllm_max_num_seqs
VLLM_CLIENT_CONCURRENCY -> vllm_client_concurrency
LOSS_MICRO_BATCH_SIZE -> loss_micro_batch_size
CONSOLE_SAMPLE_COUNT -> console_sample_count
```

## Launchers

| Script | Default model | Effective responses / optimizer step |
| --- | --- | --- |
| `run_qwen25_1_5b_fsdp.sh` | `Qwen/Qwen2.5-1.5B-Instruct` | `4 * 2 * 8 = 64` |
| `run_qwen25_3b_fsdp.sh` | `Qwen/Qwen2.5-3B-Instruct` | `2 * 4 * 8 = 64` |
| `run_qwen25_7b_fsdp.sh` | `Qwen/Qwen2.5-7B-Instruct` | `1 * 8 * 8 = 64` |
| `run_qwen25_math_1_5b_fsdp.sh` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | `4 * 2 * 8 = 64` |
| `run_qwen25_math_3b_fsdp.sh` | `Qwen/Qwen2.5-Math-3B-Instruct` | `2 * 4 * 8 = 64` |
| `run_qwen25_math_7b_fsdp.sh` | `Qwen/Qwen2.5-Math-7B-Instruct` | `1 * 8 * 8 = 64` |

The defaults target a single RTX PRO 6000 Blackwell-class 96GB GPU.

## Dataset Defaults

DAPO-Math-17K is the default training set:

```bash
TRAIN_DATASET=zhuzilin/dapo-math-17k
TRAIN_DATASET_CONFIG=default
TRAIN_SPLIT=train
EVAL_DATASETS='[math-ai/math500,math-ai/olympiadbench,math-ai/amc23,math-ai/aime24,math-ai/aime25,math-ai/aime26]'
```

Skywork OR1 Math remains available through env overrides:

```bash
TRAIN_DATASET=sungyub/skywork-or1-math-verl \
TRAIN_DATASET_CONFIG=null \
examples/fepo_trainer/run_qwen25_math_1_5b_fsdp.sh
```

## Tiny Smoke Run

```bash
MAX_OPTIMIZER_STEPS=1 \
TRAIN_PROMPT_BATCH_SIZE=1 \
GRADIENT_ACCUMULATION_STEPS=1 \
EVAL_EVERY_STEPS=0 \
LOGGER='[console]' \
examples/fepo_trainer/run_qwen25_math_1_5b_fsdp.sh
```

## Recommended Comparison Order

```text
1. GRPO baseline
2. Dr.GRPO baseline
3. FEPO / failure-escape
```

Checkpoints are saved after every optimizer step by default and resume with `resume_mode=auto`.
