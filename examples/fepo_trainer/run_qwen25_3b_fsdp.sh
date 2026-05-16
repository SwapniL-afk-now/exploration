#!/usr/bin/env bash
# FEPO | Qwen2.5-3B-Instruct | experimental single-GPU LoRA trainer | NVIDIA GPUs
# FEPO uses solver/failure LoRA adapters through verl.experimental.fepo.main_fepo.

set -euo pipefail
if [[ "${DEBUG_LAUNCH:-0}" == "1" ]]; then
    set -x
fi

# Load local secrets/config without echoing them under `set -x`.
if [[ -f .env ]]; then
    _XTRACE_WAS_ON=0
    case $- in
        *x*) _XTRACE_WAS_ON=1; set +x ;;
    esac
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    if [[ ${_XTRACE_WAS_ON} -eq 1 ]]; then
        set -x
    fi
    unset _XTRACE_WAS_ON
fi

# Default local caches for repeatable training runs.
export HF_HOME=${HF_HOME:-/workspace/failure-escape-cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/failure-escape-cache/hf/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/workspace/failure-escape-cache/hf/hub}
export TORCH_HOME=${TORCH_HOME:-/workspace/failure-escape-cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/failure-escape-cache/xdg}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/failure-escape-cache/vllm}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"

# Keep terminal logs focused on run status, samples, metrics, and warnings that matter.
export VERL_PRINT_CONFIG=${VERL_PRINT_CONFIG:-0}
export VERL_VERBOSE_METRICS=${VERL_VERBOSE_METRICS:-0}
export VERL_CONSOLE_FULL_METRICS=${VERL_CONSOLE_FULL_METRICS:-0}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export RAY_DISABLE_DOCKER_CPU_WARNING=${RAY_DISABLE_DOCKER_CPU_WARNING:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_SILENT=${WANDB_SILENT:-true}

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-3B-Instruct}
TRAIN_DATASET=${TRAIN_DATASET:-zhuzilin/dapo-math-17k}
TRAIN_DATASET_CONFIG=${TRAIN_DATASET_CONFIG:-default}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
EVAL_DATASETS=${EVAL_DATASETS:-[math-ai/math500,math-ai/olympiadbench,math-ai/amc23,math-ai/aime24,math-ai/aime25,math-ai/aime26]}
EVAL_SPLIT=${EVAL_SPLIT:-test}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/fepo/dapo/qwen25_3b/${RUN_TIMESTAMP}}

PROJECT_NAME=${PROJECT_NAME:-failure-escape}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25-3b-fepo-dapo-fsdp-${RUN_TIMESTAMP}}
LOGGER=${LOGGER:-'["console","wandb"]'}

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-3000}
EVAL_EVERY_STEPS=${EVAL_EVERY_STEPS:-20}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
EVAL_GENERATIONS=${EVAL_GENERATIONS:-16}
EVAL_TEMPERATURE=${EVAL_TEMPERATURE:-1.0}
EVAL_TOP_P=${EVAL_TOP_P:-0.95}

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-2}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_RESPONSE_TOKENS=${MAX_RESPONSE_TOKENS:-2048}

LEARNING_RATE=${LEARNING_RATE:-5.0e-6}
FAILURE_LEARNING_RATE=${FAILURE_LEARNING_RATE:-1.25e-6}
FEPO_ESCAPE_ALPHA=${FEPO_ESCAPE_ALPHA:-0.01}
FEPO_REFERENCE_BETA=${FEPO_REFERENCE_BETA:-0.03}

VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.40}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
VLLM_CLIENT_CONCURRENCY=${VLLM_CLIENT_CONCURRENCY:-4}
LOSS_MICRO_BATCH_SIZE=${LOSS_MICRO_BATCH_SIZE:-8}
CONSOLE_SAMPLE_COUNT=${CONSOLE_SAMPLE_COUNT:-3}
########################### end user-adjustable ###########################

########################### launch ###########################
python3 -m verl.experimental.fepo.main_fepo \
    --config-name fepo_qwen25_3b \
    model_id="$MODEL_PATH" \
    train_dataset="$TRAIN_DATASET" \
    train_dataset_config=${TRAIN_DATASET_CONFIG} \
    train_split="$TRAIN_SPLIT" \
    eval_datasets="${EVAL_DATASETS}" \
    eval_split="$EVAL_SPLIT" \
    output_dir="$OUTPUT_DIR" \
    project_name="$PROJECT_NAME" \
    experiment_name="$EXPERIMENT_NAME" \
    logger="${LOGGER}" \
    max_optimizer_steps=${MAX_OPTIMIZER_STEPS} \
    eval_every_steps=${EVAL_EVERY_STEPS} \
    eval_batch_size=${EVAL_BATCH_SIZE} \
    eval_generations=${EVAL_GENERATIONS} \
    eval_temperature=${EVAL_TEMPERATURE} \
    eval_top_p=${EVAL_TOP_P} \
    train_prompt_batch_size=${TRAIN_PROMPT_BATCH_SIZE} \
    gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS} \
    num_generations=${NUM_GENERATIONS} \
    max_model_len=${MAX_MODEL_LEN} \
    max_response_tokens=${MAX_RESPONSE_TOKENS} \
    learning_rate=${LEARNING_RATE} \
    failure_learning_rate=${FAILURE_LEARNING_RATE} \
    fepo_escape_alpha=${FEPO_ESCAPE_ALPHA} \
    fepo_reference_beta=${FEPO_REFERENCE_BETA} \
    vllm_gpu_memory_utilization=${VLLM_GPU_MEM_UTIL} \
    vllm_attention_backend=${VLLM_ATTENTION_BACKEND} \
    actor_attention_impl=${ACTOR_ATTENTION_IMPL} \
    vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS} \
    vllm_client_concurrency=${VLLM_CLIENT_CONCURRENCY} \
    loss_micro_batch_size=${LOSS_MICRO_BATCH_SIZE} \
    console_sample_count=${CONSOLE_SAMPLE_COUNT} \
    "$@"
