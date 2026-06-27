#!/usr/bin/env bash
# Dr.GRPO | Qwen2.5-3B-Instruct | FSDP training (param+optimizer CPU offload) | NVIDIA GPUs
# Dr.GRPO uses GRPO advantages without std normalization and constant response-length loss scaling.
# Trained on agentica-org/DeepScaleR-Preview-Dataset (~40k AIME/AMC/Omni-MATH/Still problems).
#
# Before first run:
#   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B
#   python3 -m verl.experimental.fepo.data \
#       --dataset agentica-org/DeepScaleR-Preview-Dataset --split train \
#       --output /workspace/jepa-grpo-cache/data/deepscaler_preview_train.parquet
#   (rows with an empty "answer" field must be filtered first; see git history of this file
#    or filter with datasets.Dataset.filter before calling convert_split_to_verl_rows)

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
export HF_HOME=${HF_HOME:-/workspace/jepa-grpo-cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/jepa-grpo-cache/hf/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/workspace/jepa-grpo-cache/hf/hub}
export TORCH_HOME=${TORCH_HOME:-/workspace/jepa-grpo-cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/jepa-grpo-cache/xdg}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/jepa-grpo-cache/vllm}
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
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
# Contamination-free cutoff for LiveCodeBench: only problems released on/after this
# date are kept. Should match the base model's pretraining cutoff, not the RL
# dataset's date. DeepSeek-R1-Distill-Qwen is distilled from Qwen2.5, reported with
# a mid-2024 knowledge cutoff.
LCB_CUTOFF_DATE=${LCB_CUTOFF_DATE:-2024-08-01}
# Full 10-benchmark suite (math + code) — expensive (2405 problems x VAL_ROLLOUT_N
# samples each). Used for the one-off final evaluation after training, not every
# test_freq step. Set VAL_FILES=$FULL_VAL_FILES to run it during training instead.
FULL_VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet,${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/humanevalplus.parquet,${EVAL_DATA_DIR}/mbppplus.parquet,${EVAL_DATA_DIR}/livecodebench.parquet]"
# Fast core-math subset (AIME 24/25/26 + AMC23, ~149 problems) used for routine
# in-training validation and best-checkpoint selection (see BEST_CKPT_SOURCES below).
# The remaining benchmarks (math500, minervamath, olympiadbench, code suites) are
# held out and only evaluated once at the end, with the best checkpoint -- see
# FINAL_VAL_FILES below.
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
fi
# data_source names (see verl/experimental/fepo/data.py) used to compute the average
# accuracy that decides whether to overwrite the `best/` checkpoint. Must be a subset
# of whatever's actually in VAL_FILES, or best-checkpoint tracking silently no-ops.
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
# Held-out benchmarks for the one-off final evaluation after training, using the
# best checkpoint (not seen during training validation).
FINAL_VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet,${EVAL_DATA_DIR}/humanevalplus.parquet,${EVAL_DATA_DIR}/mbppplus.parquet,${EVAL_DATA_DIR}/livecodebench.parquet]"
if [[ "${PREPARE_EVAL_DATA}" == "true" ]]; then
    mkdir -p "${EVAL_DATA_DIR}"
    for ds in math500 olympiadbench amc23 aime24 aime25 aime26 minervamath; do
        if [[ ! -f "${EVAL_DATA_DIR}/${ds}.parquet" ]]; then
            python3 -m verl.experimental.fepo.data \
                --dataset "math-ai/${ds}" --split test \
                --output "${EVAL_DATA_DIR}/${ds}.parquet"
        fi
    done
    # Code benchmarks (see verl/experimental/fepo/code_data.py). LiveCodeBench is
    # restricted to its stdin/stdout-test subset and filtered to LCB_CUTOFF_DATE.
    for cb in humanevalplus mbppplus; do
        if [[ ! -f "${EVAL_DATA_DIR}/${cb}.parquet" ]]; then
            python3 -m verl.experimental.fepo.code_data \
                --benchmark "${cb}" --output "${EVAL_DATA_DIR}/${cb}.parquet"
        fi
    done
    if [[ ! -f "${EVAL_DATA_DIR}/livecodebench.parquet" ]]; then
        python3 -m verl.experimental.fepo.code_data \
            --benchmark livecodebench --cutoff-date "${LCB_CUTOFF_DATE}" \
            --output "${EVAL_DATA_DIR}/livecodebench.parquet"
    fi
fi

# Dr.GRPO uses data.train_batch_size as prompts per optimizer step.
# On the local 96GB Blackwell GPU, 64 prompts x 8 rollouts gave the best tested utilization.
TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}   # 65536 OOM'd during actor backward; activation memory, not just optimizer state, scales with this
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
DRGRPO_USE_LORA=${DRGRPO_USE_LORA:-true}   # false -> full fine-tuning; set true to re-enable LoRA
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ACTOR_LR=${ACTOR_LR:-5e-7}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
CLIP_RATIO=${CLIP_RATIO:-0.2}
USE_KL_LOSS=${USE_KL_LOSS:-true}   # false -> drop KL entirely; set true to re-enable
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}   # 0.75 left vLLM holding too much memory for the log_prob/backward passes alongside it
# ROLLOUT_N matches NUM_GENERATIONS for consistency
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}   # more parallel sequences during vLLM rollout
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}    # halved vs. 1.5B to bound val-time rollout memory
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-500}   # fixed-length run on dapo-math-17k
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}   # catches val-path bugs/crashes immediately instead of after test_freq steps
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}   # only the most recent checkpoint is kept on disk

PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_deepscaler}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen2_5_3b_instruct_drgrpo_fsdp-${RUN_TIMESTAMP}}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files=${VAL_FILES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)

if [[ "${DRGRPO_USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    # FEPO-style: token-mean for per-token gradient signals
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    # CPU-offload the actor's params+grads (param_offload) and optimizer state
    # (optimizer_offload) to fit the 3B model on a single GPU. Gradients ride with
    # the params under FSDP CPU offload; there is no separate gradient_offload knob.
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD:-True}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD:-True}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
)

EXTRA=(
)

########################### launch ###########################
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND} python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
