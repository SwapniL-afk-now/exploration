#!/usr/bin/env bash
set -euo pipefail

# Example TAFR-GRPO launch. Override paths and model/data settings from the
# environment to keep the script usable for baselines.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}

cd "$REPO_ROOT"

if [[ -f "${REPO_ROOT}/.env" ]]; then
    _XTRACE_WAS_ON=0
    case $- in
        *x*) _XTRACE_WAS_ON=1; set +x ;;
    esac
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
    if [[ ${_XTRACE_WAS_ON} -eq 1 ]]; then
        set -x
    fi
    unset _XTRACE_WAS_ON
fi

PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_dapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_tafr_grpo_fsdp}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

# Match the wesserstein trainer's exact training and testing datasets.
TRAIN_DATASET=${TRAIN_DATASET:-zhuzilin/dapo-math-17k}
TRAIN_DATASET_CONFIG=${TRAIN_DATASET_CONFIG:-default}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TRAIN_FILE=${TRAIN_FILE:-"${WORKSPACE_ROOT}/failure-escape-runs/data/dapo_math_17k_train_full.parquet"}
PREPARE_TRAIN_DATA=${PREPARE_TRAIN_DATA:-true}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-"${WORKSPACE_ROOT}/failure-escape-runs/data/wesserstein_eval"}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
if [[ -z "${VAL_FILES:-}" ]]; then
    # Same eval set as examples/wesserstein_trainer/run_qwen25_1_5b_fsdp.sh.
    VAL_FILES="[${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet]"
fi

if [[ "${PREPARE_TRAIN_DATA}" == "true" && ! -f "${TRAIN_FILE}" ]]; then
    mkdir -p "$(dirname "${TRAIN_FILE}")"
    "$PYTHON_BIN" -m verl.experimental.fepo.data \
        --dataset "${TRAIN_DATASET}" \
        --config "${TRAIN_DATASET_CONFIG}" \
        --split "${TRAIN_SPLIT}" \
        --max-samples "${TRAIN_MAX_SAMPLES}" \
        --output "${TRAIN_FILE}"
fi

if [[ "${PREPARE_EVAL_DATA}" == "true" ]]; then
    mkdir -p "${EVAL_DATA_DIR}"
    if [[ ! -f "${EVAL_DATA_DIR}/amc23.parquet" ]]; then
        "$PYTHON_BIN" -m verl.experimental.fepo.data --dataset math-ai/amc23 --split test --output "${EVAL_DATA_DIR}/amc23.parquet"
    fi
    if [[ ! -f "${EVAL_DATA_DIR}/aime24.parquet" ]]; then
        "$PYTHON_BIN" -m verl.experimental.fepo.data --dataset math-ai/aime24 --split test --output "${EVAL_DATA_DIR}/aime24.parquet"
    fi
    if [[ ! -f "${EVAL_DATA_DIR}/aime25.parquet" ]]; then
        "$PYTHON_BIN" -m verl.experimental.fepo.data --dataset math-ai/aime25 --split test --output "${EVAL_DATA_DIR}/aime25.parquet"
    fi
fi

# ── TAFR-GRPO hyperparameters ────────────────────────────────────────────────
TAFR_VARIANT=${TAFR_VARIANT:-full}               # full | anchor_only | replay_only
TAFR_LOGPROB_BACKEND=${TAFR_LOGPROB_BACKEND:-vllm} # hf | vllm
TAFR_VLLM_SCORE_MICRO_BATCH_SIZE=${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE:-16}
TAFR_BETA=${TAFR_BETA:-0.01}                     # KL coefficient for both anchor and replay terms
TAFR_EMA_GAMMA=${TAFR_EMA_GAMMA:-0.99}           # EMA decay for GRPO and failure EMA trackers
TAFR_MIX_ETA=${TAFR_MIX_ETA:-1.0}               # mix weight: theta_anchor = (1-eta)*ref + eta*ema

# Failure-SFT schedule
TAFR_SFT_UPDATE_INTERVAL=${TAFR_SFT_UPDATE_INTERVAL:-2}       # run SFT every N GRPO steps
TAFR_CHECKPOINT_INTERVAL=${TAFR_CHECKPOINT_INTERVAL:-10}      # save + refresh EMA every N GRPO steps

# Failure-SFT optimizer
TAFR_SFT_LR=${TAFR_SFT_LR:-1.0e-6}                           # failure-SFT learning rate
# Chunk size when iterating over the interval's buffered failures.
# The buffer is cleared after every SFT update, so this controls
# how many examples go into each optimizer step within the interval.
TAFR_SFT_BATCH_SIZE=${TAFR_SFT_BATCH_SIZE:-4}
# Hard cap on optimizer steps per interval (9999 = effectively unlimited).
TAFR_SFT_MAX_UPDATES=${TAFR_SFT_MAX_UPDATES:-9999}

# Failure data collector — cleared automatically after each SFT update,
# so this is just a safety cap in case of an unusually large interval.
TAFR_FAILURE_DATA_MAX_SIZE=${TAFR_FAILURE_DATA_MAX_SIZE:-null} # null = unlimited
TAFR_FAILURE_DATA_SAMPLING=${TAFR_FAILURE_DATA_SAMPLING:-recent} # recent | uniform

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
DRGRPO_USE_LORA=${DRGRPO_USE_LORA:-true}
LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-256}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ACTOR_LR=${ACTOR_LR:-5e-7}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
PPO_LOSS_COEF=${PPO_LOSS_COEF:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-400}
SAVE_FREQ=${SAVE_FREQ:--10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

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
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_loss_coef=${PPO_LOSS_COEF}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}
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
)

TAFR=(
    custom_tafr_grpo.enable=true
    custom_tafr_grpo.variant="${TAFR_VARIANT}"
    custom_tafr_grpo.logprob_backend="${TAFR_LOGPROB_BACKEND}"
    custom_tafr_grpo.vllm_score_micro_batch_size="${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE}"
    # KL coefficients and EMA
    custom_tafr_grpo.beta="${TAFR_BETA}"
    custom_tafr_grpo.ema_gamma="${TAFR_EMA_GAMMA}"
    custom_tafr_grpo.mix_eta="${TAFR_MIX_ETA}"
    # Disable verl built-in KL (TAFR manages its own)
    custom_tafr_grpo.disable_builtin_kl=true
    # Failure-SFT schedule
    custom_tafr_grpo.sft_update_interval_grpo_steps="${TAFR_SFT_UPDATE_INTERVAL}"
    custom_tafr_grpo.checkpoint_interval_grpo_steps="${TAFR_CHECKPOINT_INTERVAL}"
    # Failure-SFT optimizer
    custom_tafr_grpo.failure_sft_lr="${TAFR_SFT_LR}"
    custom_tafr_grpo.failure_sft_batch_size="${TAFR_SFT_BATCH_SIZE}"
    custom_tafr_grpo.failure_sft_max_updates_per_interval="${TAFR_SFT_MAX_UPDATES}"
    # Failure data collector
    custom_tafr_grpo.failure_data_max_size="${TAFR_FAILURE_DATA_MAX_SIZE}"
    custom_tafr_grpo.failure_data_sampling="${TAFR_FAILURE_DATA_SAMPLING}"
)

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${TAFR[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
