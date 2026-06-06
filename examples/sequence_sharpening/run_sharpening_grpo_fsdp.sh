#!/usr/bin/env bash
set -euo pipefail

# Example Sequence Sharpening GRPO launch. Mirrors
# examples/tafr_grpo/run_tafr_grpo_fsdp.sh so that the only differences
# are the algorithm-specific knobs. Override paths and model/data
# settings from the environment to keep the script usable for baselines.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"/venv/main/bin/python3"}

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

PROJECT_NAME=${PROJECT_NAME:-verl_sharpening_grpo_ttt_aime24}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_SILENT=${WANDB_SILENT:-true}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_sharpening_grpo_ttt_aime24}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-Math-1.5B-Instruct}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-"${WORKSPACE_ROOT}/failure-escape-runs/data/wesserstein_eval"}

# Test-time training: train and eval on the same benchmark (AIME24).
TRAIN_DATASET=${TRAIN_DATASET:-math-ai/aime24}
TRAIN_DATASET_CONFIG=${TRAIN_DATASET_CONFIG:-default}
TRAIN_SPLIT=${TRAIN_SPLIT:-test}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TRAIN_FILE=${TRAIN_FILE:-"${EVAL_DATA_DIR}/aime24.parquet"}
PREPARE_TRAIN_DATA=${PREPARE_TRAIN_DATA:-false}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
if [[ -z "${VAL_FILES:-}" ]]; then
    # Test-time training: eval on the same benchmark.
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet]"
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

# ── Sequence Sharpening hyperparameters ──────────────────────────────────────
# When SHARPEN_ENABLE is false the entire custom_sharpening_grpo namespace
# is still passed in (with enable=false) so the trainer's validation hook
# runs but the loss falls back to the verl vanilla PPO loss.
SHARPEN_ENABLE=${SHARPEN_ENABLE:-true}
SHARPEN_USE_GRPO_REWARD=${SHARPEN_USE_GRPO_REWARD:-false}
SHARPEN_GAMMA=${SHARPEN_GAMMA:-0.3}
SHARPEN_ALPHA=${SHARPEN_ALPHA:-1.0}
SHARPEN_BETA=${SHARPEN_BETA:-0.1}
SHARPEN_CLIP_RATIO=${SHARPEN_CLIP_RATIO:-0.2}

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-8}
NUM_GENERATIONS=${NUM_GENERATIONS:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
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
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-49152}
ROLLOUT_MAX_LORAS=${ROLLOUT_MAX_LORAS:-3}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
# 32 val rollouts per prompt => the verl auto-metric block emits
# val/<dataset>/pass_at_{1,4,8,16,32} and val/<dataset>/avg_at_{1,4,8,16,32}.
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-30}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-320}
SAVE_FREQ=${SAVE_FREQ:-10}
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
    data.dataloader_num_workers=${DATALOADER_NUM_WORKERS}
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
    # Sequence Sharpening manages its own K3 KL term, so disable verl's
    # built-in KL loss to avoid double-counting. The validation hook in
    # verl.utils.config enforces this when custom_sharpening_grpo is enabled.
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
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_AGENT_NUM_WORKERS}
    actor_rollout_ref.rollout.free_cache_engine=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.enable_sleep_mode=${ROLLOUT_FREE_CACHE_ENGINE}
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max_loras=${ROLLOUT_MAX_LORAS}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.resume_mode=auto
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger=${LOGGER}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=2
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
)

SHARPENING=(
    custom_sharpening_grpo.enable=${SHARPEN_ENABLE}
    custom_sharpening_grpo.use_grpo_reward=${SHARPEN_USE_GRPO_REWARD}
    custom_sharpening_grpo.gamma=${SHARPEN_GAMMA}
    custom_sharpening_grpo.alpha=${SHARPEN_ALPHA}
    custom_sharpening_grpo.beta=${SHARPEN_BETA}
    custom_sharpening_grpo.group_size=${NUM_GENERATIONS}
    custom_sharpening_grpo.clip_ratio=${SHARPEN_CLIP_RATIO}
)

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${SHARPENING[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
