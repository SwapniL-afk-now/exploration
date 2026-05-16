#!/usr/bin/env bash
# Dr.GRPO | Qwen2.5-1.5B-Instruct | FSDP training | NVIDIA GPUs
# Dr.GRPO uses GRPO advantages without std normalization and constant response-length loss scaling.

set -xeuo pipefail

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
LOSS_SCALE_FACTOR=${LOSS_SCALE_FACTOR:-${MAX_RESPONSE_LENGTH}}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

ACTOR_LR=${ACTOR_LR:-1e-6}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
CLIP_RATIO=${CLIP_RATIO:-0.2}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
ROLLOUT_N=${ROLLOUT_N:-8}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}

PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_dapo_math}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_1_5b_drgrpo_fsdp}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
    actor_rollout_ref.actor.loss_scale_factor=${LOSS_SCALE_FACTOR}
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
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
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
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
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
)

EXTRA=(
)

########################### launch ###########################
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
