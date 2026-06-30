#!/usr/bin/env bash
set -euo pipefail

# Example TAFR-GRPO launch. Override paths and model/data settings from the
# environment to keep the script usable for baselines.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
WORKSPACE_ROOT=$(cd -- "${REPO_ROOT}/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}

cd "$REPO_ROOT"

# Reduce CUDA fragmentation on the colocated (vLLM + FSDP) single GPU.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

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

PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_deepscaler}
export WANDB_PROJECT=${WANDB_PROJECT:-${PROJECT_NAME}}
export WANDB_SILENT=${WANDB_SILENT:-true}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
# Default model/data/benchmark settings mirror
# examples/jepa_grpo_trainer/run_qwen_1_5b_ray.sh so TAFR-GRPO
# is an apples-to-apples comparison against the JEPA-GRPO runs.
TAFR_VARIANT=${TAFR_VARIANT:-full}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-tafr_grpo_${TAFR_VARIANT}_qwen25math_1_5b-${RUN_TIMESTAMP}}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-2}
ROLLOUT_AGENT_NUM_WORKERS=${ROLLOUT_AGENT_NUM_WORKERS:-1}

# Match the wesserstein trainer's exact training and testing datasets.
TRAIN_DATASET=${TRAIN_DATASET:-zhuzilin/dapo-math-17k}
TRAIN_DATASET_CONFIG=${TRAIN_DATASET_CONFIG:-default}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
PREPARE_TRAIN_DATA=${PREPARE_TRAIN_DATA:-false}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-false}
if [[ -z "${VAL_FILES:-}" ]]; then
    # Same in-training core-math subset as the JEPA-GRPO run.
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
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
# TAFR_VARIANT defaulted near the top (used in EXPERIMENT_NAME): full | anchor_only | replay_only
TAFR_LOGPROB_BACKEND=${TAFR_LOGPROB_BACKEND:-vllm} # hf | vllm  (vllm=fast multi-LoRA prefill scored in the awake-after-gen window, then sleep; falls back to hf on error)
TAFR_VLLM_SCORE_MICRO_BATCH_SIZE=${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE:-16}
TAFR_BETA=${TAFR_BETA:-0.1}                      # KL coefficient for both anchor and replay terms
TAFR_ANCHOR_BETA=${TAFR_ANCHOR_BETA:-0.0}        # anchor KL coefficient (overrides beta; 0 = off)
TAFR_EMA_GAMMA=${TAFR_EMA_GAMMA:-0.9}            # EMA decay for GRPO and failure EMA trackers
TAFR_MIX_ETA=${TAFR_MIX_ETA:-0.5}               # mix weight: theta_anchor = (1-eta)*ref + eta*ema

# Failure-SFT schedule
TAFR_SFT_UPDATE_INTERVAL=${TAFR_SFT_UPDATE_INTERVAL:-2}       # run SFT every N GRPO steps
TAFR_CHECKPOINT_INTERVAL=${TAFR_CHECKPOINT_INTERVAL:-10}      # save + refresh EMA every N GRPO steps

# Failure-SFT optimizer
TAFR_SFT_LR=${TAFR_SFT_LR:-5.0e-7}                           # failure-SFT learning rate
# Chunk size when iterating over the interval's buffered failures.
# The buffer is cleared after every SFT update, so this controls
# how many examples go into each optimizer step within the interval.
TAFR_SFT_BATCH_SIZE=${TAFR_SFT_BATCH_SIZE:-8}
# Token-budgeted failure-SFT micro-batching. >0 packs records greedily so each
# step's padded tokens (rows*max_len) stay under this budget => higher GPU util
# than the fixed batch size. Defaults to PPO_MAX_TOKEN_LEN_PER_GPU below (mirrors the
# actor-update budget); set 0 to disable and fall back to TAFR_SFT_BATCH_SIZE.
# (Real default applied after PPO_MAX_TOKEN_LEN_PER_GPU is defined; only a user
# override is honored here.)
TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU:-}

# TAFR disk-save throttling (disk shortage): the EMA refresh + vLLM adapter export
# still run every checkpoint interval, but the failure model / optimizer / tafr_state
# are only written when global_step % this == 0 (0 = every refresh). Old dirs rotate,
# so the latest save replaces the previous. Set the optimizer flag false to drop the
# largest file (loses failure-SFT momentum across a resume).
TAFR_SAVE_TO_DISK_INTERVAL=${TAFR_SAVE_TO_DISK_INTERVAL:-0}
TAFR_SAVE_FAILURE_OPTIMIZER=${TAFR_SAVE_FAILURE_OPTIMIZER:-true}
# Hard cap on optimizer steps per interval (9999 = effectively unlimited).
TAFR_SFT_MAX_UPDATES=${TAFR_SFT_MAX_UPDATES:-9999}

# Failure data collector — cleared automatically after each SFT update,
# so this is just a safety cap in case of an unusually large interval.
TAFR_FAILURE_DATA_MAX_SIZE=${TAFR_FAILURE_DATA_MAX_SIZE:-null} # null = unlimited
TAFR_FAILURE_DATA_SAMPLING=${TAFR_FAILURE_DATA_SAMPLING:-recent} # recent | uniform

TRAIN_PROMPT_BATCH_SIZE=${TRAIN_PROMPT_BATCH_SIZE:-64}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-${TRAIN_PROMPT_BATCH_SIZE}}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
# Failure-SFT token budget mirrors the actor-update budget unless overridden above.
TAFR_SFT_MAX_TOKEN_LEN_PER_GPU=${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
DRGRPO_USE_LORA=${DRGRPO_USE_LORA:-true}
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ACTOR_LR=${ACTOR_LR:-5e-7}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
PPO_LOSS_COEF=${PPO_LOSS_COEF:-1}
CLIP_RATIO=${CLIP_RATIO:-0.2}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-65536}
ROLLOUT_MAX_LORAS=${ROLLOUT_MAX_LORAS:-3}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-True}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}

# Wasserstein guidance
WG_ENABLE=${WG_ENABLE:-false}
WG_LAMBDA=${WG_LAMBDA:-0.01}
WG_ALPHA=${WG_ALPHA:-0.2}
WG_EMBED_MODEL=${WG_EMBED_MODEL:-BAAI/bge-small-en-v1.5}
WG_DECODE_MODEL=${WG_DECODE_MODEL:-${MODEL_PATH}}  # use actor tokenizer to decode response token IDs

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-250}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-false}
# best-checkpoint selection sources (matches the JEPA-GRPO run)
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}
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
    trainer.max_actor_ckpt_to_keep=1
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.log_val_generations=${LOG_VAL_GENERATIONS}
    trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}
    trainer.validation_data_dir=${VALIDATION_DATA_DIR}
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
)

TAFR=(
    custom_tafr_grpo.enable=true
    custom_tafr_grpo.variant="${TAFR_VARIANT}"
    custom_tafr_grpo.logprob_backend="${TAFR_LOGPROB_BACKEND}"
    custom_tafr_grpo.vllm_score_micro_batch_size="${TAFR_VLLM_SCORE_MICRO_BATCH_SIZE}"
    # KL coefficients and EMA
    custom_tafr_grpo.beta="${TAFR_BETA}"
    custom_tafr_grpo.anchor_beta="${TAFR_ANCHOR_BETA}"
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
    custom_tafr_grpo.failure_sft_max_token_len_per_gpu="${TAFR_SFT_MAX_TOKEN_LEN_PER_GPU}"
    custom_tafr_grpo.failure_sft_max_updates_per_interval="${TAFR_SFT_MAX_UPDATES}"
    # Disk-save throttling
    custom_tafr_grpo.save_to_disk_interval_grpo_steps="${TAFR_SAVE_TO_DISK_INTERVAL}"
    custom_tafr_grpo.save_failure_sft_optimizer="${TAFR_SAVE_FAILURE_OPTIMIZER}"
    # Failure data collector
    custom_tafr_grpo.failure_data_max_size="${TAFR_FAILURE_DATA_MAX_SIZE}"
    custom_tafr_grpo.failure_data_sampling="${TAFR_FAILURE_DATA_SAMPLING}"
)

WASSERSTEIN=(
    actor_rollout_ref.actor.wasserstein_guidance.enable=${WG_ENABLE}
    actor_rollout_ref.actor.wasserstein_guidance.lambda_wg=${WG_LAMBDA}
    actor_rollout_ref.actor.wasserstein_guidance.alpha_transport=${WG_ALPHA}
    actor_rollout_ref.actor.wasserstein_guidance.embed_model=${WG_EMBED_MODEL}
    actor_rollout_ref.actor.wasserstein_guidance.decode_model="${WG_DECODE_MODEL}"
)

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${TAFR[@]}" \
    "${WASSERSTEIN[@]}" \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    critic.enable=false \
    "$@"
