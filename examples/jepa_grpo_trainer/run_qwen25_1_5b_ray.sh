#!/usr/bin/env bash
# JEPA-GRPO | Qwen2.5-1.5B-Instruct | Ray + FSDP + hybrid-engine vLLM
#
# Uses verl's full production stack:
#   - Ray for distribution
#   - FSDP (fsdp2) for actor training
#   - Hybrid engine: zero-copy FSDP→vLLM weight sync (no checkpoint save/reload)
#   - ActorRolloutRefWorker extended with EMA target encoder + JEPA update
#   - RayPPOTrainer extended with Code-view rollout and LeJEPA loss step
#
# L_total = L_DrGRPO(CoT) + alpha * L_LeJEPA(enc_q_cot, enc_a_code)

set -euo pipefail
if [[ "${DEBUG_LAUNCH:-0}" == "1" ]]; then
    set -x
fi

if [[ -f .env ]]; then
    _XTRACE_WAS_ON=0
    case $- in *x*) _XTRACE_WAS_ON=1; set +x ;; esac
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
    [[ ${_XTRACE_WAS_ON} -eq 1 ]] && set -x
    unset _XTRACE_WAS_ON
fi

export HF_HOME=${HF_HOME:-/workspace/jepa-grpo-cache/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/workspace/jepa-grpo-cache/hf/datasets}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-/workspace/jepa-grpo-cache/hf/hub}
export TORCH_HOME=${TORCH_HOME:-/workspace/jepa-grpo-cache/torch}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-/workspace/jepa-grpo-cache/xdg}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/jepa-grpo-cache/vllm}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"

export VERL_PRINT_CONFIG=${VERL_PRINT_CONFIG:-0}
export VERL_CONSOLE_FULL_METRICS=${VERL_CONSOLE_FULL_METRICS:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-1}
export RAY_DISABLE_DOCKER_CPU_WARNING=${RAY_DISABLE_DOCKER_CPU_WARNING:-1}
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-ERROR}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-warning}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export WANDB_SILENT=${WANDB_SILENT:-true}
export WANDB_MODE=${WANDB_MODE:-online}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet]"
fi
if [[ "${PREPARE_EVAL_DATA}" == "true" ]]; then
    mkdir -p "${EVAL_DATA_DIR}"
    for ds in math500 olympiadbench amc23 aime24 aime25 aime26; do
        if [[ ! -f "${EVAL_DATA_DIR}/${ds}.parquet" ]]; then
            python3 -m verl.experimental.fepo.data \
                --dataset "math-ai/${ds}" --split test \
                --output "${EVAL_DATA_DIR}/${ds}.parquet"
        fi
    done
fi

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_dapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_1_5b_jepa_grpo_ray-${RUN_TIMESTAMP}}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOGGER=${LOGGER:-'["console","wandb"]'}

# Training size
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
# Split of ROLLOUT_N completions/prompt between CoT-framed and Code-framed
# system prompts (see jepa.n_cot/jepa.n_code below). Both views now
# contribute to the GRPO policy-gradient update, not just CoT — the
# code-framed subset is additionally used to build JEPA pairs. Must sum to
# ROLLOUT_N.
N_COT=${N_COT:-4}
N_CODE=${N_CODE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-32768}
MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-400}

# Actor optimiser
ACTOR_LR=${ACTOR_LR:-5e-7}
CLIP_RATIO=${CLIP_RATIO:-0.2}
LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-256}

# Rollout
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.70}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-512}

# JEPA
ALPHA=${ALPHA:-0.1}
EMA_DECAY=${EMA_DECAY:-0.99}
EMBED_MICRO_BATCH_SIZE=${EMBED_MICRO_BATCH_SIZE:-8}
MIN_VALID_PAIRS=${MIN_VALID_PAIRS:-2}
# JEPA objective: "lejepa" (squared-Euclidean align + SIGReg) or
# "llm-jepa-loss" (default; LLM-JEPA paper arXiv:2509.14252 cosine prediction loss + SIGReg).
JEPA_LOSS_TYPE=${JEPA_LOSS_TYPE:-llm-jepa-loss}
# Number of LLM-JEPA tied-weight predictor tokens (paper §3.1). Only used when
# JEPA_LOSS_TYPE=llm-jepa-loss; k=0 is the identity predictor, Pred(x) = x.
LLM_JEPA_PREDICTOR_K=${LLM_JEPA_PREDICTOR_K:-1}

SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}

# Validation rollout
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}

# KL penalty
KL_COEF=${KL_COEF:-0.001}
########################### end user-adjustable ###########################

DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="${VAL_FILES}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.lora_rank=${LORA_RANK}
    actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
)

JEPA=(
    jepa.n_cot=${N_COT}
    jepa.n_code=${N_CODE}
    jepa.alpha=${ALPHA}
    jepa.ema_decay=${EMA_DECAY}
    jepa.embed_micro_batch_size=${EMBED_MICRO_BATCH_SIZE}
    jepa.min_valid_pairs=${MIN_VALID_PAIRS}
    jepa.loss_type=${JEPA_LOSS_TYPE}
    jepa.predictor_k=${LLM_JEPA_PREDICTOR_K}
)

TRAINER=(
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.total_training_steps=${MAX_OPTIMIZER_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.logger=${LOGGER}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}
)

python3 -m verl.experimental.jepa_grpo.main_ray \
    --config-name jepa_grpo_ray_qwen25_1_5b \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${JEPA[@]}" \
    "${TRAINER[@]}" \
    "${ALGORITHM[@]}" \
    "$@"
