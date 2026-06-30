#!/usr/bin/env bash
# JEPA-GRPO (dual-cache) | Qwen2.5-1.5B-Instruct | Ray + FSDP + hybrid-engine vLLM
#
# JEPA_LOSS_TYPE=jepa-tcr-dual (the only supported objective): the DIFFERENTIABLE dual loss
# (align_cot + align_code + self-consistency + SIGReg as a separate alpha-weighted backward)
# AND per-view teacher-alignment reward shaping. Each rollout's [PRED] latent is scored against
# its OWN view's teacher cache (CoT->CoT cache, code->code cache), standardized within its
# (uid, view, is_correct) reward stratum; beta*s_hat is folded into token_level_rewards BEFORE
# the GRPO advantage. Set TCR_REWARD_BETA=0 to disable shaping (pure differentiable dual).
#   L_total = L_DrGRPO(CoT+Code) + alpha * L_jepa-tcr-dual, advantages shaped by beta*s_hat
#
# Prereqs (build once, offline):
#   - CoT  cache: precompute_teacher_targets.py --view cot  -> TEACHER_CACHE
#   - Code cache: precompute_teacher_targets.py --view code -> CODE_TEACHER_CACHE
#     (the Qwen2.5-Coder-7B run produces CODE_TEACHER_CACHE)

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
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet]"
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
PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_deepscaler}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25_math_1_5b_jepa_tcr_dual_ray-${RUN_TIMESTAMP}}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOGGER=${LOGGER:-'["console","wandb"]'}

# Training size
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
# 4 CoT-framed + 4 Code-framed completions per prompt (must sum to ROLLOUT_N).
# Both views feed the GRPO policy gradient; the CORRECT ones become JEPA anchors.
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

# JEPA (jepa-tcr-dual)
ALPHA=${ALPHA:-0.1}
EMA_DECAY=${EMA_DECAY:-0.99}
# Raise so the per-step CORRECT-anchor block fits in ONE micro-batch: that keeps the
# GradCache memory-saving path at a SINGLE encoder forward (it only does the 2x
# cache+recompute when N_anchors > this). With <=64 correct anchors/step at the 1.5B
# scale this avoids the double forward. Lower it if the JEPA forward OOMs.
EMBED_MICRO_BATCH_SIZE=${EMBED_MICRO_BATCH_SIZE:-64}
MIN_VALID_PAIRS=${MIN_VALID_PAIRS:-2}
# jepa-tcr-dual = differentiable dual loss (align_cot + align_code + self-consistency
# + SIGReg) AND beta reward shaping (Step 3.5), both active.
JEPA_LOSS_TYPE=${JEPA_LOSS_TYPE:-jepa-tcr-dual}
# Tied-weight predictor tokens (paper §3.1). MUST be >0 for a real predictor
# (k=0 => Pred(x)=x identity; the shaping score would then be on the raw last token).
LLM_JEPA_PREDICTOR_K=${LLM_JEPA_PREDICTOR_K:-1}
# Reward-shaping knobs (jepa-tcr-dual): reward term = beta * standardized score.
TCR_REWARD_BETA=${TCR_REWARD_BETA:-0.5}
TCR_REWARD_SIGMA_FLOOR=${TCR_REWARD_SIGMA_FLOOR:-0.1}
# Auto-disable the JEPA aux signal once teacher-alignment plateaus: after AUTO_OFF_WARMUP
# steps, AUTO_OFF_PATIENCE consecutive steps without a >AUTO_OFF_MIN_DELTA improvement latch
# it off for the rest of training. ON by default here. For jepa-tcr-dual this is now PER-ARM:
# cos_cot, cos_code and cos_self each plateau and latch INDEPENDENTLY, disabling only their
# own loss term (and matching-view shaping); the whole block stops only once all three are off.
AUTO_OFF_ENABLE=${AUTO_OFF_ENABLE:-True}
AUTO_OFF_PATIENCE=${AUTO_OFF_PATIENCE:-10}
AUTO_OFF_MIN_DELTA=${AUTO_OFF_MIN_DELTA:-0.002}
AUTO_OFF_WARMUP=${AUTO_OFF_WARMUP:-20}
# (auto_off_metric left at config default "" => per-arm tracking above; SET it to a single
#  key via the extra-args passthrough to collapse back to one global latch instead.)
# Differentiable-variant knobs (only used when JEPA_LOSS_TYPE=jepa-tcr-dual).
SELF_CONSIST_W=${SELF_CONSIST_W:-1.0}
TCR_SIGREG_LAMBDA=${TCR_SIGREG_LAMBDA:-0.05}
JEPA_ANCHOR_SET=${JEPA_ANCHOR_SET:-correct}
# Offline target caches (build with precompute_teacher_targets.py --view {cot,code}).
TEACHER_CACHE=${TEACHER_CACHE:-/workspace/jepa-grpo-cache/teacher_targets.pt}
CODE_TEACHER_CACHE=${CODE_TEACHER_CACHE:-/workspace/jepa-grpo-cache/code_teacher_targets.pt}

SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}

# Validation rollout
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}

# KL penalty — TURNED OFF, per request.
KL_COEF=${KL_COEF:-0.0}
########################### end user-adjustable ###########################

for c in "${TEACHER_CACHE}" "${CODE_TEACHER_CACHE}"; do
    if [[ ! -f "${c}" ]]; then
        echo "ERROR: target cache not found: ${c}" >&2
        echo "Build it with examples/jepa_grpo_trainer/precompute_teacher_targets.py" >&2
        exit 1
    fi
done

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
    jepa.enable=True
    jepa.n_cot=${N_COT}
    jepa.n_code=${N_CODE}
    jepa.alpha=${ALPHA}
    jepa.ema_decay=${EMA_DECAY}
    jepa.embed_micro_batch_size=${EMBED_MICRO_BATCH_SIZE}
    jepa.min_valid_pairs=${MIN_VALID_PAIRS}
    jepa.loss_type=${JEPA_LOSS_TYPE}
    jepa.predictor_k=${LLM_JEPA_PREDICTOR_K}
    jepa.tcr_reward_beta=${TCR_REWARD_BETA}
    jepa.tcr_reward_sigma_floor=${TCR_REWARD_SIGMA_FLOOR}
    jepa.auto_off_enable=${AUTO_OFF_ENABLE}
    jepa.auto_off_patience=${AUTO_OFF_PATIENCE}
    jepa.auto_off_min_delta=${AUTO_OFF_MIN_DELTA}
    jepa.auto_off_warmup_steps=${AUTO_OFF_WARMUP}
    jepa.self_consist_w=${SELF_CONSIST_W}
    jepa.triplet_sigreg_lambda=${TCR_SIGREG_LAMBDA}
    jepa.jepa_anchor_set=${JEPA_ANCHOR_SET}
    jepa.teacher_cache_path=${TEACHER_CACHE}
    jepa.code_teacher_cache_path=${CODE_TEACHER_CACHE}
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
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES:-'[amc23,aime24,aime25,aime26]'}
    +trainer.best_ckpt_metric=${BEST_CKPT_METRIC:-avg@8}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.actor.use_kl_loss=False
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
