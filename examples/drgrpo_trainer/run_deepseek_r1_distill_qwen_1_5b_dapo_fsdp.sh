#!/usr/bin/env bash
# DAPO | DeepSeek-R1-Distill-Qwen-1.5B | FSDP training | NVIDIA GPUs
# DAPO (Decoupled Clip and Dynamic sAmpling Policy Optimization, arXiv:2503.14476) keeps the
# GRPO group-relative advantage (WITH std normalization, unlike Dr.GRPO) and adds:
#   - Clip-Higher: decoupled clip epsilons (clip_ratio_low=0.2, clip_ratio_high=0.28) + clip_ratio_c
#   - Token-level policy-gradient loss (loss_agg_mode=token-mean)
#   - Overlong reward shaping via the `dapo` reward manager (soft length penalty)
#   - KL divergence dropped entirely (no KL loss, no entropy bonus)
# Trained on agentica-org/DeepScaleR-Preview-Dataset (~40k AIME/AMC/Omni-MATH/Still problems).
#
# NOTE on Dynamic Sampling: DAPO's full dynamic sampling (resample gen batches until enough
# non-degenerate groups remain) lives in the dedicated dapo recipe trainer. The standard
# `verl.trainer.main_ppo` entry used here honours data.gen_batch_size but does NOT run the
# resampling loop, so algorithm.filter_groups is left disabled by default below. Enable it only
# when launching through the dapo recipe trainer.
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
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
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
# Best-checkpoint selection metric (verl/trainer/ppo/ray_trainer.py):
#   combined (default) = 0.5*(avg@k + pass@k) | avg | pass | avg@N | pass@N
# "pass@8" selects on the per-source val pass_at_8 averaged across BEST_CKPT_SOURCES,
# i.e. maximize MEAN pass@8 (best-of-8 success rate). VAL_ROLLOUT_N must be >= 8.
BEST_CKPT_METRIC=${BEST_CKPT_METRIC:-pass@8}
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

# DAPO uses data.train_batch_size as prompts per optimizer step.
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
DAPO_USE_LORA=${DAPO_USE_LORA:-true}   # false -> full fine-tuning; set true to re-enable LoRA
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

ACTOR_LR=${ACTOR_LR:-5e-7}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
# DAPO Clip-Higher: decoupled lower/upper PPO clip epsilons, plus clip_ratio_c (dual-clip).
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}
CLIP_RATIO_C=${CLIP_RATIO_C:-10.0}
# DAPO drops the KL term entirely (no KL loss, no KL-in-reward). Override to re-enable.
USE_KL_LOSS=${USE_KL_LOSS:-false}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.0}

# DAPO overlong reward shaping (soft length penalty applied by the `dapo` reward manager).
# Canonical DAPO uses a buffer of ~20% of max_response_length (official recipe: 4096/20480),
# with the penalty ramping linearly from 0 to penalty_factor over that final window.
ENABLE_OVERLONG_BUFFER=${ENABLE_OVERLONG_BUFFER:-true}
OVERLONG_BUFFER_LEN=${OVERLONG_BUFFER_LEN:-614}           # ~20% of MAX_RESPONSE_LENGTH=3072 (matches DAPO's 4096/20480 ratio)
OVERLONG_PENALTY_FACTOR=${OVERLONG_PENALTY_FACTOR:-1.0}

# DAPO dynamic sampling (group filtering). Canonical DAPO oversamples the generation
# batch (official recipe: gen 1536 / train 512 = 3x) and resamples groups whose rollouts
# all share the same metric (acc all-0 or all-1) until train_batch_size qualified groups
# remain. IMPORTANT: the resample loop is NOT implemented in this tree's
# `verl.trainer.main_ppo` (no Python reads algorithm.filter_groups here) — it lives only in
# the dedicated dapo recipe trainer. So this stays OFF for main_ppo; the canonical values
# below are pre-set so the run is correct the moment it's launched under a dapo trainer.
FILTER_GROUPS_ENABLE=${FILTER_GROUPS_ENABLE:-false}
FILTER_GROUPS_METRIC=${FILTER_GROUPS_METRIC:-acc}
MAX_NUM_GEN_BATCHES=${MAX_NUM_GEN_BATCHES:-10}
# Under main_ppo (no resample loop) gen must equal train, else extra prompts are generated
# and silently dropped. With dynamic sampling on, DAPO's 3x oversample applies instead.
if [[ "${FILTER_GROUPS_ENABLE}" == "true" ]]; then
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-$(( TRAIN_BATCH_SIZE * 3 ))}
else
    GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
fi

# DAPO training-rollout sampling: high-exploration generation (paper §4.1). Temperature 1.0
# with unrestricted top_p/top_k is what produces the diverse groups dynamic sampling/clip-higher
# rely on. (verl's rollout defaults already match these; set explicitly for reproducibility.)
TRAIN_TEMPERATURE=${TRAIN_TEMPERATURE:-1.0}
TRAIN_TOP_P=${TRAIN_TOP_P:-1.0}
TRAIN_TOP_K=${TRAIN_TOP_K:--1}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.75}   # 0.75 left vLLM holding too much memory for the log_prob/backward passes alongside it
# ROLLOUT_N matches NUM_GENERATIONS for consistency
ROLLOUT_N=${ROLLOUT_N:-${NUM_GENERATIONS}}
VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASHINFER}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}   # more parallel sequences during vLLM rollout
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}    # halved vs. 1.5B to bound val-time rollout memory
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}

MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-400}   # fixed-length run on dapo-math-17k
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}   # catches val-path bugs/crashes immediately instead of after test_freq steps
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}   # only the most recent checkpoint is kept on disk

PROJECT_NAME=${PROJECT_NAME:-verl_dapo_deepscaler}
RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
LOGGER=${LOGGER:-'["console","wandb"]'}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-${MAX_OPTIMIZER_STEPS}}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-0}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-null}
VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-null}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen_1_5b_dapo_fsdp-${RUN_TIMESTAMP}}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
########################### end user-adjustable ###########################

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$TRAIN_FILE"
    data.val_files=${VAL_FILES}
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    +data.gen_batch_size=${GEN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

# DAPO dynamic sampling (group filtering). Only effective under the dapo recipe trainer.
if [[ "${FILTER_GROUPS_ENABLE}" == "true" ]]; then
    DATA+=(
        +algorithm.filter_groups.enable=True
        +algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC}
        +algorithm.filter_groups.max_num_gen_batches=${MAX_NUM_GEN_BATCHES}
    )
fi

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)

if [[ "${DAPO_USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    # DAPO token-level loss: token-mean across all tokens in the mini-batch.
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    # DAPO Clip-Higher: decoupled lower/upper clip epsilons + dual-clip constant.
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=${CLIP_RATIO_C}
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${TRAIN_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${TRAIN_TOP_P}
    actor_rollout_ref.rollout.top_k=${TRAIN_TOP_K}
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

# DAPO overlong reward shaping, applied by the `dapo` reward manager.
REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${ENABLE_OVERLONG_BUFFER}
    +reward.reward_kwargs.overlong_buffer_cfg.len=${OVERLONG_BUFFER_LEN}
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${OVERLONG_PENALTY_FACTOR}
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
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
    +trainer.best_ckpt_metric=${BEST_CKPT_METRIC}
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
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${EXTRA[@]}" \
    "$@"
