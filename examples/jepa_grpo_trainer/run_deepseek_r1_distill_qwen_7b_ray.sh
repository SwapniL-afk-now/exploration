#!/usr/bin/env bash
# JEPA-GRPO | DeepSeek-R1-Distill-Qwen-7B | Ray + FSDP + hybrid-engine vLLM
# Trained on agentica-org/DeepScaleR-Preview-Dataset (~40k AIME/AMC/Omni-MATH/Still problems).
#
# Uses verl's full production stack:
#   - Ray for distribution
#   - FSDP (fsdp2) for actor training
#   - Hybrid engine: zero-copy FSDP→vLLM weight sync (no checkpoint save/reload)
#   - ActorRolloutRefWorker extended with EMA target encoder + JEPA update
#   - RayPPOTrainer extended with Code-view rollout and LeJEPA loss step
#
# L_total = L_DrGRPO(CoT) + alpha * L_LeJEPA(enc_q_cot, enc_a_code)
#
# Tuned down from run_deepseek_r1_distill_qwen_1_5b_ray.sh for the 7B model's larger
# footprint on a single 96GB GPU. Two things force LoRA back on here (the 1.5B script
# does full fine-tuning instead):
#   1. Full-FT Adam optimizer state for 7B params (fp32 master + momentum + variance)
#      needs ~80-90GB by itself, leaving no room for the vLLM rollout engine.
#   2. main_ray.py sets ref_in_actor = (lora_rank > 0): with LoRA enabled the reference
#      policy reuses the LoRA-disabled actor (no extra weights); with LoRA disabled a
#      second full 7B reference model copy must be loaded, which doesn't fit either.
# Rollout/batch sizes are also reduced to leave headroom for the larger base weights.
#
# Before first run:
#   hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --local-dir /workspace/models/DeepSeek-R1-Distill-Qwen-7B
#   python3 -m verl.experimental.fepo.data \
#       --dataset agentica-org/DeepScaleR-Preview-Dataset --split train \
#       --output /workspace/jepa-grpo-cache/data/deepscaler_preview_train.parquet
#   (rows with an empty "answer" field must be filtered first; see git history of this file
#    or filter with datasets.Dataset.filter before calling convert_split_to_verl_rows)

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
MODEL_PATH=${MODEL_PATH:-/workspace/models/DeepSeek-R1-Distill-Qwen-7B}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/deepscaler_preview_train.parquet}
NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}

EVAL_DATA_DIR=${EVAL_DATA_DIR:-/workspace/jepa-grpo-cache/eval_data}
PREPARE_EVAL_DATA=${PREPARE_EVAL_DATA:-true}
# Contamination-free cutoff for LiveCodeBench: only problems released on/after this
# date are kept. Should match the base model's pretraining cutoff, not the RL
# dataset's date. DeepSeek-R1-Distill-Qwen is distilled from Qwen2.5, reported with
# a mid-2024 knowledge cutoff.
LCB_CUTOFF_DATE=${LCB_CUTOFF_DATE:-2024-08-01}
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet,${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/humanevalplus.parquet,${EVAL_DATA_DIR}/mbppplus.parquet,${EVAL_DATA_DIR}/livecodebench.parquet]"
fi
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

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_dapo_math}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_r1_distill_qwen_7b_jepa_grpo_ray-${RUN_TIMESTAMP}}
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
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}    # 64/16 = 4 gradient steps per batch; smaller for 7B activation memory
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}  # halved vs. 1.5B; 7B activations are ~5x larger per token
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-32768}      # back to the conservative 1.5B-FSDP baseline for 7B
MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-400}

# Actor optimiser
ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO=${CLIP_RATIO:-0.2}
USE_LORA=${USE_LORA:-true}    # true -> required on a single 96GB GPU at 7B scale (see header)
LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-256}

# Rollout
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.50}   # lower than 1.5B's 0.85; 7B base weights take ~14GB alone
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-256}    # fewer parallel sequences to bound KV-cache growth

# JEPA
ALPHA=${ALPHA:-0.1}
EMA_DECAY=${EMA_DECAY:-0.99}
EMBED_MICRO_BATCH_SIZE=${EMBED_MICRO_BATCH_SIZE:-4}  # halved vs. 1.5B for embedding-pass activation memory
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
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}    # halved vs. 1.5B to bound val-time rollout memory
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-1.0}
VAL_TOP_P=${VAL_TOP_P:-0.95}

# KL penalty
USE_KL_LOSS=${USE_KL_LOSS:-false}   # false -> drop KL entirely; set true to re-enable
KL_COEF=${KL_COEF:-0.0}
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
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

if [[ "${USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
    )
else
    # The base config defaults lora_rank to 128; verl only treats LoRA as
    # disabled when lora_rank <= 0, so it must be explicitly zeroed out here.
    MODEL+=(
        actor_rollout_ref.model.lora_rank=0
    )
fi

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
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}
)

# Reuses the 1.5B-scale Hydra config as a base; every value that matters for 7B
# memory (model path, LoRA, rollout/batch sizes above) is overridden via CLI args.
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
