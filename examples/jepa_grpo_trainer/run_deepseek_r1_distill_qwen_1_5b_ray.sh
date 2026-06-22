#!/usr/bin/env bash
# JEPA-GRPO | DeepSeek-R1-Distill-Qwen-1.5B | Ray + FSDP + hybrid-engine vLLM
# Trained on agentica-org/DeepScaleR-Preview-Dataset (~40k AIME/AMC/Omni-MATH/Still problems).
#
# Uses verl's full production stack:
#   - Ray for distribution
#   - FSDP (fsdp2) for actor training
#   - Hybrid engine: zero-copy FSDP→vLLM weight sync (no checkpoint save/reload)
#   - ActorRolloutRefWorker extended with EMA target encoder + JEPA update
#   - RayPPOTrainer extended with Code-view rollout and LeJEPA loss step
#
# L_total = L_DrGRPO(CoT) + alpha * L_JEPA(...), where L_JEPA is selected by
# JEPA_LOSS_TYPE below: lejepa | llm-jepa-loss | jepa-triplet-loss
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
# Dataset matched to the dr_grpo baseline (examples/drgrpo_trainer): DAPO-Math-17k.
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
# Full 10-benchmark suite (math + code) — expensive. Used for the one-off final
# evaluation after training, not every test_freq step. Set VAL_FILES=$FULL_VAL_FILES
# to run it during training instead.
FULL_VAL_FILES="[${EVAL_DATA_DIR}/math500.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/minervamath.parquet,${EVAL_DATA_DIR}/olympiadbench.parquet,${EVAL_DATA_DIR}/amc23.parquet,${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/humanevalplus.parquet,${EVAL_DATA_DIR}/mbppplus.parquet,${EVAL_DATA_DIR}/livecodebench.parquet]"
# Fast core-math subset (AIME 24/25/26 + AMC23) used for routine in-training
# validation and best-checkpoint selection (see BEST_CKPT_SOURCES below). The
# remaining benchmarks are held out and only evaluated once at the end, with the
# best checkpoint -- see FINAL_VAL_FILES below.
if [[ -z "${VAL_FILES:-}" ]]; then
    VAL_FILES="[${EVAL_DATA_DIR}/aime24.parquet,${EVAL_DATA_DIR}/aime25.parquet,${EVAL_DATA_DIR}/aime26.parquet,${EVAL_DATA_DIR}/amc23.parquet]"
fi
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

RUN_TIMESTAMP=${RUN_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}
PROJECT_NAME=${PROJECT_NAME:-verl_drgrpo_deepscaler}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-deepseek_r1_distill_qwen_1_5b_jepa_grpo_ray_nosep_nokl-${RUN_TIMESTAMP}}
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
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}   # 64/32 = 2 gradient steps per batch, more stable
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-3072}
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-24576}     # 1.5x: 32768 OOM'd at step 80 (save+eval boundary) on the ~9.2GB full-vocab lm_head logits block; 24576 shrinks that block to ~6.9GB for real headroom
MAX_OPTIMIZER_STEPS=${MAX_OPTIMIZER_STEPS:-400}   # fixed-length run on dapo-math-17k, matches the dr_grpo baseline

# Actor optimiser
ACTOR_LR=${ACTOR_LR:-5e-7}   # matched to the dr_grpo baseline (examples/drgrpo_trainer)
CLIP_RATIO=${CLIP_RATIO:-0.2}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.00}   # small entropy bonus; previously 0 let the policy drift unconstrained (length blow-up)
ACTOR_ATTENTION_IMPL=${ACTOR_ATTENTION_IMPL:-flash_attention_2}
USE_LORA=${USE_LORA:-true}   # false -> full fine-tuning; set false to disable LoRA
LORA_RANK=${LORA_RANK:-512}
LORA_ALPHA=${LORA_ALPHA:-1024}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}

# Rollout
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}   # leaves headroom for the Code-view rollout + JEPA forward
# Gradient checkpointing: True (default) recomputes activations in backward to save memory;
# False is faster on update_actor/jepa_update but uses more activation memory (OOM risk).
GRAD_CKPT=${GRAD_CKPT:-True}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-1024}   # more parallel sequences during vLLM rollout

# JEPA
# Scaled down from 0.5: jepa/grad_norm was measured at ~13-17x actor/grad_norm
# even after alpha=0.5 and max_grad_norm clipping (JEPA's loss is averaged
# over a ~30-80 row pool vs PPO's token-mean over ~370k tokens, so it survives
# backward much less diluted) — JEPA's update was dominating the shared LoRA
# weights instead of the RL signal. 0.01 is a first attempt at parity; compare
# jepa/grad_norm vs actor/grad_norm in this run to see if it lands closer.
ALPHA=${ALPHA:-0.005}
EMA_DECAY=${EMA_DECAY:-0.99}
EMBED_MICRO_BATCH_SIZE=${EMBED_MICRO_BATCH_SIZE:-8}
MIN_VALID_PAIRS=${MIN_VALID_PAIRS:-2}
# JEPA objective (one of four; see verl/experimental/jepa_grpo/core_algos.py):
#   "lejepa"             - squared-Euclidean align (live CoT vs EMA-target Code)
#                          + SIGReg. Uses SIGREG_LAMBDA.
#   "llm-jepa-loss"      - LLM-JEPA paper arXiv:2509.14252 cosine prediction loss
#                          (one shared encoder, no EMA) + SIGReg. Uses SIGREG_LAMBDA.
#   "jepa-triplet-loss"  - llm-jepa-loss + a hard-negative triplet term against a
#                          "clean wrong" code rollout; see
#                          jepa-llm-hard-neg-triplet-mode-AUDIT.md. Uses
#                          TRIPLET_MARGIN/TRIPLET_W/TRIPLET_SIGREG_LAMBDA.
#   "jepa-separation-loss" - llm-jepa-loss + a direct correct/wrong code-pair
#                          separation hinge (creates the gap the triplet's
#                          vanishing gradient cannot). Uses
#                          SEPARATION_MARGIN/SEPARATION_W/TRIPLET_SIGREG_LAMBDA.
#   "jepa-clreg-loss"    - v3 CLReg objective (jepa_separation_loss.md): replaces
#                          the 1:1 hinge with a per-GRPO-group FULL cross product
#                          of every correct anchor against EVERY wrong joint
#                          embedding, scored by a DPO log-sigmoid at temperature
#                          SEPARATION_TAU (or InfoNCE when SEPARATION_MODE=info).
#                          Uses SEPARATION_TAU/SEPARATION_MODE/SEPARATION_W/
#                          TRIPLET_SIGREG_LAMBDA (SEPARATION_MARGIN is ignored).
JEPA_LOSS_TYPE=${JEPA_LOSS_TYPE:-jepa-clreg-loss}
case "${JEPA_LOSS_TYPE}" in
    lejepa|llm-jepa-loss|jepa-triplet-loss|jepa-separation-loss|jepa-clreg-loss) ;;
    *) echo "ERROR: JEPA_LOSS_TYPE='${JEPA_LOSS_TYPE}' is invalid. Must be one of:" \
            "lejepa | llm-jepa-loss | jepa-triplet-loss | jepa-separation-loss | jepa-clreg-loss" >&2
       exit 1 ;;
esac
# SIGReg anti-collapse weight for the modes that read the GENERAL lambda
# (lejepa, llm-jepa-loss). The triplet/separation modes use TRIPLET_SIGREG_LAMBDA
# instead (see worker.py: lambda_=cfg.triplet_sigreg_lambda there vs
# cfg.sigreg_lambda here). Pretrained LLMs need far more SIGReg weight than
# LeJEPA's from-scratch default (0.1) to fight their anisotropy prior.
SIGREG_LAMBDA=${SIGREG_LAMBDA:-0.5}
# Number of LLM-JEPA tied-weight predictor tokens (paper §3.1). Only used when
# JEPA_LOSS_TYPE=llm-jepa-loss or jepa-triplet-loss; k=0 is the identity predictor,
# Pred(x) = x.
LLM_JEPA_PREDICTOR_K=${LLM_JEPA_PREDICTOR_K:-1}
# Hard-negative triplet hyperparameters. Only used when JEPA_LOSS_TYPE=jepa-triplet-loss.
TRIPLET_MARGIN=${TRIPLET_MARGIN:-0.1}
TRIPLET_W=${TRIPLET_W:-1.0}
TRIPLET_SIGREG_LAMBDA=${TRIPLET_SIGREG_LAMBDA:-0.3}   # SIGReg must both replace the dropped LLM-JEPA NTP anti-collapse leg AND fight the pretrained LLM's anisotropy prior, so it needs far more weight than LeJEPA's from-scratch default (0.05/0.1); 0.3 still lost (pos/neg re-converged, margin went negative by step ~24)
# Separation-loss hyperparameters. Only used when JEPA_LOSS_TYPE=jepa-separation-loss.
# L_sep = (1/T) Σ relu(SEPARATION_MARGIN - (1 - <e^c,e^w>)); creates the correct/wrong
# gap the triplet's vanishing (e^c-e^w) gradient could not. e^w is added to the SIGReg
# pool here so it cannot collapse. SEPARATION_W=1.0 => unweighted negative term.
SEPARATION_MARGIN=${SEPARATION_MARGIN:-0.1}
SEPARATION_W=${SEPARATION_W:-0.0}   # 0.0 => separation/CLReg negative term OFF (align + SIGReg only)
# CLReg (v3) hyperparameters. Only used when JEPA_LOSS_TYPE=jepa-clreg-loss
# (SEPARATION_MARGIN is ignored in that mode). SEPARATION_TAU is the contrastive
# temperature replacing the hinge margin; SEPARATION_MODE selects the negative
# form: "dpo" (averaged-pairwise log-sigmoid, doc default) or "info" (InfoNCE).
# Re-tune SEPARATION_W once live (bounded hinge -> unbounded log-sigmoid); sweep
# SEPARATION_TAU in {0.1,0.3,0.5,0.7,0.9}.
SEPARATION_TAU=${SEPARATION_TAU:-0.5}
SEPARATION_MODE=${SEPARATION_MODE:-dpo}
# Number of random projection directions for SIGReg's Epps-Pulley statistic.
# Each direction is a random unit vector in R^d, where d is the encoder's final
# hidden size (DeepSeek-R1-Distill-Qwen-1.5B: hidden_size=1536; d is read off the
# pooled embeddings automatically, this knob is only the COUNT of directions).
# Set to d=1536 so the directions match the embedding dimensionality (one per
# hidden dim) — enough to span the space while keeping the statistic stable.
# More directions = sharper, lower-variance anti-collapse gradient (LeJEPA default
# 1024 is calibrated for large from-scratch batches; our per-step pool is only
# ~2*valid_pairs embeddings, so the statistic is noisy and easily outrun by the
# dense alignment gradient — keep it >= d to stabilize SIGReg's gradient direction).
N_PROJECTIONS=${N_PROJECTIONS:-4096}
# Gradient-clip max-norm for jepa_update()'s own optimizer step, tighter than the
# actor's PPO clip_grad (typically 1.0) since jepa_update shares the actor's
# optimizer/parameters but runs as its own uncoordinated backward+step.
JEPA_MAX_GRAD_NORM=${JEPA_MAX_GRAD_NORM:-0.5}
# Ramp effective jepa.alpha from 0 -> ALPHA over this many jepa_update steps
# (0 disables warmup, full alpha from step 0). Pre-clip jepa/grad_norm is
# elevated (~5) at the very start before the predictor/encoder geometry
# settles; this reduces how much weight that early noisy direction gets.
ALPHA_WARMUP_STEPS=${ALPHA_WARMUP_STEPS:-0}

SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}   # skip the step-0 pre-train validation; go straight into training
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-1}   # only the most recent checkpoint is kept on disk

# Validation rollout
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-16}    # matches drgrpo_trainer's VAL_ROLLOUT_N
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-True}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.6}
VAL_TOP_P=${VAL_TOP_P:-0.95}
# data_source names (see verl/experimental/fepo/data.py) used to compute the average
# accuracy that decides whether to overwrite the `best/` checkpoint. Must be a subset
# of whatever's actually in VAL_FILES, or best-checkpoint tracking silently no-ops.
BEST_CKPT_SOURCES=${BEST_CKPT_SOURCES:-'["aime24","aime25","aime26","amc23"]'}

# KL penalty
USE_KL_LOSS=${USE_KL_LOSS:-false}   # anchors policy to the reference model; false -> drop KL entirely
KL_COEF=${KL_COEF:-0.001}   # NOTE: 0.0 previously caused a grad_norm/ppo_kl blowup in this exact script — watch actor/ppo_kl and actor/grad_norm closely
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
    actor_rollout_ref.model.enable_gradient_checkpointing=${GRAD_CKPT}
    +actor_rollout_ref.model.override_config.attn_implementation=${ACTOR_ATTENTION_IMPL}
)

if [[ "${USE_LORA}" == "true" ]]; then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
    )
else
    # The base config defaults lora_rank to 128; verl only treats LoRA as
    # disabled when lora_rank <= 0, so it must be explicitly zeroed out here.
    MODEL+=(
        actor_rollout_ref.model.lora_rank=0
    )
fi

ACTOR=(
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla
    # FEPO-style: token-mean for per-token gradient signals
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.clip_ratio=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO}
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN}
    actor_rollout_ref.actor.use_dynamic_bsz=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
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
    jepa.sigreg_lambda=${SIGREG_LAMBDA}
    jepa.triplet_sigreg_lambda=${TRIPLET_SIGREG_LAMBDA}
    jepa.separation_margin=${SEPARATION_MARGIN}
    jepa.separation_w=${SEPARATION_W}
    jepa.separation_tau=${SEPARATION_TAU}
    jepa.separation_mode=${SEPARATION_MODE}
    jepa.n_projections=${N_PROJECTIONS}
    jepa.alpha_warmup_steps=${ALPHA_WARMUP_STEPS}
    jepa.max_grad_norm=${JEPA_MAX_GRAD_NORM}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.nnodes=${NNODES}
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.total_training_steps=${MAX_OPTIMIZER_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${CKPTS_DIR}
    trainer.logger=${LOGGER}
    +trainer.best_ckpt_sources=${BEST_CKPT_SOURCES}
)

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.norm_adv_by_std_in_grpo=False
    algorithm.use_kl_in_reward=False
    actor_rollout_ref.actor.use_kl_loss=${USE_KL_LOSS}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.kl_loss_coef=${KL_COEF}
)

# Reuses the 1.5B-scale Qwen2.5 hyperparameter config; actor_rollout_ref.model.path
# below overrides its default checkpoint with the DeepSeek-R1-Distill-Qwen-1.5B path.
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
