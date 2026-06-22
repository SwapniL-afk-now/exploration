#!/usr/bin/env bash
# Phase 1 of the jepa-tcr-loss workflow: generate + verify + encode teacher targets.
# Run this to completion BEFORE launching training with JEPA_LOSS_TYPE=jepa-tcr-loss;
# point that run's TEACHER_CACHE at the OUT path produced here.
set -euo pipefail

# Stronger teacher used ONLY to generate verified-correct solution text.
TEACHER_MODEL=${TEACHER_MODEL:-/workspace/models/Qwen2.5-Math-3B-Instruct}
# Frozen student-size reference encoder (must match the TRAINING model's hidden size).
REF_MODEL=${REF_MODEL:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
OUT=${OUT:-/workspace/jepa-grpo-cache/teacher_targets.pt}

N_SAMPLES=${N_SAMPLES:-8}
N_TARGETS=${N_TARGETS:-4}
MAX_ROWS=${MAX_ROWS:--1}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.95}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
TP_SIZE=${TP_SIZE:-1}
GPU_MEM_FRAC=${GPU_MEM_FRAC:-0.85}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-16}

python3 "$(dirname "$0")/precompute_teacher_targets.py" \
    --train-file "${TRAIN_FILE}" \
    --teacher-model "${TEACHER_MODEL}" \
    --ref-model "${REF_MODEL}" \
    --out "${OUT}" \
    --n-samples "${N_SAMPLES}" \
    --n-targets "${N_TARGETS}" \
    --max-rows "${MAX_ROWS}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --tp-size "${TP_SIZE}" \
    --gpu-mem-frac "${GPU_MEM_FRAC}" \
    --encode-batch-size "${ENCODE_BATCH_SIZE}"
