#!/usr/bin/env bash
# Phase 1 of the jepa-tcr-dual workflow: generate + verify + encode teacher targets.
# Run this to completion BEFORE launching training with JEPA_LOSS_TYPE=jepa-tcr-dual;
# point that run's TEACHER_CACHE at the OUT path produced here.
set -euo pipefail

# Target view: "cot" (default, math teacher) or "code" (coder model -> code-view cache).
VIEW=${VIEW:-cot}
# Generator used ONLY to generate verified-correct solution text. For VIEW=code pass a
# coder checkpoint (e.g. Qwen2.5-Coder-*); for VIEW=cot a stronger math teacher.
TEACHER_MODEL=${TEACHER_MODEL:-/workspace/models/Qwen2.5-Math-3B-Instruct}
# Frozen student-size reference encoder (must match the TRAINING model's hidden size).
REF_MODEL=${REF_MODEL:-/workspace/models/Qwen2.5-Math-1.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/workspace/jepa-grpo-cache/data/dapo_math_17k_train.parquet}
# Default OUT differs per view so the two caches don't clobber each other.
if [[ "${VIEW}" == "code" ]]; then
  OUT=${OUT:-/workspace/jepa-grpo-cache/code_teacher_targets.pt}
else
  OUT=${OUT:-/workspace/jepa-grpo-cache/teacher_targets.pt}
fi
# Optional explicit system-prompt override (VIEW=code defaults to the code framing).
SYSTEM_PROMPT=${SYSTEM_PROMPT:-}

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
    --view "${VIEW}" \
    ${SYSTEM_PROMPT:+--system-prompt "${SYSTEM_PROMPT}"} \
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
