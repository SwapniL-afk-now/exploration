#!/usr/bin/env bash
# Full eval sweep, sequential (each python frees the GPU on exit before the next starts).
# Code benchmarks + remaining math benchmarks, both checkpoints, 3 seeds, n=8.
cd /workspace/exploration
SEEDS=1234,2025,7777
DAPO=/workspace/exploration/ds_best_final_hf
OURS=/workspace/exploration/best_final_hf
LOG=/workspace/exploration/eval_sweep_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1

wait_gpu () {
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 5; done
}

run () {  # $1=label $2=script $3=model $4=benchmarks $5=out
  echo; echo "############ $(date '+%F %T')  $1 ############"
  wait_gpu
  python3 "$2" --model "$3" --benchmarks "$4" --seeds "$SEEDS" --n 8 --out "$5" \
    | grep -E "seed |loaded|MACRO|saved|rror|Traceback"
}

echo "=== EVAL SWEEP START $(date '+%F %T') ==="

# 1) Code benchmarks --------------------------------------------------------
run "CODE / DAPO" eval_code_benchmarks.py "$DAPO" "humanevalplus,mbppplus,livecodebench" dapo_code_eval_n8.json
run "CODE / OURS" eval_code_benchmarks.py "$OURS" "humanevalplus,mbppplus,livecodebench" jepa_code_eval_n8.json

# 2) Remaining math benchmarks (not yet evaluated) --------------------------
run "MATH-EXTRA / DAPO" eval_math_benchmarks.py "$DAPO" "math500,minervamath,olympiadbench" dapo_mathextra_eval_n8.json
run "MATH-EXTRA / OURS" eval_math_benchmarks.py "$OURS" "math500,minervamath,olympiadbench" jepa_mathextra_eval_n8.json

echo; echo "=== EVAL SWEEP DONE $(date '+%F %T') ==="
echo "outputs: dapo_code_eval_n8.json jepa_code_eval_n8.json dapo_mathextra_eval_n8.json jepa_mathextra_eval_n8.json"
