#!/usr/bin/env bash
cd /workspace/exploration
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 5000 ]; do sleep 10; done
echo "=== DIAG START $(date '+%T') ==="
python3 diag_code_trunc.py --model /workspace/exploration/ds_best_final_hf 2>/dev/null
python3 diag_code_trunc.py --model /workspace/exploration/best_final_hf 2>/dev/null
echo "=== DIAG DONE $(date '+%T') ==="
