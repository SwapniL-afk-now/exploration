#!/usr/bin/env bash
set -euo pipefail

python3 -m verl.experimental.fepo.main_fepo \
  --config-name fepo_qwen25_7b \
  train_dataset=zhuzilin/dapo-math-17k \
  train_dataset_config=default \
  train_split=train \
  output_dir=checkpoints/fepo/dapo/qwen25_7b \
  experiment_name=qwen25-7b-fepo-dapo-single-gpu \
  "$@"
