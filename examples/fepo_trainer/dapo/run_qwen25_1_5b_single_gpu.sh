#!/usr/bin/env bash
set -euo pipefail

python3 -m verl.experimental.fepo.main_fepo \
  --config-name fepo_trainer \
  train_dataset=zhuzilin/dapo-math-17k \
  train_dataset_config=default \
  train_split=train \
  output_dir=checkpoints/fepo/dapo/qwen25_1_5b \
  experiment_name=qwen25-1.5b-fepo-dapo-single-gpu \
  "$@"
