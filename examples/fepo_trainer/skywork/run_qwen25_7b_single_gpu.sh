#!/usr/bin/env bash
set -euo pipefail

python3 -m verl.experimental.fepo.main_fepo \
  --config-name fepo_qwen25_7b \
  train_dataset=sungyub/skywork-or1-math-verl \
  train_dataset_config=null \
  train_split=train \
  output_dir=checkpoints/fepo/skywork/qwen25_7b \
  experiment_name=qwen25-7b-fepo-skywork-single-gpu \
  "$@"
