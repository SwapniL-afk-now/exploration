#!/usr/bin/env bash
set -euo pipefail

python3 -m verl.experimental.fepo.main_fepo \
  --config-name fepo_trainer \
  train_dataset=sungyub/skywork-or1-math-verl \
  train_dataset_config=null \
  train_split=train \
  output_dir=checkpoints/fepo/skywork/qwen25_1_5b \
  experiment_name=qwen25-1.5b-fepo-skywork-single-gpu \
  "$@"
