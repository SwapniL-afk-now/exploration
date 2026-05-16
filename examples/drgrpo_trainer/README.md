# Dr.GRPO Trainer

Dr.GRPO is a GRPO baseline that removes the standard-deviation normalization from group advantages and uses constant response-length normalization for the sequence loss. In this repo it runs through the standard `verl.trainer.main_ppo` path; FEPO/failure-escape remains separate under `verl.experimental.fepo`.

## Key Overrides

Each launcher applies the Dr.GRPO settings on top of the normal GRPO recipe:

```bash
algorithm.adv_estimator=grpo
algorithm.norm_adv_by_std_in_grpo=False
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm
actor_rollout_ref.actor.loss_scale_factor=${LOSS_SCALE_FACTOR}
actor_rollout_ref.actor.use_kl_loss=False
actor_rollout_ref.rollout.n=${ROLLOUT_N}
```

`LOSS_SCALE_FACTOR` defaults to `MAX_RESPONSE_LENGTH` so the normalization stays constant throughout training. `ROLLOUT_N` defaults to `8` to match the FEPO `num_generations=8` comparison setup.

## Launchers

| Script | Default model | Batch defaults |
| --- | --- | --- |
| `run_qwen25_1_5b_fsdp.sh` | `Qwen/Qwen2.5-1.5B-Instruct` | `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=8` |
| `run_qwen25_3b_fsdp.sh` | `Qwen/Qwen2.5-3B-Instruct` | `TRAIN_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=4` |
| `run_qwen25_7b_fsdp.sh` | `Qwen/Qwen2.5-7B-Instruct` | `TRAIN_BATCH_SIZE=4`, `PPO_MINI_BATCH_SIZE=2` |
| `run_qwen25_math_1_5b_fsdp.sh` | `Qwen/Qwen2.5-Math-1.5B-Instruct` | `TRAIN_BATCH_SIZE=16`, `PPO_MINI_BATCH_SIZE=8` |
| `run_qwen25_math_3b_fsdp.sh` | `Qwen/Qwen2.5-Math-3B-Instruct` | `TRAIN_BATCH_SIZE=8`, `PPO_MINI_BATCH_SIZE=4` |
| `run_qwen25_math_7b_fsdp.sh` | `Qwen/Qwen2.5-Math-7B-Instruct` | `TRAIN_BATCH_SIZE=4`, `PPO_MINI_BATCH_SIZE=2` |

The defaults target a single RTX PRO 6000 Blackwell-class 96GB GPU. Override any path or training knob through environment variables.

Dataset defaults follow the standard trainer parquet interface:

```bash
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/dapo-math-17k/train.parquet}
VAL_FILE=${VAL_FILE:-$HOME/data/aime-2024/test.parquet}
```

Use `VAL_FILE` to point at AMC23, MATH500, or any other prepared parquet evaluation set.

## Tiny Smoke Run

```bash
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=4 \
TOTAL_EPOCHS=1 \
TEST_FREQ=1 \
examples/drgrpo_trainer/run_qwen25_math_1_5b_fsdp.sh
```

## Recommended Comparison Order

```text
1. GRPO baseline
2. Dr.GRPO baseline
3. FEPO / failure-escape
```
