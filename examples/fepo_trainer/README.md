# Experimental FEPO/Dr.GRPO Single-GPU Trainer

This example runs the Kaggle FEPO notebook algorithm through `verl.experimental.fepo`.
It uses two named LoRA adapters:

- `solver`: synced into vLLM for generation and trained with the FEPO solver loss.
- `failure`: trained with wrong-only SFT and used as the solver escape target.

DAPO-Math-17K launchers:

```bash
examples/fepo_trainer/dapo/run_qwen25_1_5b_single_gpu.sh
examples/fepo_trainer/dapo/run_qwen25_3b_single_gpu.sh
examples/fepo_trainer/dapo/run_qwen25_7b_single_gpu.sh
```

Skywork OR1 Math launchers:

```bash
examples/fepo_trainer/skywork/run_qwen25_1_5b_single_gpu.sh
examples/fepo_trainer/skywork/run_qwen25_3b_single_gpu.sh
examples/fepo_trainer/skywork/run_qwen25_7b_single_gpu.sh
```

Each launcher accepts extra Hydra overrides. For example:

```bash
examples/fepo_trainer/skywork/run_qwen25_7b_single_gpu.sh \
  max_optimizer_steps=100
```

Checkpoints are saved after every optimizer step by default and resume with `resume_mode=auto`.
