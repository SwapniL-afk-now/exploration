# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""Representation-quality analysis: does the JEPA objective actually do anything
to the representations, vs. a Dr.GRPO baseline trained with no JEPA loss?

Produces the three reviewer-facing artifacts:
  1. Mean cosine similarity between Enc(CoT) and Enc(Code), tracked across
     training steps, one line per run. The JEPA run should converge upward;
     a Dr.GRPO baseline (never optimizes this) should stay near its random
     initial value.
  2. Singular value spectrum of the (N, d) matrix [Enc(CoT)_i - Enc(Code)_i]_i
     over the probe set, at the final step. A JEPA-aligned run should show a
     few dominant singular values (low-rank structure); an unaligned baseline
     should show a flatter, higher-rank spectrum (echoes LLM-JEPA Fig. 3).
  3. (Optional, --tsne) 2D t-SNE of paired CoT/Code embeddings at each step,
     one subplot per (run, step). JEPA pairs should cluster tightly by
     problem; baseline pairs should show no such structure.

Embedding convention matches `JEPAActorRolloutRefWorker._extract_embeddings`
exactly: the L2-normalized hidden state at the last real token, captured via a
forward hook on the model's final norm submodule (not the LM head) — so this
analysis measures precisely the representations the JEPA loss itself operates
on, not some other notion of "embedding".

Checkpoint resolution:
  --steps 0   always uses --base-model directly (untrained reference point).
  --steps N>0 looks for <run_dir>/global_step_<N>/actor and merges it via
              `python -m verl.model_merger` into a plain HF directory (cached
              under .../actor/_merged_hf so repeat runs skip the merge).

Example:
    python -m verl.experimental.jepa_grpo.analysis.representation_quality \
        --run jepa=checkpoints/verl_drgrpo_dapo_math/<jepa_exp_name> \
        --run baseline=checkpoints/verl_drgrpo_dapo_math/<drgrpo_exp_name> \
        --steps 0,200,400 \
        --base-model /workspace/models/DeepSeek-R1-Distill-Qwen-1.5B \
        --probe-file /workspace/jepa-grpo-cache/eval_data/math500.parquet \
        --num-probes 32 \
        --output-dir /workspace/analysis/representation_quality \
        --tsne

Requires (only for the plotting step; embedding extraction needs none of
these beyond what verl already depends on): `pip install matplotlib scikit-learn`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Prompt construction — matches verl.experimental.fepo.data (CoT view, same
# convention used to build the training/eval parquets) and the Ray trainer's
# default jepa.code_system_prompt (see config/jepa_grpo_ray_trainer.yaml).
# Override via --cot-system-prompt/--code-system-prompt if your run customized them.
# ---------------------------------------------------------------------------

DEFAULT_COT_SYSTEM_PROMPT = (
    "You are a careful mathematical problem solver. "
    "Reason step by step and put the final answer in \\boxed{}."
)
DEFAULT_COT_USER_SUFFIX = "Solve the problem. Show your reasoning and put the final answer in \\boxed{}."
DEFAULT_CODE_SYSTEM_PROMPT = (
    "You are a Python programming expert. Solve the following math problem by "
    "writing a complete, executable Python program that prints the answer."
)


def build_messages(problem: str, view: str, args: argparse.Namespace) -> list[dict[str, str]]:
    if view == "cot":
        return [
            {"role": "system", "content": args.cot_system_prompt},
            {"role": "user", "content": f"{problem}\n{args.cot_user_suffix}"},
        ]
    elif view == "code":
        return [
            {"role": "system", "content": args.code_system_prompt},
            {"role": "user", "content": problem},
        ]
    raise ValueError(f"unknown view: {view}")


# ---------------------------------------------------------------------------
# Probe set
# ---------------------------------------------------------------------------


def load_probes(parquet_path: str, n: int, seed: int) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    problems = []
    for extra_info in df["extra_info"]:
        problem = extra_info.get("problem") if isinstance(extra_info, dict) else None
        if problem:
            problems.append(str(problem))
    if not problems:
        raise ValueError(f"No extra_info.problem field found in {parquet_path}")
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(problems), size=min(n, len(problems)), replace=False)
    return [problems[i] for i in sorted(idx)]


# ---------------------------------------------------------------------------
# Checkpoint resolution
# ---------------------------------------------------------------------------


def resolve_checkpoint(run_dir: str, step: int, base_model: str) -> str:
    if step == 0:
        return base_model

    actor_dir = os.path.join(run_dir, f"global_step_{step}", "actor")
    if not os.path.isdir(actor_dir):
        raise FileNotFoundError(f"Expected verl FSDP checkpoint at {actor_dir}")

    merged_dir = os.path.join(actor_dir, "_merged_hf")
    if os.path.isdir(merged_dir) and os.path.exists(os.path.join(merged_dir, "config.json")):
        return merged_dir

    print(f"[representation_quality] merging {actor_dir} -> {merged_dir}", flush=True)
    subprocess.run(
        [
            sys.executable, "-m", "verl.model_merger", "merge",
            "--backend", "fsdp",
            "--local_dir", actor_dir,
            "--target_dir", merged_dir,
        ],
        check=True,
    )
    return merged_dir


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def _final_norm_module(model):
    """Best-effort lookup of the model's final pre-LM-head norm, matching
    JEPAActorRolloutRefWorker._final_norm() for Qwen2/Llama-family architectures
    (DeepSeek-R1-Distill-Qwen is Qwen2-based)."""
    for path in ("model.norm", "norm"):
        obj = model
        ok = True
        for part in path.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise AttributeError(
        f"Could not find a final-norm submodule on {type(model).__name__}; "
        "pass a custom lookup if this architecture differs from Qwen2/Llama."
    )


@contextlib.contextmanager
def _capture_final_norm_output(model):
    captured: dict[str, "object"] = {}

    def _hook(module, inp, out):
        captured["h"] = out[0] if isinstance(out, tuple) else out

    handle = _final_norm_module(model).register_forward_hook(_hook)
    try:
        yield captured
    finally:
        handle.remove()


def extract_embeddings(
    model, tokenizer, problems: list[str], view: str, args: argparse.Namespace,
) -> np.ndarray:
    """Generate a response for each problem under `view`, then return the
    (N, d) matrix of L2-normalized last-real-token embeddings of (prompt +
    response), exactly as JEPAActorRolloutRefWorker._extract_embeddings reads
    them during training."""
    import torch

    # Processed one prompt at a time (no padding) rather than batched: HF's
    # generate() needs left-padding for correct batched decoding, and
    # reconstructing a post-hoc attention mask from token equality is
    # unreliable here since pad_token_id falls back to eos_token_id (a real
    # EOS in the response would be misread as padding). Unbatched generation
    # sidesteps the whole class of bugs at the cost of throughput, which is an
    # acceptable trade for an offline analysis tool over a small probe set.
    device = next(model.parameters()).device
    embs = []
    for problem in problems:
        text = tokenizer.apply_chat_template(
            build_messages(problem, view, args), tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                         max_length=args.max_prompt_length).to(device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        attn = torch.ones_like(gen)
        with torch.no_grad(), _capture_final_norm_output(model) as captured:
            model(input_ids=gen, attention_mask=attn, use_cache=False)

        last_h = captured["h"].float()  # (1, L, d)
        vec = torch.nn.functional.normalize(last_h[0, -1, :], dim=-1)
        embs.append(vec.cpu().numpy())
    return np.stack(embs, axis=0)


def load_model_and_tokenizer(path: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # unused at batch size 1, but the correct default if ever batched
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, device_map="cuda")
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity for two (N, d) L2-normalized matrices."""
    return np.sum(a * b, axis=-1)


def svd_spectrum(diff: np.ndarray) -> np.ndarray:
    """Singular values of the (N, d) difference matrix, descending."""
    _, s, _ = np.linalg.svd(diff, full_matrices=False)
    return s


# ---------------------------------------------------------------------------
# Plotting (lazy imports so the extraction path has no hard dependency on
# matplotlib/scikit-learn)
# ---------------------------------------------------------------------------


def plot_cosine_curve(records: list[dict], output_path: str) -> None:
    import matplotlib.pyplot as plt

    runs = sorted({r["run"] for r in records})
    plt.figure(figsize=(6, 4))
    for run in runs:
        pts = sorted([(r["step"], r["cos_sim_mean"]) for r in records if r["run"] == run])
        xs, ys = zip(*pts)
        plt.plot(xs, ys, marker="o", label=run)
    plt.xlabel("Training step")
    plt.ylabel("Mean cosine similarity, Enc(CoT) vs Enc(Code)")
    plt.title("Representation alignment over training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_svd_spectrum(spectra: dict[str, np.ndarray], output_path: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for run, s in spectra.items():
        plt.plot(np.arange(1, len(s) + 1), s, marker="o", label=run)
    plt.yscale("log")
    plt.xlabel("Singular value index")
    plt.ylabel("Singular value (log scale)")
    plt.title("Spectrum of Enc(CoT) - Enc(Code) at final step")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_tsne_grid(
    embeddings: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], output_path: str,
) -> None:
    """embeddings[(run, step)] = (enc_cot (N,d), enc_code (N,d))."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    keys = sorted(embeddings.keys(), key=lambda k: (k[0], k[1]))
    runs = sorted({k[0] for k in keys})
    steps = sorted({k[1] for k in keys})

    fig, axes = plt.subplots(len(steps), len(runs), figsize=(4 * len(runs), 4 * len(steps)), squeeze=False)
    for si, step in enumerate(steps):
        for ri, run in enumerate(runs):
            ax = axes[si][ri]
            key = (run, step)
            if key not in embeddings:
                ax.axis("off")
                continue
            enc_cot, enc_code = embeddings[key]
            n = enc_cot.shape[0]
            combined = np.concatenate([enc_cot, enc_code], axis=0)
            proj = TSNE(n_components=2, init="pca", random_state=0,
                        perplexity=min(30, max(2, n // 3))).fit_transform(combined)
            colors = np.arange(n)
            ax.scatter(proj[:n, 0], proj[:n, 1], c=colors, cmap="tab20", marker="o", label="CoT", alpha=0.8)
            ax.scatter(proj[n:, 0], proj[n:, 1], c=colors, cmap="tab20", marker="x", label="Code", alpha=0.8)
            ax.set_title(f"{run} @ step {step}")
            if si == 0 and ri == 0:
                ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=DIR",
        help="Repeatable. e.g. --run jepa=checkpoints/.../jepa_exp --run baseline=checkpoints/.../drgrpo_exp",
    )
    parser.add_argument("--steps", required=True, help="Comma-separated checkpoint steps, e.g. 0,200,400")
    parser.add_argument("--base-model", required=True, help="HF path/id used for step 0 in every run")
    parser.add_argument("--probe-file", required=True, help="verl-format parquet with extra_info.problem")
    parser.add_argument("--num-probes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--cot-system-prompt", default=DEFAULT_COT_SYSTEM_PROMPT)
    parser.add_argument("--cot-user-suffix", default=DEFAULT_COT_USER_SUFFIX)
    parser.add_argument("--code-system-prompt", default=DEFAULT_CODE_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tsne", action="store_true", help="Also produce the t-SNE grid (needs scikit-learn)")
    parser.add_argument("--tsne-steps", default=None, help="Comma-separated subset of --steps to t-SNE (default: all)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    runs = dict(r.split("=", 1) for r in args.run)
    steps = [int(s) for s in args.steps.split(",")]
    tsne_steps = {int(s) for s in args.tsne_steps.split(",")} if args.tsne_steps else set(steps)

    probes = load_probes(args.probe_file, args.num_probes, args.seed)
    print(f"[representation_quality] {len(probes)} probe problems from {args.probe_file}")

    records = []
    final_diffs: dict[str, np.ndarray] = {}
    tsne_embeddings: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}

    for run_label, run_dir in runs.items():
        for step in steps:
            ckpt_path = resolve_checkpoint(run_dir, step, args.base_model)
            print(f"[representation_quality] {run_label} @ step {step}: loading {ckpt_path}", flush=True)
            model, tokenizer = load_model_and_tokenizer(ckpt_path, args.dtype)
            try:
                enc_cot = extract_embeddings(model, tokenizer, probes, "cot", args)
                enc_code = extract_embeddings(model, tokenizer, probes, "code", args)
            finally:
                import torch

                del model
                torch.cuda.empty_cache()

            sims = cosine_sim(enc_cot, enc_code)
            diff = enc_cot - enc_code
            records.append({
                "run": run_label, "step": step,
                "cos_sim_mean": float(sims.mean()), "cos_sim_std": float(sims.std()),
            })
            if step == max(steps):
                final_diffs[run_label] = diff
            if step in tsne_steps:
                tsne_embeddings[(run_label, step)] = (enc_cot, enc_code)

            np.savez(
                os.path.join(args.output_dir, f"embeddings_{run_label}_step{step}.npz"),
                enc_cot=enc_cot, enc_code=enc_code,
            )

    with open(os.path.join(args.output_dir, "cosine_similarity.json"), "w") as f:
        json.dump(records, f, indent=2)
    print(json.dumps(records, indent=2))

    plot_cosine_curve(records, os.path.join(args.output_dir, "cosine_similarity_over_training.png"))

    spectra = {run: svd_spectrum(diff) for run, diff in final_diffs.items()}
    np.savez(os.path.join(args.output_dir, "svd_spectra.npz"), **spectra)
    plot_svd_spectrum(spectra, os.path.join(args.output_dir, "svd_spectrum.png"))

    if args.tsne:
        plot_tsne_grid(tsne_embeddings, os.path.join(args.output_dir, "tsne_grid.png"))

    print(f"[representation_quality] wrote outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
