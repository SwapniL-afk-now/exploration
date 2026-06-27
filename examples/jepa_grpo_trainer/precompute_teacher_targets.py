#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline teacher-target precompute for jepa.loss_type='jepa-tcr-loss'.

Phase 1 of the two-phase TCR workflow (run this to completion BEFORE training):

Supports two views via --view (run once per view to build both caches):
  - cot  (default): a stronger math TEACHER generates step-by-step solutions.
  - code:           a CODER model generates executable Python under the code
                    system prompt (mirrors JEPARayConfig.code_system_prompt);
                    used by jepa-tcr-dual as the Code-view target cache.
Both views verify with the SAME parse-based checker training uses
(compute_math_reward) and encode with the SAME frozen student-size reference, so
both caches live in one shared 1536-d student space.

  1. A stronger TEACHER / CODER model generates K solutions per training question.
  2. Each is VERIFIED with the same checker used in training
     (verl.experimental.fepo.math_parser.compute_math_reward); only correct ones
     are kept (up to --n-targets per question).
  3. Each kept teacher-correct *text* [x, y_T+] is ENCODED by a frozen
     STUDENT-SIZE reference model (e.g. Qwen2.5-Math-1.5B): final-layer hidden
     state at the last real token, L2-normalized -> a d-dim vector in the
     student's own representation space (so training needs NO projector and no
     cross-dimension cosine).
  4. The per-question targets are SAVED as a torch dict
     {dataset_index (int): float16 tensor (n_i, d)} consumed at training time by
     ray_trainer._build_jepa_batch_tcr via jepa.teacher_cache_path.

The teacher model is only loaded here; it is NOT loaded during training.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.experimental.fepo.math_parser import compute_math_reward

# Default Code-view system prompt. Mirrors JEPARayConfig.code_system_prompt so the
# code-view targets are generated under the SAME framing the train-time code
# rollouts use (keep these two in sync).
CODE_SYSTEM_PROMPT = (
    "You are a Python programming expert. "
    "Solve the following math problem by writing a complete, executable Python program "
    "that prints the answer. Do not include any natural language explanation outside comments."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-file", required=True, help="Adapted train parquet (prompt/extra_info/reward_model columns)")
    p.add_argument(
        "--view", choices=("cot", "code"), default="cot",
        help="Target view to build. 'cot' (default): generate step-by-step CoT solutions "
             "with the dataset's own prompt (current behavior). 'code': prepend the code "
             "system prompt so a CODER model emits executable Python; produces a code-view "
             "target cache. Run once per view to build both caches.",
    )
    p.add_argument(
        "--system-prompt", default="",
        help="Override the system prompt prepended to every question. Empty (default) uses "
             "no override for --view cot, and CODE_SYSTEM_PROMPT for --view code.",
    )
    p.add_argument("--teacher-model", "--gen-model", dest="teacher_model", required=True,
                   help="HF path of the generator: stronger math teacher for --view cot, "
                        "or a coder model (e.g. Qwen2.5-Coder-*) for --view code")
    p.add_argument("--ref-model", required=True, help="HF path of the frozen student-size reference encoder")
    p.add_argument("--out", required=True, help="Output .pt path for the {index: (n_i,d)} target cache")
    p.add_argument("--n-samples", type=int, default=8, help="Teacher samples generated per question")
    p.add_argument("--n-targets", type=int, default=4, help="Max correct targets kept per question")
    p.add_argument("--max-rows", type=int, default=-1, help="Cap on questions processed (-1 = all)")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=3072)
    p.add_argument("--tp-size", type=int, default=1, help="vLLM tensor-parallel size for the teacher")
    p.add_argument("--gpu-mem-frac", type=float, default=0.85)
    p.add_argument("--encode-batch-size", type=int, default=16)
    p.add_argument("--gen-batch-prompts", type=int, default=64, help="Prompts per vLLM generate() call")
    p.add_argument("--save-responses", default="", help="Optional .pt path to also save {index: [correct_texts]}")
    return p.parse_args()


def _row_messages(prompt, system_prompt: str = "") -> list[dict]:
    # Adapted rows store `prompt` as a list of {role, content} dicts (data.make_messages).
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt]
    if system_prompt:
        # Prepend the view's system prompt as a leading system turn (code view). If the
        # row already carries a system turn, this stacks before it so the code framing
        # takes precedence without dropping the dataset's own instructions.
        msgs = [{"role": "system", "content": system_prompt}] + msgs
    return msgs


@torch.no_grad()
def encode_targets(
    ref_model, ref_tok, prompt_text: str, responses: list[str], device, batch_size: int = 16
) -> torch.Tensor:
    """Encode [x, y_T+] with the frozen ref model; last-token final hidden, L2-normalized.

    Mirrors worker._extract_embeddings (last real token of the final layer norm,
    then F.normalize) so offline targets live in the same space as the student
    preds produced online.
    """
    embs: list[torch.Tensor] = []
    for i in range(0, len(responses), batch_size):
        chunk = responses[i : i + batch_size]
        texts = [prompt_text + r for r in chunk]
        enc = ref_tok(texts, return_tensors="pt", padding=True, truncation=False).to(device)
        out = ref_model(**enc, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]  # (B, L, d)
        # Last real token index per row (right padding -> sum of attention mask - 1;
        # left padding -> always the final column). Handle both via attention mask.
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1  # works for right padding
        # If the tokenizer left-pads, the real tokens end at column L-1.
        if ref_tok.padding_side == "left":
            last_idx = torch.full_like(last_idx, last_hidden.shape[1] - 1)
        rows = last_hidden[torch.arange(last_hidden.shape[0], device=device), last_idx]
        embs.append(F.normalize(rows.float(), dim=-1).cpu())
    return torch.cat(embs, dim=0)


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.train_file)
    if args.max_rows > 0:
        df = df.iloc[: args.max_rows]

    # System prompt prepended to every question. For --view code this is the code
    # framing (mirrors JEPARayConfig.code_system_prompt) so the generator emits
    # executable Python; for --view cot it stays empty (dataset prompt untouched).
    system_prompt = args.system_prompt or (CODE_SYSTEM_PROMPT if args.view == "code" else "")
    print(f"[precompute] view={args.view!r}; system_prompt={'<custom>' if args.system_prompt else ('code' if system_prompt else 'none')}", flush=True)

    # ---- Phase 1a: teacher generation (vLLM) ----
    from vllm import LLM, SamplingParams

    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    teacher = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tp_size,
        gpu_memory_utilization=args.gpu_mem_frac,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    rows = df.to_dict("records")
    # Each row's `prompt` carries the dataset's step-by-step \boxed{} framing
    # (fepo.data.make_messages). For --view code we additionally prepend the code
    # system prompt so the coder model emits an executable Python program.
    prompt_texts = [
        teacher_tok.apply_chat_template(
            _row_messages(r["prompt"], system_prompt), tokenize=False, add_generation_prompt=True
        )
        for r in rows
    ]
    # Generate in fixed prompt batches (mirrors the train-time batching of 64).
    gen = []
    bsz = max(1, args.gen_batch_prompts)
    for i in range(0, len(prompt_texts), bsz):
        chunk = prompt_texts[i : i + bsz]
        gen.extend(teacher.generate(chunk, sampling))
        print(f"[precompute] generated {min(i + bsz, len(prompt_texts))}/{len(prompt_texts)} prompts", flush=True)

    # ---- Phase 1b: verify -> keep correct teacher-correct responses ----
    kept: list[tuple[int, str, list[str]]] = []  # (index, ref_prompt_text, correct_responses)
    n_correct_total = 0
    for r, g in zip(rows, gen):
        idx = int(r["extra_info"]["index"])
        ground_truth = r["reward_model"]["ground_truth"]
        dataset_kind = r.get("data_source")
        correct: list[str] = []
        for o in g.outputs:
            text = o.text
            res = compute_math_reward(text, ground_truth, dataset_kind=dataset_kind)
            if res.is_correct:
                correct.append(text)
                if len(correct) >= args.n_targets:
                    break
        if correct:
            n_correct_total += len(correct)
            kept.append((idx, r["prompt"], correct))

    print(f"[precompute] {len(kept)}/{len(rows)} questions have >=1 correct teacher response "
          f"({n_correct_total} target texts total)")

    # Free the teacher before loading the reference encoder.
    del teacher
    torch.cuda.empty_cache()

    # ---- Phase 1c: encode with the frozen student-size reference model ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_tok = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
    if ref_tok.pad_token_id is None:
        ref_tok.pad_token = ref_tok.eos_token
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    cache: dict[int, torch.Tensor] = {}
    responses_by_idx: dict[int, list[str]] = {}
    for idx, prompt, responses in kept:
        prompt_text = ref_tok.apply_chat_template(
            _row_messages(prompt, system_prompt), tokenize=False, add_generation_prompt=True
        )
        targets = encode_targets(
            ref_model, ref_tok, prompt_text, responses, device, args.encode_batch_size
        )
        cache[idx] = targets.half()
        responses_by_idx[idx] = responses

    torch.save(cache, args.out)
    print(f"[precompute] wrote {len(cache)} questions -> {args.out} "
          f"(d={next(iter(cache.values())).shape[-1] if cache else 'n/a'})")

    # Persist the raw teacher-correct response texts (default: alongside the target cache).
    resp_path = args.save_responses or (args.out + ".responses.pt")
    torch.save(responses_by_idx, resp_path)
    print(f"[precompute] wrote correct response texts for {len(responses_by_idx)} questions -> {resp_path}")


if __name__ == "__main__":
    main()
