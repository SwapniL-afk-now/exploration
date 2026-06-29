# Working with this user

## Who / what this repo is
RL post-training research (verl, FSDP) on Qwen2.5-Math-1.5B-Instruct: DAPO baseline vs. the
"ours" JEPA / TCR-dual auxiliary objective. Outputs feed an ICLR paper at
`/workspace/paper/iclr2025_conference.tex`. Main work lives in `/workspace/exploration`.

## How the user likes to work
- **Run long jobs in tmux**, never blocking foreground. They detach and reattach. For
  training use the `jepa` tmux session specifically. Status checks should be quick: capture
  the pane + GPU (`nvidia-smi`) and report progress/ETA tersely.
- **Single GPU** — schedule jobs sequentially; each python frees the GPU on exit. Use a
  `wait_gpu()` poll (memory.used < 5000 MiB) between steps.
- **Minimize permission prompts.** They've allowed broad Bash incl. `rm -rf *` in
  `~/.claude/settings.json`. Settings changes only take effect next session, so until then
  hand them the exact command to run via `!` or terminal.
- **Be direct and concise.** Short answers, tables for results, no preamble/filler. They ask
  "whats the status" / "whats the space" often — answer with numbers.

## Evaluation conventions (match these exactly)
- Always use **3 fixed seeds** (default `1234,2025,7777`; AIME26-ours used `42/99/2024`).
- Report **mean ± std** across seeds. In LaTeX: `$VAL${\scriptsize$\,\pm X.X$}`.
- Default rollout budget **n=8** (sometimes n=16). `pass@16` is meaningless at n=8 (always 1.0)
  — only report pass@8 and avg@1 there.
- Math benchmarks: amc23, aime24/25/26 (+ math500, minervamath, olympiadbench as extended).
  Code: humanevalplus, mbppplus, livecodebench. Model context is 4096 → use max_tokens 3072.
- Eval scripts: `eval_math_benchmarks.py`, `eval_code_benchmarks.py`, fast judge
  `code_judge_worker.py` (one subprocess per program, SIGALRM timeout — don't regress to
  verl's per-test subprocess scorer, it's slow).

## What the user values (learned from corrections)
- **Honesty + root-cause over acceptance.** When a result looks wrong they push back ("this
  is not possible, ours is actually better") — that means *investigate the mechanism*
  (e.g. truncation of verbose CoT before the code fence closes), don't just accept or bury it.
  Report failures and caveats plainly.
- Put results into the paper **"correctly"** — exact numbers, right seeds, honest framing.
  When they say don't average old+new seeds, keep only the specified set.
- After a run: **push artifacts to HuggingFace, then clean local disk.** See
  memory `reference_hf_push.md` (token in `/workspace/.env`, repos: framework /
  dapo-baseline / framework-teacher-cache).

## Gotchas
- verl model_merger writes `adapter_config.json` with `lora_alpha=0`; set it to `1024` before
  PEFT `merge_and_unload` or the LoRA contribution is zeroed.
- vLLM v1 needs eval scripts wrapped in `if __name__ == "__main__": main()`.
- Reward import is `from verl.utils.reward_score import default_compute_score`.
