"""LiveCodeBench (code generation) pass@k for a local HF checkpoint.

Standalone runner that bypasses the lcb_runner CLI (its HF script-dataset loader is
incompatible with datasets>=3) but REUSES lcb_runner's problem parsing, official
prompt template, code extraction, and sandboxed grader so the numbers stay faithful.

Reads the release jsonl directly from the cloned LiveCodeBench repo's data dir (or a
HuggingFace-downloaded copy). Driven by env vars set in eval_coding_benchmarks.sh:

  MODEL_PATH       merged HF checkpoint dir
  LCB_REPO         path to a `git clone` of LiveCodeBench (provides lcb_runner + few-shot json)
  LCB_JSONL        path to release jsonl (e.g. test5.jsonl == release_v5 delta)
  N_SAMPLES        samples per problem (default 8)
  TEMPERATURE      sampling temperature (default 0.6)
  TOP_P            nucleus top-p (default 0.95)
  MAX_TOKENS       generation cap (default 2048)
  OUT_JSON         where to dump the full metrics dict
"""
import json, os, sys

MODEL = os.environ["MODEL_PATH"]
LCB_REPO = os.environ["LCB_REPO"]
LCB_JSONL = os.environ["LCB_JSONL"]
N = int(os.environ.get("N_SAMPLES", "8"))
TEMP = float(os.environ.get("TEMPERATURE", "0.6"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAXTOK = int(os.environ.get("MAX_TOKENS", "2048"))
OUT_JSON = os.environ.get("OUT_JSON", "lcb_metrics.json")

sys.path.insert(0, LCB_REPO)
os.chdir(LCB_REPO)  # prompt module reads a relative few-shot json at import time

from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.prompts.code_generation import (
    get_generic_question_template_answer, PromptConstants,
)
from lcb_runner.utils.extraction_utils import extract_code
from lcb_runner.lm_styles import LMStyle
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics

STYLE = LMStyle.CodeQwenInstruct  # chat style -> extractor grabs the last ``` block

import re

def extract_code_robust(text: str) -> str:
    """Prefer the actual python solution block.

    Math-tuned models often append a ```output ...``` block (and a \\boxed{} line)
    AFTER the real ```python ...``` block. LCB's default extractor takes the last two
    fences, so it wrongly grabs the printed result instead of the program. Here we keep
    real code blocks (drop output/result/text fences) and prefer one that looks like
    code (def / import / input( / print(). Falls back to the stock extractor.
    """
    blocks = re.findall(r"```(\w*)[ \t]*\n(.*?)```", text, re.DOTALL)
    if not blocks:
        return extract_code(text, STYLE)
    NONCODE = {"output", "text", "result", "results", "stdout", "console", ""}
    cand = [body for lang, body in blocks if lang.lower() not in NONCODE]
    if not cand:  # only bare/output fences -> keep non-output ones at least
        cand = [body for lang, body in blocks if lang.lower() not in {"output", "result", "results", "stdout"}]
    if not cand:
        cand = [body for _lang, body in blocks]
    code_like = [b for b in cand if ("def " in b or "input(" in b or "import " in b or "print(" in b)]
    pick = code_like or cand
    return pick[-1].strip()


def main():
    with open(LCB_JSONL) as f:
        problems = [CodeGenerationProblem(**json.loads(l)) for l in f]
    print(f"[lcb] {len(problems)} problems from {os.path.basename(LCB_JSONL)}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    prompts = []
    for p in problems:
        msgs = [
            {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
            {"role": "user", "content": get_generic_question_template_answer(p)},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL, dtype="bfloat16", max_model_len=4096,
              gpu_memory_utilization=0.85, trust_remote_code=True)
    sp = SamplingParams(n=N, temperature=TEMP, top_p=TOP_P, max_tokens=MAXTOK)
    outs = llm.generate(prompts, sp)

    gens = [[extract_code_robust(c.text) for c in o.outputs] for o in outs]
    empty = sum(1 for g in gens for c in g if not c.strip())
    print(f"[lcb] empty extractions: {empty}/{sum(len(g) for g in gens)}", flush=True)
    samples = [p.get_evaluation_sample() for p in problems]
    metrics, _results, _meta = codegen_metrics(
        samples, gens, k_list=[1, N], num_process_evaluate=16, timeout=6,
    )
    passk = {k: metrics[k] for k in metrics if k.startswith("pass@")}
    print(f"[lcb] {passk}", flush=True)
    with open(OUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print("LCB_DONE", flush=True)


if __name__ == "__main__":
    main()
