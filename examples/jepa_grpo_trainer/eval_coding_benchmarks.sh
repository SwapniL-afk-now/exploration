#!/bin/bash
# ==============================================================================
# Coding-benchmark eval for a JEPA-GRPO checkpoint: HumanEval+, MBPP+, LiveCodeBench.
#
# Merges a verl FSDP actor checkpoint to HF format, then samples N completions per
# problem with vLLM and reports pass@1 / pass@N (default N=8). EvalPlus supplies the
# HumanEval+/MBPP+ plus-test grading; a standalone runner reuses lcb_runner's parser,
# prompt, extractor, and sandbox grader for LiveCodeBench.
#
# Usage:
#   bash eval_coding_benchmarks.sh /path/to/checkpoints/<exp>/best/actor
#   CKPT=/path/to/best/actor BENCHES="humaneval mbpp lcb" N_SAMPLES=8 bash eval_coding_benchmarks.sh
#
# Key env vars (all optional except the checkpoint):
#   CKPT             verl FSDP actor dir (has model_world_size_*.pt + huggingface/);
#                    may also be passed as $1. If it already looks like a merged HF
#                    dir (has model.safetensors), it is used directly (no merge).
#   OUT_DIR          scratch root for merged weights + results  (default below)
#   BENCHES          space-separated subset of: humaneval mbpp lcb   (default all)
#   N_SAMPLES        samples per problem                              (default 8)
#   TEMPERATURE      sampling temperature                             (default 0.6)
#   TOP_P            nucleus top-p                                    (default 0.95)
#   LCB_JSONL        LiveCodeBench release jsonl (default: release_v5 == test5.jsonl)
#   LCB_VERSION      which release file to fetch if LCB_JSONL unset   (default test5)
# ==============================================================================
set -euo pipefail

CKPT="${CKPT:-${1:-}}"
[ -z "$CKPT" ] && { echo "ERROR: pass the checkpoint dir (verl best/actor) as \$1 or CKPT="; exit 1; }

OUT_DIR="${OUT_DIR:-/workspace/scratch_ckpt/coding_eval}"
BENCHES="${BENCHES:-humaneval mbpp lcb}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
LCB_VERSION="${LCB_VERSION:-test5}"
LCB_REPO="${LCB_REPO:-$OUT_DIR/LiveCodeBench}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUT_DIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ---- 1. Merge verl FSDP checkpoint -> HF (skip if already a merged HF dir) ----
if [ -f "$CKPT/model.safetensors" ] || ls "$CKPT"/model-*.safetensors >/dev/null 2>&1; then
  MODEL_PATH="$CKPT"
  echo "[merge] $CKPT already HF format; using as-is"
else
  MODEL_PATH="$OUT_DIR/merged_hf"
  if [ ! -f "$MODEL_PATH/model.safetensors" ]; then
    echo "[merge] verl FSDP -> HF at $MODEL_PATH"
    python -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$MODEL_PATH"
  else
    echo "[merge] reuse existing merged weights at $MODEL_PATH"
  fi
fi
export MODEL_PATH
RESULTS="$OUT_DIR/results"; mkdir -p "$RESULTS"

EP_TAG="$(echo "$MODEL_PATH" | sed 's#/#--#g')_vllm_temp_${TEMPERATURE}"

run_evalplus () {  # $1 = humaneval|mbpp
  local ds="$1"
  echo "[$ds] codegen n=$N_SAMPLES temp=$TEMPERATURE"
  python -m evalplus.codegen "$MODEL_PATH" "$ds" --root "$RESULTS" \
    -n "$N_SAMPLES" --temperature "$TEMPERATURE" --backend vllm --tp 1 --trust_remote_code True
  local samples="$RESULTS/$ds/${EP_TAG}.jsonl"
  echo "[$ds] evaluate (plus tests)"
  python -m evalplus.evaluate --dataset "$ds" --samples "$samples" > "$RESULTS/eval_${ds}.txt" 2>&1 || true
  # Unbiased pass@1 / pass@N from the per-sample eval_results.json (base & plus).
  python "$SCRIPT_DIR/eval_passk_from_evalplus.py" \
    "${samples%.jsonl}_eval_results.json" "$N_SAMPLES" > "$RESULTS/passk_${ds}.json"
  echo "[$ds] -> $(cat "$RESULTS/passk_${ds}.json")"
}

for b in $BENCHES; do
  case "$b" in
    humaneval) run_evalplus humaneval ;;
    mbpp)      run_evalplus mbpp ;;
    lcb)
      if [ ! -d "$LCB_REPO" ]; then
        echo "[lcb] cloning LiveCodeBench"
        git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git "$LCB_REPO"
      fi
      LCB_JSONL="${LCB_JSONL:-$OUT_DIR/lcb_data/${LCB_VERSION}.jsonl}"
      if [ ! -f "$LCB_JSONL" ]; then
        echo "[lcb] downloading ${LCB_VERSION}.jsonl"
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download('livecodebench/code_generation_lite','${LCB_VERSION}.jsonl',repo_type='dataset',local_dir='$OUT_DIR/lcb_data')"
      fi
      echo "[lcb] generate + grade (release=$LCB_VERSION)"
      LCB_REPO="$LCB_REPO" LCB_JSONL="$LCB_JSONL" \
      N_SAMPLES="$N_SAMPLES" TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" \
      OUT_JSON="$RESULTS/passk_lcb.json" \
        python "$SCRIPT_DIR/eval_lcb_passk.py"
      echo "[lcb] -> $(cat "$RESULTS/passk_lcb.json")"
      ;;
    *) echo "unknown bench '$b' (use: humaneval mbpp lcb)"; exit 1 ;;
  esac
done

echo
echo "================= SUMMARY (pass@1 / pass@${N_SAMPLES}) ================="
python - "$RESULTS" "$N_SAMPLES" <<'PY'
import json, os, sys, glob
res, N = sys.argv[1], sys.argv[2]
for tag, label in [("humaneval","HumanEval+"),("mbpp","MBPP+"),("lcb","LiveCodeBench")]:
    f = os.path.join(res, f"passk_{tag}.json")
    if not os.path.exists(f): continue
    d = json.load(open(f))
    if tag == "lcb":
        p1, pn = d.get("pass@1"), d.get(f"pass@{N}")
        print(f"{label:14s}  pass@1={p1:.1f}  pass@{N}={pn:.1f}")
    else:
        # evalplus helper writes {"base":{"pass@1":..}, "plus":{...}}
        b, p = d["base"], d["plus"]
        print(f"{label:14s}  [base] p@1={b['pass@1']*100:.1f} p@{N}={b[f'pass@{N}']*100:.1f}"
              f"   [plus] p@1={p['pass@1']*100:.1f} p@{N}={p[f'pass@{N}']*100:.1f}")
PY
