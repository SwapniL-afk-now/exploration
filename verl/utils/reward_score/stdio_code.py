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
"""Stdin/stdout code-execution reward (LiveCodeBench's `testtype: stdin` problems).

This fork's `prime_code` module depends on `pyext`, which is incompatible with
Python 3.12 (it calls the removed `inspect.getargspec`). Rather than depend on a
sandbox_fusion service or a broken local module, this implements a minimal,
self-contained subprocess runner: the candidate program is executed as a script,
fed each test's stdin, and its stdout is compared (whitespace-normalized) against
the expected output.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(solution_str: str) -> str:
    matches = _CODE_FENCE_RE.findall(solution_str)
    return matches[-1] if matches else solution_str


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def compute_score(
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
    timeout: float = 6.0,
    continuous: bool = True,
) -> dict:
    """Score a completion against stdin/stdout test cases.

    `ground_truth` is a JSON object (or already-decoded dict) with parallel
    "inputs" and "outputs" lists. Each input is piped to the candidate program's
    stdin in a fresh subprocess; stdout is compared (whitespace-normalized)
    against the matching output. Scored as the pass fraction across all test
    cases when `continuous=True` (else 1.0 only if every case passes), or -1.0 if
    the code fails to run at all.
    """
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    inputs = spec["inputs"]
    outputs = spec["outputs"]
    code = _extract_code(solution_str)

    passed = 0
    for stdin, expected in zip(inputs, outputs, strict=True):
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if _normalize(proc.stdout) == _normalize(expected):
                passed += 1
        except Exception:
            continue

    total = len(inputs) or 1
    frac = passed / total
    score = frac if continuous else (1.0 if passed == total else -1.0)
    return {"score": score, "acc": passed == total, "pass_rate": frac}
