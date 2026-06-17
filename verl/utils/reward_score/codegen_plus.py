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
"""Assert-based code-execution reward for HumanEval+/MBPP+-style benchmarks.

Unlike the stdin/stdout APPS-style format handled by `prime_code`, HumanEval+ and
MBPP+ ship a self-contained `test` script per problem: it either defines
`check(candidate)` (HumanEval+ convention — caller invokes `check(entry_point_fn)`)
or directly references the entry-point function name at module level (MBPP+
convention). Both are handled here by executing `candidate_code + test_script` in
an isolated subprocess and, if a `check` function was defined, calling it explicitly.
"""

from __future__ import annotations

import json
import multiprocessing
import re

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def _extract_code(solution_str: str) -> str:
    """Pull the last fenced Python code block out of a completion, else use it raw."""
    matches = _CODE_FENCE_RE.findall(solution_str)
    return matches[-1] if matches else solution_str


def _run(code: str, test: str, entry_point: str, result: list) -> None:
    namespace: dict = {}
    try:
        exec(code, namespace)  # noqa: S102 - sandboxed in a separate, timeout-bounded process
        exec(test, namespace)  # noqa: S102
        if entry_point not in namespace:
            result.append(False)
            return
        if "check" in namespace:
            namespace["check"](namespace[entry_point])
        # else: MBPP+-style `test` scripts call the entry point directly at module
        # level, so simply exec'ing it above already ran (and would have raised on
        # failure) the assertions.
        result.append(True)
    except Exception:
        result.append(False)


def compute_score(
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
    timeout: float = 10.0,
) -> dict:
    """Score a completion against a HumanEval+/MBPP+-style assert test.

    `ground_truth` is a JSON object (or already-decoded dict) with:
      - "prompt": the function stub (signature + docstring) the model was given
      - "test": the assert-based test script
      - "entry_point": the function name under test

    The model's completion is combined with `prompt` (covers completions that only
    contain the function body), executed with `test` in an isolated subprocess
    under a hard timeout, and scored 1.0 on pass / -1.0 on any failure, exception,
    or timeout.
    """
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    prompt = spec.get("prompt", "")
    test = spec["test"]
    entry_point = spec["entry_point"]

    completion = _extract_code(solution_str)
    code = completion if f"def {entry_point}" in completion else f"{prompt}\n{completion}"

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_run, args=(code, test, entry_point, result))
    p.start()
    p.join(timeout=timeout)
    if p.is_alive():
        p.kill()
        p.join()

    passed = bool(result) and result[0] is True
    return {"score": 1.0 if passed else -1.0, "acc": passed}
