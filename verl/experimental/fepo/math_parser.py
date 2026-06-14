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
"""Math-answer extraction and normalization for FEPO math RL.

The notebook this module ports used a pragmatic verifier: extract the last boxed
answer when present, fall back to GSM8K-style ``####`` answers and finally the
last numeric/text answer, then compare normalized answer strings.  The helpers
below keep that behavior but centralize it so training and validation use the
same parser.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, getcontext
from fractions import Fraction
from typing import Any, Optional

getcontext().prec = 50

BOXED_PATTERN = r"\boxed{"
PYTHON_CODE_FENCE_PATTERN = re.compile(r"```(?:python|py)\b.*?```", flags=re.IGNORECASE | re.DOTALL)
MARKDOWN_FENCE_PATTERN = re.compile(r"```(?:[^\n`]*)\n?(.*?)```", flags=re.DOTALL)


@dataclass(frozen=True)
class MathRewardResult:
    """Parsed math reward details for one generation."""

    reward: float
    is_correct: bool
    has_parseable_answer: bool
    prediction_normalized: Optional[str]
    ground_truth_normalized: Optional[str]
    anti_reward: float


def strip_latex_wrappers(text: str) -> str:
    s = str(text).strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    for left, right in [("\\(", "\\)"), ("\\[", "\\]")]:
        if s.startswith(left) and s.endswith(right) and len(s) >= len(left) + len(right):
            s = s[len(left) : -len(right)].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("−", "-")
    return s.strip()


def decimal_to_canonical(value: Decimal) -> str:
    if value == value.to_integral():
        return str(int(value))
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _decimal_from_fraction(num: str, den: str) -> Optional[Decimal]:
    try:
        denominator = Decimal(den.replace(",", ""))
        if denominator == 0:
            return None
        return Decimal(num.replace(",", "")) / denominator
    except InvalidOperation:
        return None


def try_decimal(text: str) -> Optional[Decimal]:
    s = strip_latex_wrappers(text)
    s = s.replace(",", "").strip()

    frac_match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
    if frac_match:
        return _decimal_from_fraction(frac_match.group(1), frac_match.group(2))

    slash_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)/([-+]?\d+(?:\.\d+)?)", s)
    if slash_match:
        return _decimal_from_fraction(slash_match.group(1), slash_match.group(2))

    percent_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)%", s)
    if percent_match:
        try:
            return Decimal(percent_match.group(1)) / Decimal(100)
        except InvalidOperation:
            return None

    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", s):
        try:
            return Decimal(s)
        except InvalidOperation:
            return None
    return None


def normalize_answer_text(text: Any) -> Optional[str]:
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        if not text:
            return None
        text = text[0]

    s = strip_latex_wrappers(str(text))
    if not s:
        return None

    direct_value = try_decimal(s)
    if direct_value is not None:
        return decimal_to_canonical(direct_value)

    s = s.replace("\\boxed", "")
    s = re.sub(r"\\(?:mathrm|text|textbf)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\overline\{([^{}]*)\}", r"\1", s)
    s = s.strip()
    for left, right in [("{", "}"), ("(", ")"), ("[", "]")]:
        if s.startswith(left) and s.endswith(right):
            s = s[1:-1].strip()
            break
    s = re.sub(r"\s+", " ", s).strip()

    value = try_decimal(s)
    if value is not None:
        return decimal_to_canonical(value)

    s = s.replace(",", "")
    s = s.replace("\\%", "%")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\sqrt", "sqrt")
    s = s.strip().lower()
    while len(s) > 1 and s[-1] in ".,;:":
        s = s[:-1].strip()
    return s or None


def extract_all_boxed(text: str) -> list[str]:
    text = str(text)
    results: list[str] = []
    start = 0
    while True:
        idx = text.find(BOXED_PATTERN, start)
        if idx == -1:
            break
        pos = idx + len(BOXED_PATTERN)
        depth = 1
        chars: list[str] = []
        while pos < len(text) and depth > 0:
            ch = text[pos]
            if ch == "{":
                depth += 1
                chars.append(ch)
            elif ch == "}":
                depth -= 1
                if depth:
                    chars.append(ch)
            else:
                chars.append(ch)
            pos += 1
        if depth == 0:
            results.append("".join(chars).strip())
        start = pos

    boxed_space = re.findall(r"\\boxed\s+([^$\n]+)", text)
    results.extend(answer.strip() for answer in boxed_space)
    return results


def extract_last_boxed(text: str) -> Optional[str]:
    boxes = extract_all_boxed(text)
    return boxes[-1] if boxes else None


def extract_after_hash_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(.+)$", str(text), flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _without_python_code_blocks(text: str) -> str:
    """Drop executable snippets but keep unlabeled output fences as text."""

    s = PYTHON_CODE_FENCE_PATTERN.sub("\n", str(text))

    def replace_fence(match: re.Match[str]) -> str:
        content = match.group(1).strip()
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        first = lines[0].lower() if lines else ""
        looks_like_code = (
            first in {"python", "py"}
            or any(re.search(r"(?<![=!<>])=(?!=)|\bprint\s*\(|\breturn\b|\bfor\b|\bwhile\b", line) for line in lines)
        )
        if looks_like_code:
            return "\n"
        return f"\n{content}\n"

    return MARKDOWN_FENCE_PATTERN.sub(replace_fence, s)


def _extract_terminal_value(fragment: str, *, allow_text: bool = False) -> Optional[str]:
    s = strip_latex_wrappers(fragment)
    if not s:
        return None

    boxed = extract_last_boxed(s)
    if boxed is not None:
        return boxed
    if "\\boxed{" in s:
        return None
    if allow_text and s.rstrip().endswith(":"):
        return None

    should_be = re.search(r"(?i)\bshould be\s+(.+?)\s*,?\s+not\s+.+$", s)
    if should_be:
        s = should_be.group(1).strip()

    evaluates_to = re.search(r"(?i)\bevaluates to\s+(.+)$", s)
    if allow_text and evaluates_to and not _is_standalone_answer_fragment(evaluates_to.group(1)):
        return None

    # Prefer the right side of the last displayed equation or assignment.
    if "=" in s:
        rhs = s.rsplit("=", 1)[-1].strip()
        if re.search(r"\d", rhs):
            s = rhs

    number_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*(?:\.\d+)?)?%?", s)
    if number_matches:
        return number_matches[-1]

    if not allow_text:
        return None

    text_match = re.search(
        r"(?i)(?:final answer|answer|final result|result|output|printed output|evaluates to)\s*(?:is|=|:)?\s*(.+)$",
        s,
    )
    candidate = text_match.group(1).strip() if text_match else s.strip()
    candidate = re.sub(r"(?i)^(?:therefore|thus|so|hence)[,\s:]+", "", candidate).strip()
    candidate = candidate.strip("`")
    if not re.search(r"\d", candidate):
        if len(candidate) > 40 or len(candidate.split()) > 4:
            return None
        if not (re.fullmatch(r"[A-Za-z]", candidate) or re.search(r"[\\=+*/^{}]", candidate)):
            return None
    if candidate and candidate not in {":", ".", ",", ";", "?", ")"} and not candidate.endswith(":"):
        return candidate
    return None


def _is_standalone_answer_fragment(fragment: str) -> bool:
    s = strip_latex_wrappers(fragment)
    if not s or s in {"\\[", "\\]", "$", "$$"}:
        return False
    if extract_last_boxed(s) is not None:
        return True
    if "\\boxed{" in s:
        return False
    if re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*(?:\.\d+)?)?%?\.?", s):
        return True
    if "=" not in s:
        return False

    without_latex_commands = re.sub(r"\\[A-Za-z]+", "", s)
    return re.search(r"\d", s) is not None and re.search(r"[A-Za-z]", without_latex_commands) is None


def extract_final_number_or_text(text: str) -> Optional[str]:
    s = _without_python_code_blocks(str(text)).strip()
    if not s:
        return None

    lines = [line.strip() for line in s.splitlines() if line.strip() and not line.strip().startswith("```")]
    explicit_line_pattern = re.compile(
        r"(?i)\b(?:final answer|answer|final result|result|therefore|thus|hence|output|printed|prints?|evaluates to)\b"
    )
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if explicit_line_pattern.search(line):
            candidate = _extract_terminal_value(line, allow_text=True)
            if candidate is not None:
                return candidate
            for lookahead_end in range(idx + 2, min(idx + 5, len(lines) + 1)):
                fragment = "\n".join(lines[idx + 1 : lookahead_end])
                if not _is_standalone_answer_fragment(fragment):
                    continue
                candidate = _extract_terminal_value(fragment)
                if candidate is not None:
                    return candidate

    # Handle a final display equation/value after prose such as
    # "So the final result is:\n\\[ 17^2 + 23^2 = 818 \\]".
    for line in reversed(lines):
        if not _is_standalone_answer_fragment(line):
            continue
        candidate = _extract_terminal_value(line)
        if candidate is not None:
            return candidate

    return None


def extract_model_answer(text: str) -> Optional[str]:
    boxed = extract_last_boxed(text)
    if boxed is not None:
        return boxed
    hashed = extract_after_hash_answer(text)
    if hashed is not None:
        return hashed
    return extract_final_number_or_text(text)


def normalize_model_answer(text: str) -> Optional[str]:
    return normalize_answer_text(extract_model_answer(text))


def _first_non_empty(values: list[Any]) -> Optional[Any]:
    for value in values:
        if value is not None and str(value).strip():
            return value
    return None


def normalize_ground_truth_answer(answer: Any, dataset_kind: str | None = None) -> Optional[str]:
    if isinstance(answer, dict):
        answer = _first_non_empty(
            [
                answer.get("ground_truth"),
                answer.get("answer"),
                answer.get("final_answer"),
                answer.get("label"),
            ]
        )
    if isinstance(answer, (list, tuple)):
        answer = _first_non_empty(list(answer))

    dataset_kind = (dataset_kind or "").lower()
    if dataset_kind in {"dapo-math-17k", "amc23", "aime24", "aime25", "aime26"}:
        candidate = str(answer)
    else:
        raw_answer = str(answer)
        candidate = extract_after_hash_answer(raw_answer) or extract_last_boxed(raw_answer)
        if candidate is None:
            stripped = raw_answer.strip()
            if stripped.startswith("$") or "\\" in stripped or "=" in stripped:
                candidate = stripped
            else:
                candidate = extract_final_number_or_text(raw_answer) or stripped
    return normalize_answer_text(candidate)


def _try_fraction(value: str) -> Optional[Fraction]:
    dec = try_decimal(value)
    if dec is None:
        return None
    try:
        return Fraction(dec)
    except (ArithmeticError, ValueError):
        return None


def answers_equivalent(prediction: Optional[str], ground_truth: Optional[str]) -> bool:
    if prediction is None or ground_truth is None:
        return False
    if prediction == ground_truth:
        return True

    pred_fraction = _try_fraction(prediction)
    gt_fraction = _try_fraction(ground_truth)
    if pred_fraction is not None and gt_fraction is not None:
        return pred_fraction == gt_fraction
    return False


def compute_math_reward(response: str, ground_truth: Any, dataset_kind: str | None = None) -> MathRewardResult:
    prediction = normalize_model_answer(response)
    target = normalize_ground_truth_answer(ground_truth, dataset_kind=dataset_kind)
    is_correct = answers_equivalent(prediction, target)
    reward = 1.0 if is_correct else 0.0
    return MathRewardResult(
        reward=reward,
        is_correct=is_correct,
        has_parseable_answer=prediction is not None,
        prediction_normalized=prediction,
        ground_truth_normalized=target,
        anti_reward=1.0 - 2.0 * reward,
    )
