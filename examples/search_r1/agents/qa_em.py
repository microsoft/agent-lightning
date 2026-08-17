# Copyright (c) Microsoft. All rights reserved.

# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from __future__ import annotations

import random
import re
import string
from collections.abc import Mapping, Sequence


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction: str, golden_answers: str | Sequence[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def subem_check(prediction: str, golden_answers: str | Sequence[str]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) in normalized_prediction:
            return 1
    return 0


def extract_solution(solution_str: str) -> str | None:
    """Extract the last <answer>...</answer> span from a solution string."""
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))
    if not matches:
        return None
    return matches[-1].group(1).strip()


def compute_score_em(
    solution_str: str,
    ground_truth: str | Sequence[str],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Scoring function for exact match (EM)."""
    del method
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    if em_check(answer, ground_truth):
        return score
    return format_score


def compute_score_subem(
    solution_str: str,
    ground_truth: Mapping[str, str | Sequence[str]],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Scoring function for substring exact match (EM)."""
    del method
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    if subem_check(answer, ground_truth["target"]):
        return score
    return format_score
