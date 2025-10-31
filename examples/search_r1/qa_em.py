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

import random
import re
import string
from typing import Mapping, Optional, Sequence, Union


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


def em_check(prediction: str, golden_answers: Union[str, Sequence[str]]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction: str, golden_answers: Union[str, Sequence[str]]) -> int:
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the last <answer>...</answer> span from a solution string.

    Returns None if fewer than two such spans are present, to match original behavior.
    """
    answer_pattern = r"<answer>(.*?)</answer>"
    match_iter = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match_iter)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(
    solution_str: str,
    ground_truth: Union[str, Sequence[str]],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Scoring function for exact match (EM)."""
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    else:
        if em_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_subem(
    solution_str: str,
    ground_truth: Mapping[str, Union[str, Sequence[str]]],
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """Scoring function for substring exact match (EM)."""
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0.0
    else:
        if subem_check(answer, ground_truth["target"]):
            return score
        else:
            return format_score
