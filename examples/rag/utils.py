# Copyright (c) Microsoft. All rights reserved.

# type: ignore

import re
import string
from collections import Counter
from typing import List, Optional, Set, Tuple

ANS_BEGIN = "<answer>"
ANS_END = "</answer>"
GEN_BEGIN = "<|im_start|>assistant\n"
FORMAT_SCORE = 0.1
FORMAT_PUNISH = -2


def normalize_answer(s: str) -> str:
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


def f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def lenient_f1_score(prediction: str, ground_truth: str) -> Tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        if normalized_ground_truth == "yes" and ("no" in normalized_prediction or "noanswer" in normalized_prediction):
            return ZERO_METRIC
        if normalized_ground_truth == "no" and ("yes" in normalized_prediction or "noanswer" in normalized_prediction):
            return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def cover_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def extract_answer(response: str) -> str:
    if ANS_BEGIN not in response or ANS_END not in response:
        return ""
    pos1 = response.rfind(ANS_BEGIN)
    pos2 = response.rfind(ANS_END)
    assert pos2 != -1
    if pos1 != -1:
        ans = response[pos1 + len(ANS_BEGIN) : pos2]
    else:
        ans = response[len(ANS_BEGIN) : pos2]
    return ans


def split_response(text: str) -> Tuple[str, str]:
    start_response = text.rfind(GEN_BEGIN)
    response = text[start_response + len(GEN_BEGIN) :]
    prompt = text[: -len(response)]
    return prompt, response


def extract_recall_chunk(prompt: str, response: str) -> Tuple[Set[str], Set[str]]:
    import re

    # 正则表达式，匹配每个search_step内1.和2.后面的内容
    pattern = r"Retrieved sentences:\s*1\.\s*(.*?)\s*2\.\s*(.*?)(?:\n\s*\d+\.|\n\n|$)"

    # 使用re.findall 提取所有的(s1, s2)
    origin_recall = re.findall(pattern, prompt, re.DOTALL)
    sequential_recall = re.findall(pattern, response, re.DOTALL)
    origin_recall_set = set(s for pair in origin_recall for s in pair)
    sequential_recall_set = set(s for pair in sequential_recall for s in pair)

    return origin_recall_set, sequential_recall_set


import re


def extract_retrieved_paragraphs(log_text: str) -> List[str]:
    # 正则表达式匹配 "Retrieved paragraph:" 后的内容
    pattern = re.compile(r"Retrieved paragraph:\s*(.*?)\n", re.DOTALL)

    # 提取匹配的段落
    matches = pattern.findall(log_text)
    matches = list(set(matches))
    return matches


def compute_score(
    prediction: str, gold: str, gold_sentences: Optional[List[str]] = None, data_source: Optional[str] = None
) -> float:
    # format acc
    format_acc = FORMAT_SCORE

    _, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == "":
        # format score 0.1
        # if '<query>' not in response or '</query>' not in response:
        #     return 0.0
        # return 0.0
        delimiter = "<|im_start|>assistant"
        last_time_ans = response.split(delimiter)[-1]
        if "<query" not in last_time_ans or "</query>" not in last_time_ans:
            return 0.0
        return format_acc

    # answer acc
    em, _ = exact_match_score(ans, gold), cover_exact_match_score(ans, gold)
    f1, _, _ = f1_score(ans, gold)

    if fact_checking_api(prediction, ans):
        answer_acc = max(float(em), f1)
    else:
        answer_acc = 0
    # # search acc
    # if gold_sentences and search_weight:
    #     origin_recall_set, sequential_recall_set = extract_recall_chunk(prompt, response)
    #     gold_sentences_set = set(gold_sentences) - origin_recall_set
    #     matched = gold_sentences_set & sequential_recall_set
    #     search_acc = len(matched) / len(gold_sentences_set) if len(gold_sentences_set) != 0 else 1.0
    #     # print(f's_acc {search_acc}|a_acc {answer_acc=}| score {format_acc + (1 - format_acc) * (search_weight + (1 - search_weight) * answer_acc)} |m_len {len(matched)}|g_len {len(gold_sentences_set)}|o_len {len(origin_recall_set)}|s_len {len(sequential_recall_set)}|{gold_sentences_set}|{sequential_recall_set}')
    #     if search_acc < 1:
    #         return format_acc + (1 - format_acc) * search_weight * search_acc
    # # print(f'SCORE: {score} | {ans} | {gold} | {prediction}' )

    return format_acc + (1 - format_acc) * answer_acc
    # return  answer_acc


def compute_reward(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> float:
    prediction = solution_str
    gold = ground_truth
    return compute_score(prediction, gold, gold_sentences=gold_sentences, data_source=data_source)


def compute_em(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> float:
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == "":
        # format score 0.1
        # if '<query>' not in response or '</query>' not in response:
        #     return 0.0
        return 0.0

    # answer acc
    em = exact_match_score(ans, gold)
    return em


def compute_cem(
    solution_str=None,
    ground_truth=None,
    gold_sentences=None,
    data_source=None,
    extra_info=None,
):
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == "":
        return 0.0

    # answer acc
    cem = cover_exact_match_score(ans, gold)
    return cem


def compute_response_cem(
    solution_str=None,
    ground_truth=None,
    gold_sentences=None,
    data_source=None,
    extra_info=None,
):
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = response
    if ans == "":
        return 0.0

    # answer acc
    cem = cover_exact_match_score(ans, gold)
    return cem


def compute_lenient_f1(
    solution_str=None,
    ground_truth=None,
    gold_sentences=None,
    data_source=None,
    extra_info=None,
):
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == "":
        return 0.0

    # answer acc
    f1, prec, recall = lenient_f1_score(ans, gold)
    return f1


def compute_lenient_response_f1(
    solution_str=None,
    ground_truth=None,
    gold_sentences=None,
    data_source=None,
    extra_info=None,
):
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = response
    if ans == "":
        return 0.0

    # answer acc
    f1, prec, recall = lenient_f1_score(ans, gold)
    return f1


def fact_checking_api(prediction: str, ans: str) -> bool:
    return True  # Placeholder for actual fact-checking logic


def compute_f1(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> float:
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == "":
        return 0.0

    # answer acc
    f1, _, _ = f1_score(ans, gold)
    return f1


def compute_format(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> float:
    prediction = solution_str
    gold = ground_truth
    _, response = split_response(prediction)
    ans = extract_answer(response)
    if ans == "":
        delimiter = "<|im_start|>assistant"
        last_time_ans = response.split(delimiter)[-1]
        if "<query" not in last_time_ans or "</query>" not in last_time_ans:
            return 0
    return FORMAT_SCORE


def split_trace(text: str) -> Tuple[str, str]:
    start_response = text.find(GEN_BEGIN)
    response = text[start_response + len(GEN_BEGIN) :]
    prompt = text[: -len(response)]
    return prompt, response


def compute_action_query(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count("<query>") + trace.count("<query,"), trace.count("</query>"))
    return res


def compute_action_bm25(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count("<query keyword"), trace.count("</query>"))
    return res


def compute_action_read_pre(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count("<query previous"), trace.count("</query>"))
    return res


def compute_action_read_nxt(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count("<query next"), trace.count("</query>"))
    return res


def compute_action_continue(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count(", continue"), trace.count("</query>"))
    return res


def compute_action_match(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count(', match_phrase="'), trace.count("</query>"))
    return res


def compute_total_action_number(
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    gold_sentences: Optional[List[str]] = None,
    data_source: Optional[str] = None,
    extra_info: Optional[str] = None,
) -> int:
    prediction = solution_str
    gold = ground_truth
    prompt, trace = split_trace(prediction)
    res = min(trace.count("<query"), trace.count("</query>"))
    return res


# define reward functions for evaluation


def compute_scores(answer: str, ground_truth: str) -> float:
    parsed_answer = extract_answer(answer)
    if parsed_answer is None:
        return -0.1
    f1, precision, recall = f1_score(parsed_answer, ground_truth)
    # em = float(exact_match_score(parsed_answer, ground_truth))
    # cem = float(cover_exact_match_score(answer, ground_truth))
    return f1
