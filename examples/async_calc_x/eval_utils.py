# Copyright (c) Microsoft. All rights reserved.

# Copied and adapted from https://github.com/prompteus/calc-x/blob/master/gadgets/metrics.py

"""Evaluation utilities for Calc-X math problems.

Pure functions — no framework dependencies. Used by hooks (on_succeeded)
to compute rewards.
"""

import math
import re
import string

import sympy


def normalize_option(option: str) -> str:
    """
    >>> normalize_option("  (A)  \\n")
    'A'
    """
    return re.sub(r"(\s+|\(|\))", "", option)


def is_option_result(result: str) -> bool:
    """
    >>> is_option_result("  A)  \\n")
    True
    >>> is_option_result("  23/7 ")
    False
    """
    return normalize_option(result) in list(string.ascii_letters)


def float_eval(input_str: str) -> float:
    if " = around " in input_str:
        input_str = input_str.split(" = around ")[0]
    expr = sympy.parse_expr(input_str, evaluate=True)
    return float(expr.evalf())


def scalar_are_results_same(pred_result: str, true_result: str, rel_tol: float) -> bool:
    """Compare predicted and true results with numeric tolerance.

    Handles exact string match, multiple-choice options, and numeric
    comparison via sympy parsing.
    """
    pred_result = str(pred_result) if pred_result is not None else ""  # type: ignore
    true_result = str(true_result) if true_result is not None else ""  # type: ignore

    if pred_result.strip() == true_result.strip():
        return True

    if is_option_result(true_result):
        # The task is to select correct option
        true_result = normalize_option(true_result)
        pred_result = normalize_option(pred_result)
        return pred_result == true_result

    # The task is to calculate the result as a number
    try:
        pred_float = float_eval(pred_result)
        true_float = float_eval(true_result)
        return math.isclose(pred_float, true_float, rel_tol=rel_tol)
    except Exception:
        pass

    return False


def evaluate(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if prediction matches ground truth, 0.0 otherwise."""
    return float(scalar_are_results_same(prediction, ground_truth, 1e-2))
