"""
Long context QA reward function.

NOTE: This task requires LLM-as-a-Judge for proper evaluation.
Run judge.py after benchmark.
"""


def compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float:
    """
    Placeholder - returns 0.
    Real evaluation should use judge.py with LLM-as-a-Judge.
    """
    return 0.0
