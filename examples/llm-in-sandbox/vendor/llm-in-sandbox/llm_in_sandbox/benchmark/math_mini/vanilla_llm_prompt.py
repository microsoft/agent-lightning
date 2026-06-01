"""
Prompt creation for math task (vanilla LLM mode).
"""
from typing import Any, Dict


def create_prompt(problem_data: Dict[str, Any]) -> str:
    """Create prompt for math problems."""
    problem_statement = problem_data['problem_statement']
    return f"{problem_statement}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
