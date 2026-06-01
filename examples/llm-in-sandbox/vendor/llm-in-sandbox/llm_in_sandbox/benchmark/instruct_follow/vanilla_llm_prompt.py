"""
Prompt creation for instruction following task (vanilla LLM mode).
"""
from typing import Any, Dict


def create_prompt(problem_data: Dict[str, Any]) -> str:
    """Create prompt for instruction following problems."""
    return problem_data['problem_statement']
