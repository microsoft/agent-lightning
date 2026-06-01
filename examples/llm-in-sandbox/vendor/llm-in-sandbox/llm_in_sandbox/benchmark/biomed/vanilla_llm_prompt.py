"""
Prompt creation for biomedical task (vanilla LLM mode).
"""
from typing import Any, Dict


def create_prompt(problem_data: Dict[str, Any]) -> str:
    """Create prompt for biomedical problems."""
    return problem_data['problem_statement']
