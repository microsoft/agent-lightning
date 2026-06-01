"""
Prompt creation for chemistry task (vanilla LLM mode).
"""
from typing import Any, Dict


def create_prompt(problem_data: Dict[str, Any]) -> str:
    """Create prompt for chemistry problems."""
    return problem_data['problem_statement']
