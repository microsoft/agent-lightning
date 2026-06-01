"""
Vanilla LLM Prompt for Demo Task
=================================
This file is OPTIONAL. If not provided, defaults to returning problem_statement directly.

This function is called in vanilla LLM mode (--mode llm) to create the prompt:
    prompt = create_prompt(problem_data)

Requirements:
    - Must define: create_prompt(problem_data: dict) -> str
    - problem_data contains: id, problem_statement, ground_truth, input_files, etc.
    - Return: The prompt string to send to the LLM
"""
from typing import Any, Dict


def create_prompt(problem_data: Dict[str, Any]) -> str:
    """
    Create prompt for vanilla LLM mode.
    
    For demo task, we just return the problem statement directly.
    For other tasks, you might want to add formatting, e.g.:
        - Math: add "Please put your answer in \\boxed{}"
        - Long context: embed documents in the prompt
    """
    return problem_data['problem_statement']
