"""
Chemistry multiple choice reward function.
"""

import re


def extract_from_boxed(text: str) -> str:
    """Extract content from \\boxed{}."""
    boxed_prefix = "\\boxed{"
    last_idx = text.rfind(boxed_prefix)
    
    if last_idx == -1:
        return text.strip()
    
    start = last_idx + len(boxed_prefix)
    depth = 1
    i = start
    
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    
    if depth == 0:
        return text[start:i-1].strip()
    return text.strip()


def extract_multiple_choice_answer(text: str) -> str:
    """
    Extract answer letter from model output.
    
    Supported formats:
    - "answer": "C"
    - Answer: C
    - The answer is C
    - C
    - Direct letter output
    """
    if not text:
        return ""
    
    # Method 1: Match "answer": "X" format
    pattern1 = r'"answer"\s*:\s*"([A-Z])"'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Method 2: Match Answer: X format
    pattern2 = r'answer\s*[:：]\s*([A-Z])\b'
    match = re.search(pattern2, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Method 3: Match "The answer is X" format
    pattern3 = r'(?:answer|choice)\s+is\s+([A-Z])\b'
    match = re.search(pattern3, text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Method 4: Match Boxed format
    # Only use if extract_from_boxed actually extracted boxed content
    extracted_answer = extract_from_boxed(text)
    if extracted_answer != text.strip() and extracted_answer and extracted_answer[0].upper().isalpha():
        return extracted_answer[0].upper()
    
    # Method 5: Find last standalone letter A-Z
    pattern4 = r'\b([A-Z])\b'
    matches = re.findall(pattern4, text.upper())
    if matches:
        return matches[-1]
    
    # Method 6: Check if text starts with a letter (some models output "A, xxx")
    text_stripped = text.strip()
    if text_stripped and text_stripped[0].upper().isalpha():
        return text_stripped[0].upper()

    return ""


def compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float:
    """
    Verify multiple choice answer.
    
    Args:
        agent_answer: Model output text
        ground_truth: Correct answer
    
    Returns:
        float: 1.0 if correct, 0.0 otherwise
    """
    if agent_answer.strip().upper() == ground_truth.strip().upper():
        return 1.0
    
    predicted = extract_multiple_choice_answer(agent_answer)
    return 1.0 if predicted.strip().upper() == ground_truth.strip().upper() else 0.0
