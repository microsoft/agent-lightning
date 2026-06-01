"""
Math problem reward function.
"""

import re

# Check math-verify at import time
try:
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    raise ImportError("math-verify is required. Install with: pip install math-verify")


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{}, supports nested braces."""
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


def compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float:
    """Compute score for math problems using math_verify."""
    try:
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        extracted_answer = extract_boxed(agent_answer)
        extracted_boxed = "\\boxed{" + extracted_answer + "}"
        
        ret_score, _ = verify_func([ground_truth_boxed], [extracted_boxed])
        
        return float(ret_score)
        
    except Exception as e:
        print(f"Error in compute_score: {e}")
        return 0.0
