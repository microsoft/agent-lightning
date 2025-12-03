# Copyright (c) Microsoft. All rights reserved.

"""
Utility functions for the Math Agent

Includes:
- Answer extraction and normalization
- Reward computation with partial credit
- Evaluation metrics
"""

import re
from typing import Optional, Tuple


def extract_answer(text: str) -> str:
    """
    Extract the final answer from the agent's response.
    
    Looks for content within <answer></answer> tags.
    
    Args:
        text: The full response text from the agent
        
    Returns:
        The extracted answer, or empty string if no answer found
        
    Examples:
        >>> extract_answer("The result is <answer>42</answer>")
        '42'
        >>> extract_answer("Let me calculate... <answer>3.5</answer> is the answer")
        '3.5'
        >>> extract_answer("No answer tags here")
        ''
    """
    # Look for content between <answer> and </answer> tags
    match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for the last number in the text
    # This handles cases where the model doesn't use tags correctly
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return ""


def normalize_number(num_str: str) -> Optional[float]:
    """
    Normalize a number string to a float for comparison.
    
    Handles:
    - Integer and decimal numbers
    - Numbers with commas (e.g., "1,234")
    - Percentages (e.g., "50%")
    - Fractions in decimal form
    
    Args:
        num_str: String representation of a number
        
    Returns:
        Float value, or None if parsing fails
        
    Examples:
        >>> normalize_number("42")
        42.0
        >>> normalize_number("3.14159")
        3.14159
        >>> normalize_number("1,234")
        1234.0
        >>> normalize_number("50%")
        50.0
    """
    if not num_str:
        return None
    
    # Remove common formatting
    cleaned = num_str.strip()
    cleaned = cleaned.replace(',', '')  # Remove thousands separators
    cleaned = cleaned.replace('$', '')  # Remove dollar signs
    cleaned = cleaned.replace('%', '')  # Remove percent signs
    cleaned = cleaned.strip()
    
    # Try to convert to float
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def numbers_match(predicted: str, ground_truth: str, tolerance: float = 1e-4) -> bool:
    """
    Check if two number strings represent the same value.
    
    Uses a small tolerance for floating point comparison.
    
    Args:
        predicted: The predicted answer
        ground_truth: The correct answer
        tolerance: Maximum absolute difference to consider equal
        
    Returns:
        True if the numbers match within tolerance
        
    Examples:
        >>> numbers_match("42", "42.0")
        True
        >>> numbers_match("3.14159", "3.14160")
        True  # Within tolerance
        >>> numbers_match("10", "20")
        False
    """
    pred_num = normalize_number(predicted)
    truth_num = normalize_number(ground_truth)
    
    if pred_num is None or truth_num is None:
        return False
    
    return abs(pred_num - truth_num) <= tolerance


def has_valid_format(response: str) -> bool:
    """
    Check if the response has valid formatting.
    
    A valid response should:
    - Contain <answer> tags
    - Have some reasoning before the answer
    - Not be empty
    
    Args:
        response: The agent's full response
        
    Returns:
        True if formatting is valid
    """
    if not response or len(response.strip()) < 10:
        return False
    
    # Check for answer tags
    has_answer_tags = '<answer>' in response.lower() and '</answer>' in response.lower()
    
    return has_answer_tags


def used_calculator_check(response: str) -> bool:
    """
    Check if the agent used the calculator tool.
    
    Args:
        response: The agent's full response
        
    Returns:
        True if calculator tool was mentioned/used
    """
    calculator_indicators = [
        'tool_call',
        'calculator',
        'tool',
        'function',
    ]
    
    response_lower = response.lower()
    return any(indicator in response_lower for indicator in calculator_indicators)


def compute_reward(
    predicted: str,
    ground_truth: str,
    used_calculator: bool,
    full_response: str = "",
) -> float:
    """
    Compute reward for the agent's answer.
    
    Reward structure:
    - Correct answer: +1.0 (full credit)
    - Used calculator but wrong: +0.3 (partial credit for tool use)
    - Valid format but wrong: +0.1 (partial credit for following format)
    - Invalid format: -0.1 (penalty for not following instructions)
    
    This reward shaping encourages:
    1. Correct answers (highest reward)
    2. Using tools appropriately (partial credit)
    3. Following output format (minimal credit)
    
    Args:
        predicted: The predicted answer extracted from response
        ground_truth: The correct answer
        used_calculator: Whether the calculator tool was used
        full_response: Full response text for format checking
        
    Returns:
        Reward value between -0.1 and 1.0
    """
    # Check if answer is correct
    if numbers_match(predicted, ground_truth):
        return 1.0  # Perfect!
    
    # Check format validity
    valid_format = has_valid_format(full_response)
    
    if not valid_format:
        return -0.1  # Penalty for not following format
    
    # Partial credit for using calculator (shows correct behavior)
    if used_calculator:
        return 0.3
    
    # Minimal credit for valid format
    return 0.1


def compute_accuracy(predicted: str, ground_truth: str) -> float:
    """
    Compute binary accuracy (0 or 1).
    
    Args:
        predicted: The predicted answer
        ground_truth: The correct answer
        
    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    return 1.0 if numbers_match(predicted, ground_truth) else 0.0


def evaluate_batch(
    predictions: list[str],
    ground_truths: list[str],
    full_responses: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Evaluate a batch of predictions.
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of correct answers
        full_responses: Optional list of full response texts
        
    Returns:
        Dictionary containing evaluation metrics:
        - accuracy: Proportion of correct answers
        - avg_reward: Average reward across examples
        - format_compliance: Proportion with valid format
        - tool_usage: Proportion that used calculator
    """
    if full_responses is None:
        full_responses = [""] * len(predictions)
    
    total = len(predictions)
    if total == 0:
        return {
            "accuracy": 0.0,
            "avg_reward": 0.0,
            "format_compliance": 0.0,
            "tool_usage": 0.0,
        }
    
    correct = sum(
        numbers_match(pred, truth)
        for pred, truth in zip(predictions, ground_truths)
    )
    
    total_reward = sum(
        compute_reward(
            pred, 
            truth, 
            used_calculator_check(resp),
            resp
        )
        for pred, truth, resp in zip(predictions, ground_truths, full_responses)
    )
    
    valid_formats = sum(
        has_valid_format(resp)
        for resp in full_responses
    )
    
    tool_usage = sum(
        used_calculator_check(resp)
        for resp in full_responses
    )
    
    return {
        "accuracy": correct / total,
        "avg_reward": total_reward / total,
        "format_compliance": valid_formats / total,
        "tool_usage": tool_usage / total,
    }


if __name__ == "__main__":
    """
    Test the utility functions with sample data.
    """
    print("Testing Math Agent Utilities")
    print("=" * 60)
    
    # Test answer extraction
    print("\n1. Testing answer extraction:")
    test_responses = [
        "The answer is <answer>42</answer>",
        "Let me calculate: 5 + 3 = 8. <answer>8</answer>",
        "No tags here, just 99",
        "",
    ]
    for resp in test_responses:
        answer = extract_answer(resp)
        print(f"  Response: {resp[:50]}")
        print(f"  Extracted: '{answer}'")
    
    # Test number normalization
    print("\n2. Testing number normalization:")
    test_numbers = ["42", "3.14159", "1,234", "50%", "$100", "invalid"]
    for num in test_numbers:
        normalized = normalize_number(num)
        print(f"  '{num}' -> {normalized}")
    
    # Test number matching
    print("\n3. Testing number matching:")
    test_pairs = [
        ("42", "42.0"),
        ("3.14159", "3.14160"),
        ("10", "20"),
        ("1,234", "1234"),
    ]
    for pred, truth in test_pairs:
        match = numbers_match(pred, truth)
        print(f"  '{pred}' vs '{truth}': {match}")
    
    # Test reward computation
    print("\n4. Testing reward computation:")
    test_cases = [
        ("42", "42", True, "Used <tool_call> and got <answer>42</answer>"),
        ("42", "43", True, "Used calculator but got <answer>42</answer>"),
        ("42", "43", False, "Just guessed <answer>42</answer>"),
        ("42", "43", False, "No proper format at all"),
    ]
    for pred, truth, used_calc, response in test_cases:
        reward = compute_reward(pred, truth, used_calc, response)
        print(f"  Pred: {pred}, Truth: {truth}, Calc: {used_calc}")
        print(f"  Reward: {reward:.2f}")
    
    # Test batch evaluation
    print("\n5. Testing batch evaluation:")
    predictions = ["42", "43", "44", "45"]
    ground_truths = ["42", "43", "45", "45"]
    responses = [
        "Used tool <answer>42</answer>",
        "Used tool <answer>43</answer>",
        "Just guess <answer>44</answer>",
        "Calculated <answer>45</answer>",
    ]
    metrics = evaluate_batch(predictions, ground_truths, responses)
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  Avg Reward: {metrics['avg_reward']:.3f}")
    print(f"  Format Compliance: {metrics['format_compliance']:.2%}")
    print(f"  Tool Usage: {metrics['tool_usage']:.2%}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")