"""
Demo Reward Function - Simple String Match
===========================================
Use this as a template for custom tasks.

This function is called for each problem to compute a score:
    score = compute_score(agent_answer, ground_truth)

Requirements:
    - Must define: compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float
    - Return value: 0.0 to 1.0
    - agent_answer: Content of {output_dir}/answer.txt (or empty string if not found)
    - ground_truth: The 'ground_truth' field from dataset
"""


def compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float:
    """
    Simple case-insensitive string match.
    
    Args:
        agent_answer: The agent's answer (from answer.txt)
        ground_truth: The expected answer
        
    Returns:
        1.0 if match, 0.0 otherwise
    """
    if not agent_answer or not ground_truth:
        return 0.0
    
    # Normalize and compare
    pred = agent_answer.strip().lower()
    gold = ground_truth.strip().lower()
    
    return 1.0 if pred == gold else 0.0
