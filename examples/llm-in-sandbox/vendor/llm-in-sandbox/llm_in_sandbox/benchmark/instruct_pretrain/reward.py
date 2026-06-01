"""
Instruction-Pretrain Reward Function
=====================================

Routes to appropriate scorer based on domain and qa_type:
- math domain -> math/reward.py
- chem/biomed domain -> biomed/reward.py (multiple choice)
- open_ended -> Math verification first, then ROUGE-L
"""

# Import reward functions from other tasks
from llm_in_sandbox.benchmark.math.reward import compute_score as compute_math_score
from llm_in_sandbox.benchmark.biomed.reward import compute_score as compute_biomed_score
from llm_in_sandbox.benchmark.chem.reward import compute_score as compute_chem_score
from llm_in_sandbox.benchmark.instruct_follow.reward import compute_score as compute_if_score

# ROUGE for open-ended questions
try:
    from rouge_score import rouge_scorer
except ImportError:
    raise ImportError("rouge-score is required. Install with: pip install rouge-score")


def map_question_type(original_type: str, ground_truth: str) -> str:
    """
    Map original qa_type to standardized question type.
    
    Returns: "multiple_choice_single_answer", "multiple_choice_multiple_answers", or "open_ended"
    """
    if not original_type:
        return "open_ended"
    
    original_type_lower = original_type.lower()
    
    if "multiple choice" in original_type_lower:
        if ',' in ground_truth:
            return "multiple_choice_multiple_answers"
        else:
            return "multiple_choice_single_answer"
    elif "free-form" in original_type_lower:
        return "open_ended"
    
    return "open_ended"


def compute_score(agent_answer: str, ground_truth: str, **kwargs) -> float:
    """
    Compute score for instruction-pretrain problems.
    
    Routes to appropriate scorer based on domain and qa_type.
    """
    if not agent_answer or not ground_truth:
        return 0.0
    
    domain = kwargs['domain'].lower()
    model_output = agent_answer.strip()
    ground_truth = ground_truth.strip()
    
    # Route by domain first
    if domain.startswith('math') or domain.startswith('physics'):
        return compute_math_score(model_output, ground_truth)
    
    if domain.startswith('chem'):
        return compute_chem_score(model_output, ground_truth)
    
    if domain.startswith('biomed'):
        return compute_biomed_score(model_output, ground_truth)
    if domain.startswith('instruct_follow'):
        return compute_if_score(model_output, ground_truth)
    else:
        assert domain.startswith('instruct_pretrain') or domain.startswith('long_context')
    
    qa_type = kwargs.get('qa_type', 'open_ended')
    standardized_type = map_question_type(qa_type, ground_truth)
    
    # Then route by question type
    if standardized_type == "multiple_choice_single_answer":
        return compute_biomed_score(model_output, ground_truth) # biomed scorer works for single-answer MCQs
    
    elif standardized_type == "multiple_choice_multiple_answers":
        # Multiple choice: F1 score
        model_choices = set(c.strip().upper() for c in model_output.split(',') if c.strip())
        gt_choices = set(c.strip().upper() for c in ground_truth.split(',') if c.strip())
        
        if len(model_choices) == 0 or len(gt_choices) == 0:
            return 0.0
        
        correct = model_choices & gt_choices
        precision = len(correct) / len(model_choices)
        recall = len(correct) / len(gt_choices)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    elif standardized_type == "open_ended":
        # First try math verification
        score = compute_math_score(model_output, ground_truth)
        if score > 0:
            return 1.0
        
        # ROUGE-L for text similarity
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(ground_truth, model_output)
        return scores['rougeL'].fmeasure
    
    return 0.0
