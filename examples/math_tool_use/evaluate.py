# Copyright (c) Microsoft. All rights reserved.

"""
Evaluation Script for Math Agent

Evaluates a trained math agent checkpoint on the GSM8K test set.
Provides detailed metrics and error analysis.

Usage:
    python evaluate.py --checkpoint path/to/checkpoint --test_file data/gsm8k_test.parquet
    python evaluate.py --checkpoint Qwen/Qwen2.5-1.5B-Instruct --n_examples 100
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from openai import AsyncOpenAI
from tqdm import tqdm

from calculator_tool import calculator_tool
from utils import extract_answer, numbers_match, evaluate_batch, normalize_number


MATH_AGENT_PROMPT = """You are a helpful assistant that solves grade school math problems step by step.

When solving a problem:
1. Read the problem carefully and identify what is being asked
2. Break down the problem into smaller steps
3. Use the calculator tool for any arithmetic operations
4. Show your reasoning for each step
5. Provide your final answer wrapped in <answer></answer> tags

Available tools:
- calculator: Evaluates mathematical expressions
  Example: {"name": "calculator", "arguments": {"expression": "24 * 7 + 15"}}

Always use <answer></answer> tags for your final numerical answer.
"""


async def evaluate_single_problem(
    client: AsyncOpenAI,
    problem: Dict[str, Any],
    model: str,
    max_iterations: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate the agent on a single problem.
    
    Args:
        client: OpenAI client
        problem: Dictionary with 'question' and 'answer'
        model: Model name
        max_iterations: Maximum tool calls
        
    Returns:
        Dictionary with results
    """
    messages = [
        {"role": "system", "content": MATH_AGENT_PROMPT},
        {"role": "user", "content": f"Problem: {problem['question']}"}
    ]
    
    used_calculator = False
    tool_calls_made = []
    
    for _ in range(max_iterations):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluates a mathematical expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "Mathematical expression to evaluate"
                            }
                        },
                        "required": ["expression"]
                    }
                }
            }],
            temperature=0.0,  # Greedy decoding for evaluation
            max_tokens=1024,
        )
        
        message = response.choices[0].message
        
        if message.content:
            messages.append({
                "role": "assistant",
                "content": message.content
            })
        
        if message.tool_calls:
            used_calculator = True
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                    expression = arguments.get("expression", "")
                    result = calculator_tool(expression)
                    
                    tool_calls_made.append({
                        "expression": expression,
                        "result": result
                    })
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                        "name": "calculator"
                    })
                except Exception as e:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": f"Error: {str(e)}",
                        "name": "calculator"
                    })
            continue
        
        break
    
    # Extract final response
    final_response = messages[-1]["content"] if messages else ""
    predicted = extract_answer(final_response)
    ground_truth = str(problem['answer'])
    correct = numbers_match(predicted, ground_truth)
    
    return {
        "question": problem['question'],
        "ground_truth": ground_truth,
        "predicted": predicted,
        "correct": correct,
        "used_calculator": used_calculator,
        "num_tool_calls": len(tool_calls_made),
        "tool_calls": tool_calls_made,
        "full_response": final_response,
    }


async def evaluate_checkpoint(
    checkpoint_path: str,
    test_file: str,
    n_examples: Optional[int] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a checkpoint on the test set.
    
    Args:
        checkpoint_path: Path to model checkpoint or model name
        test_file: Path to test parquet file
        n_examples: Number of examples to evaluate (None = all)
        output_file: Optional path to save detailed results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Math Agent Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test file: {test_file}")
    print()
    
    # Load test data
    test_df = pd.read_parquet(test_file)
    if n_examples:
        test_df = test_df.head(n_examples)
    
    print(f"Evaluating on {len(test_df)} examples...")
    print()
    
    # Create client
    # Note: For local checkpoints, you'll need to serve them with vLLM first
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",  # Adjust as needed
        api_key="dummy"
    )
    
    # Evaluate each example
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        result = await evaluate_single_problem(
            client=client,
            problem=row.to_dict(),
            model=checkpoint_path,
        )
        results.append(result)
    
    # Compute metrics
    predictions = [r["predicted"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]
    responses = [r["full_response"] for r in results]
    
    metrics = evaluate_batch(predictions, ground_truths, responses)
    
    # Additional metrics
    total = len(results)
    metrics["total_examples"] = total
    metrics["correct_examples"] = sum(r["correct"] for r in results)
    metrics["tool_usage_rate"] = sum(r["used_calculator"] for r in results) / total
    metrics["avg_tool_calls"] = sum(r["num_tool_calls"] for r in results) / total
    
    # Error analysis
    errors = [r for r in results if not r["correct"]]
    if errors:
        # Categorize errors
        no_answer_format = sum(1 for e in errors if not e["predicted"])
        wrong_calculation = sum(
            1 for e in errors 
            if e["predicted"] and e["used_calculator"]
        )
        no_tool_use = sum(
            1 for e in errors 
            if e["predicted"] and not e["used_calculator"]
        )
        
        metrics["error_analysis"] = {
            "total_errors": len(errors),
            "no_answer_format": no_answer_format,
            "wrong_calculation": wrong_calculation,
            "no_tool_use": no_tool_use,
        }
    
    # Print results
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Average Reward: {metrics['avg_reward']:.3f}")
    print(f"Format Compliance: {metrics['format_compliance']:.2%}")
    print(f"Tool Usage Rate: {metrics['tool_usage_rate']:.2%}")
    print(f"Avg Tool Calls: {metrics['avg_tool_calls']:.2f}")
    
    if "error_analysis" in metrics:
        print("\nError Analysis:")
        ea = metrics["error_analysis"]
        print(f"  Total Errors: {ea['total_errors']}")
        print(f"  No Answer Format: {ea['no_answer_format']}")
        print(f"  Wrong Calculation: {ea['wrong_calculation']}")
        print(f"  No Tool Use: {ea['no_tool_use']}")
    
    # Show some examples
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    
    # Show 3 correct and 3 incorrect
    correct_samples = [r for r in results if r["correct"]][:3]
    incorrect_samples = [r for r in results if not r["correct"]][:3]
    
    print("\nCorrect Examples:")
    for i, sample in enumerate(correct_samples, 1):
        print(f"\n{i}. Question: {sample['question'][:80]}...")
        print(f"   Answer: {sample['ground_truth']}")
        print(f"   Predicted: {sample['predicted']}")
        print(f"   Tool Calls: {sample['num_tool_calls']}")
    
    print("\nIncorrect Examples:")
    for i, sample in enumerate(incorrect_samples, 1):
        print(f"\n{i}. Question: {sample['question'][:80]}...")
        print(f"   Answer: {sample['ground_truth']}")
        print(f"   Predicted: {sample['predicted']}")
        print(f"   Tool Calls: {sample['num_tool_calls']}")
    
    # Save detailed results if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                "metrics": metrics,
                "results": results,
            }, f, indent=2)
        
        print(f"\n✓ Detailed results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math agent on GSM8K test set"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint or model name"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="data/gsm8k_test.parquet",
        help="Path to test data file"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save detailed results JSON"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible API"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    asyncio.run(evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_file=args.test_file,
        n_examples=args.n_examples,
        output_file=args.output,
    ))


if __name__ == "__main__":
    main()