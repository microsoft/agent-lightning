#!/usr/bin/env python3
# Copyright (c) Microsoft. All rights reserved.

"""
LLM-as-a-Judge.

Usage:
    # After running benchmark, use this to evaluate with LLM judge
    python judge.py \
        --input output/20260128_xxx_physics_xxx/trajectory.json \
        --judge_model qwen3-30B-A3B-instruct \
        --judge_base_url http://localhost:8000/v1 \
        --num_workers 32
"""

import json
import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def create_judge_prompt(question: str, ground_truth: str, model_answer: str) -> str:
    """Create LLM judge prompt"""
    # Remove thinking content if present
    if "</think>" in model_answer:
        _, sep, after = model_answer.partition("</think>")
        model_answer = after.strip()

    prompt = f"""Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {ground_truth}
CANDIDATE ANSWER TO ASSESS: {model_answer}

Reply only with CORRECT or INCORRECT."""
    
    return prompt


def extract_verdict(response_text: str) -> str:
    """
    Extract verdict from model response.
    Strategy: find the last occurrence of CORRECT or INCORRECT.
    
    Returns: "CORRECT", "INCORRECT", or "UNKNOWN"
    """
    text_upper = response_text.upper()
    
    # Find all positions of CORRECT and INCORRECT
    correct_positions = []
    incorrect_positions = []
    
    # Find all "CORRECT" positions (not part of "INCORRECT")
    pos = 0
    while True:
        pos = text_upper.find("CORRECT", pos)
        if pos == -1:
            break
        if pos == 0 or text_upper[pos-2:pos] != "IN":
            correct_positions.append(pos)
        pos += 1
    
    # Find all "INCORRECT" positions
    pos = 0
    while True:
        pos = text_upper.find("INCORRECT", pos)
        if pos == -1:
            break
        incorrect_positions.append(pos)
        pos += 1
    
    # Get the last occurrence
    last_correct = max(correct_positions) if correct_positions else -1
    last_incorrect = max(incorrect_positions) if incorrect_positions else -1
    
    if last_correct == -1 and last_incorrect == -1:
        return "UNKNOWN"
    
    if last_incorrect > last_correct:
        return "INCORRECT"
    else:
        return "CORRECT"


def judge_single(args: dict) -> dict:
    """Judge a single problem"""
    from llm_in_sandbox.openai_client import create_chat_completion

    problem_id = args["problem_id"]
    question = args["question"]
    ground_truth = args["ground_truth"]
    model_answer = args["model_answer"]
    judge_config = args["judge_config"]
    debug = args.get("debug", False)
    
    prompt = create_judge_prompt(question, ground_truth, model_answer)
    
    # Debug: print first few prompts
    if debug:
        print(f"\n{'='*80}")
        print(f"[DEBUG] Problem ID: {problem_id}")
        print(f"[DEBUG] Ground Truth: {ground_truth}")
        print(f"[DEBUG] Model Answer: {model_answer}")
        print(f"[DEBUG] Judge Config: model={judge_config['model']}, temp={judge_config.get('temperature')}, max_tokens={judge_config.get('max_tokens')}")
        print(f"[DEBUG] Full Prompt:\n{prompt}")
        print(f"{'='*80}\n")
    
    try:
        response = create_chat_completion(
            model=judge_config["model"],
            messages=[{"role": "user", "content": prompt}],
            temperature=judge_config.get("temperature", 0.7),
            max_tokens=judge_config.get("max_tokens", 16384),
            timeout=600,
            base_url=judge_config.get("base_url"),
            api_key=judge_config.get("api_key"),
        )
        judge_response = response.choices[0].message.content or ""
        verdict = extract_verdict(judge_response)
        score = 1.0 if verdict == "CORRECT" else 0.0
        
        if debug:
            print(f"[DEBUG] Judge Response: {judge_response}")
            print(f"[DEBUG] Verdict: {verdict}, Score: {score}")
        
        return {
            "problem_id": problem_id,
            "verdict": verdict,
            "score": score,
            "judge_response": judge_response,
            "error": None,
        }
    except Exception as e:
        return {
            "problem_id": problem_id,
            "verdict": "ERROR",
            "score": 0.0,
            "judge_response": "",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge evaluation")
    parser.add_argument("--input", required=True, help="Path to trajectory.json from benchmark run")
    parser.add_argument("--judge_model", required=True, help="Judge model name")
    parser.add_argument("--judge_base_url", default=None, help="Judge model API base URL")
    parser.add_argument("--api_key", default=None, help="API key")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of parallel workers")
    parser.add_argument("--output", default=None, help="Output file (default: input_judged.json)")
    parser.add_argument("--debug", type=int, default=0, help="Number of samples to debug (print prompt and response)")
    parser.add_argument("--num_repeats", type=int, default=4, help="Number of times to repeat each judgment (for reducing variance)")
    args = parser.parse_args()
    
    # Auto-add openai/ prefix for custom LLM endpoints
    if args.judge_base_url and not args.judge_model.startswith(("openai/", "anthropic/", "azure/", "hosted_vllm/")):
        args.judge_model = f"openai/{args.judge_model}"
        print(f"Auto-added 'openai/' prefix to judge model: {args.judge_model}")
    
    # Set dummy API key if not provided (some servers don't need auth)
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = str(args.api_key)
    elif not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "dummy"
    
    # Load trajectory data
    input_path = Path(args.input)
    with open(input_path, "r") as f:
        trajectory_data = json.load(f)
    
    print(f"📂 Loaded {len(trajectory_data)} problems from {input_path}")
    
    # Prepare judge config
    judge_config = {
        "model": args.judge_model,
        "base_url": args.judge_base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    
    # Prepare tasks (repeat num_repeats times for variance reduction)
    tasks = []
    for run_idx in range(args.num_repeats):
        for i, (problem_id, data) in enumerate(trajectory_data.items()):
            # problem_statement is now in trajectory.json
            question = data.get("problem_statement", "")
            tasks.append({
                "problem_id": problem_id,
                "run_index": run_idx,
                "question": question,
                "ground_truth": data["ground_truth"],
                "model_answer": data["agent_answer"],
                "judge_config": judge_config,
                "debug": run_idx == 0 and i < args.debug,  # Debug first N samples (only in first run)
            })
    
    print(f"🚀 Starting LLM-as-Judge evaluation")
    print(f"   Judge Model: {args.judge_model}")
    print(f"   Workers: {args.num_workers}")
    if args.num_repeats > 1:
        print(f"   Repeats: {args.num_repeats} (for variance reduction)")
    
    # Run judge in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(judge_single, task): task["problem_id"] for task in tasks}
        
        with tqdm(total=len(futures), desc="Judging") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
                pbar.set_postfix(
                    score=f"{sum(r['score'] for r in results) / len(results):.3f}",
                    errors=sum(1 for r in results if r['error']),
                )
    
    # Aggregate results by problem_id (average scores across repeats)
    from collections import defaultdict
    results_by_problem = defaultdict(list)
    for r in results:
        results_by_problem[r["problem_id"]].append(r)
    
    aggregated_results = {}
    for problem_id, runs in results_by_problem.items():
        avg_score = sum(r["score"] for r in runs) / len(runs)
        # Majority vote for verdict
        verdicts = [r["verdict"] for r in runs]
        verdict_counts = {"CORRECT": 0, "INCORRECT": 0, "UNKNOWN": 0, "ERROR": 0}
        for v in verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        majority_verdict = max(verdict_counts, key=verdict_counts.get)
        
        aggregated_results[problem_id] = {
            "problem_id": problem_id,
            "verdict": majority_verdict,
            "score": avg_score,
            "num_runs": len(runs),
            "individual_verdicts": verdicts,
            "individual_scores": [r["score"] for r in runs],
            "error": runs[0].get("error"),  # Keep first error if any
        }
    
    # Compute statistics (on aggregated results)
    num_problems = len(aggregated_results)
    scores = [r["score"] for r in aggregated_results.values()]
    correct = sum(1 for r in aggregated_results.values() if r["verdict"] == "CORRECT")
    incorrect = sum(1 for r in aggregated_results.values() if r["verdict"] == "INCORRECT")
    unknown = sum(1 for r in aggregated_results.values() if r["verdict"] == "UNKNOWN")
    errors = sum(1 for r in aggregated_results.values() if r["error"])
    
    print(f"\n📊 Results:")
    print(f"   Problems: {num_problems}")
    if args.num_repeats > 1:
        print(f"   Total judgments: {len(results)} ({args.num_repeats} repeats × {num_problems} problems)")
    print(f"   Correct: {correct} ({correct/num_problems*100:.1f}%)")
    print(f"   Incorrect: {incorrect}")
    print(f"   Unknown: {unknown}")
    print(f"   Errors: {errors}")
    print(f"   Mean Score: {sum(scores)/len(scores):.4f}")
    
    # Save results
    output_path = args.output or str(input_path).replace(".json", "_judged.json")
    output_data = {
        "judge_model": args.judge_model,
        "num_repeats": args.num_repeats,
        "stats": {
            "num_problems": num_problems,
            "total_judgments": len(results),
            "correct": correct,
            "incorrect": incorrect,
            "unknown": unknown,
            "errors": errors,
            "mean_score": sum(scores) / len(scores),
        },
        "results": aggregated_results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved to: {output_path}")


if __name__ == "__main__":
    main()
