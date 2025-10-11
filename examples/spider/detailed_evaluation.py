#!/usr/bin/env python3
"""
Enhanced evaluation script for Spider SQL Agent with detailed metrics.

This script provides comprehensive evaluation including:
- Execution accuracy by difficulty levels
- Exact matching accuracy  
- Partial matching scores for SQL components
- Turn-based accuracy analysis

Usage:
    python detailed_evaluation.py --gold_file gold.txt --pred_file pred.txt --db_dir databases/
"""

import argparse
import json
import os
import sys
from typing import Dict, Any

from spider_eval.evaluation import evaluate
from spider_eval.process_sql import get_schema


def load_json_data(file_path: str) -> list:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]


def create_gold_file(data: list, output_path: str) -> None:
    """Create gold file in the format expected by evaluation script."""
    with open(output_path, 'w') as f:
        for item in data:
            f.write(f"{item['query']}\t{item['db_id']}\n")
            f.write("\n")  # Empty line after each query


def create_pred_file(predictions: list, output_path: str) -> None:
    """Create prediction file in the format expected by evaluation script."""
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")
            f.write("\n")  # Empty line after each prediction


def evaluate_spider_detailed(
    gold_data: list, 
    predictions: list, 
    db_dir: str,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Run detailed evaluation on Spider dataset.
    
    Args:
        gold_data: List of gold data items with 'query' and 'db_id' fields
        predictions: List of predicted SQL queries
        db_dir: Directory containing database files
        output_file: Optional file to save evaluation results
    
    Returns:
        Dictionary containing detailed evaluation metrics
    """
    
    # Create temporary files for evaluation
    gold_file = "/tmp/gold_eval.txt"
    pred_file = "/tmp/pred_eval.txt"
    
    create_gold_file(gold_data, gold_file)
    create_pred_file(predictions, pred_file)
    
    # Load kmaps (assuming standard location)
    kmaps = {}
    for db_name in set(item['db_id'] for item in gold_data):
        db_path = os.path.join(db_dir, db_name, f"{db_name}.sqlite")
        if os.path.exists(db_path):
            schema = get_schema(db_path)
            kmaps[db_name] = {table: [] for table in schema}
    
    print("="*80)
    print("DETAILED SPIDER EVALUATION RESULTS")
    print("="*80)
    print(f"Dataset size: {len(gold_data)} samples")
    print(f"Database directory: {db_dir}")
    print()
    
    # Run evaluation with detailed metrics
    evaluate(
        gold=gold_file,
        predict=pred_file,
        db_dir=db_dir,
        etype="all",  # Both execution and exact matching
        kmaps=kmaps,
        plug_value=False,
        keep_distinct=False,
        progress_bar_for_each_datapoint=False
    )
    
    # Clean up temporary files
    os.remove(gold_file)
    os.remove(pred_file)
    
    return {"evaluation_completed": True}


def main():
    parser = argparse.ArgumentParser(description="Detailed evaluation for Spider SQL Agent")
    parser.add_argument("--gold_file", required=True, help="Path to gold data JSON file")
    parser.add_argument("--pred_file", required=True, help="Path to predictions JSON file") 
    parser.add_argument("--db_dir", required=True, help="Path to database directory")
    parser.add_argument("--output", help="Optional output file for results")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gold_file):
        print(f"Error: Gold file {args.gold_file} not found")
        sys.exit(1)
        
    if not os.path.exists(args.pred_file):
        print(f"Error: Prediction file {args.pred_file} not found")
        sys.exit(1)
        
    if not os.path.exists(args.db_dir):
        print(f"Error: Database directory {args.db_dir} not found")
        sys.exit(1)
    
    # Load data
    gold_data = load_json_data(args.gold_file)
    predictions = []
    
    # Load predictions (assuming one query per line)
    with open(args.pred_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(line)
    
    if len(gold_data) != len(predictions):
        print(f"Warning: Mismatch in data sizes - Gold: {len(gold_data)}, Predictions: {len(predictions)}")
        min_size = min(len(gold_data), len(predictions))
        gold_data = gold_data[:min_size]
        predictions = predictions[:min_size]
        print(f"Using first {min_size} samples for evaluation")
    
    # Run evaluation
    results = evaluate_spider_detailed(
        gold_data=gold_data,
        predictions=predictions, 
        db_dir=args.db_dir,
        output_file=args.output
    )
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()