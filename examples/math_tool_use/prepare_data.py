# Copyright (c) Microsoft. All rights reserved.

"""
Data Preparation Script for Math Agent

Downloads and prepares the GSM8K dataset for training.
GSM8K contains grade school math word problems with numerical answers.

Dataset: https://github.com/openai/grade-school-math
Paper: Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)
"""

import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset


def extract_numeric_answer(answer_text: str) -> str:
    """
    Extract the numerical answer from GSM8K format.
    
    GSM8K answers are in the format:
    "Step-by-step solution text...
    #### 42"
    
    We want to extract just the number after "####".
    
    Args:
        answer_text: The full answer text from GSM8K
        
    Returns:
        The numerical answer as a string
    """

    # Look for the pattern "#### NUMBER"
    match = re.search(r'####\s*(-?\d+\.?\d*)', answer_text)
    if match:
        return match.group(1).strip()
    
    # Fallback: try to find any number in the text
    numbers = re.findall(r'-?\d+\.?\d*', answer_text)
    if numbers:
        return numbers[-1]
    
    return "0"  # Default if no number found


def prepare_gsm8k_dataset(output_dir: str = "data", test_size: int = 1319):
    """
    Download and prepare the GSM8K dataset.
    
    Creates two files:
    - gsm8k_train.parquet: Training set (~7K examples)
    - gsm8k_test.parquet: Test set (~1.3K examples)
    
    Args:
        output_dir: Directory to save the processed data
        test_size: Number of examples to use for testing
    """
    print("=" * 60)
    print("GSM8K Dataset Preparation")
    print("=" * 60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_path.absolute()}")
    
    # Load dataset from Hugging Face
    print("\nDownloading GSM8K dataset from Hugging Face...")
    try:
        dataset = load_dataset("gsm8k", "main")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTrying alternative loading method...")
        dataset = load_dataset("openai/gsm8k", "main")
    
    print(f"✓ Dataset loaded successfully")
    print(f"  - Train split: {len(dataset['train'])} examples")
    print(f"  - Test split: {len(dataset['test'])} examples")
    
    # Process training data
    print("\nProcessing training data...")
    train_data = []
    for example in dataset['train']:

        train_data.append({
            'question': example['question'],
            'answer': extract_numeric_answer(example['answer']),
            'full_solution': example['answer'],  # Keep full solution for reference
        })
    
    train_df = pd.DataFrame(train_data)
    train_path = output_path / "gsm8k_train.parquet"
    train_df.to_parquet(train_path, index=False)
    print(f"✓ Saved training data to: {train_path}")
    print(f"  - {len(train_df)} examples")
    
    # Process test data
    print("\nProcessing test data...")
    test_data = []

    for example in dataset['test'][:test_size]:
        
          # Limit test set size
        test_data.append({
            'question': example['question'],
            'answer': extract_numeric_answer(example['answer']),
            'full_solution': example['answer'],
        })
    
    test_df = pd.DataFrame(test_data)
    test_path = output_path / "gsm8k_test.parquet"
    test_df.to_parquet(test_path, index=False)
    print(f"✓ Saved test data to: {test_path}")
    print(f"  - {len(test_df)} examples")
    
    # Display sample examples
    print("\n" + "=" * 60)
    print("Sample Examples:")
    print("=" * 60)
    
    for i, row in train_df.head(3).iterrows():
        print(f"\nExample {i + 1}:")
        print(f"Question: {row['question'][:100]}...")
        print(f"Answer: {row['answer']}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics:")
    print("=" * 60)
    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"\nAnswer distribution (train):")
    print(f"  Min: {train_df['answer'].astype(float).min()}")
    print(f"  Max: {train_df['answer'].astype(float).max()}")
    print(f"  Mean: {train_df['answer'].astype(float).mean():.2f}")
    print(f"  Median: {train_df['answer'].astype(float).median():.2f}")
    
    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start Ray cluster: bash ../../scripts/restart_ray.sh")
    print("2. Run agent workers: python math_agent.py")
    print("3. Start training: bash train.sh")


def verify_dataset(data_dir: str = "data"):
    """
    Verify that the dataset files exist and are readable.
    
    Args:
        data_dir: Directory containing the dataset files
    """
    print("\n" + "=" * 60)
    print("Verifying Dataset")
    print("=" * 60)
    
    data_path = Path(data_dir)
    train_file = data_path / "gsm8k_train.parquet"
    test_file = data_path / "gsm8k_test.parquet"
    
    if not train_file.exists():
        print(f"✗ Training file not found: {train_file}")
        return False
    
    if not test_file.exists():
        print(f"✗ Test file not found: {test_file}")
        return False
    
    try:
        train_df = pd.read_parquet(train_file)
        test_df = pd.read_parquet(test_file)
        
        print(f"✓ Training file: {train_file}")
        print(f"  - {len(train_df)} examples")
        print(f"  - Columns: {list(train_df.columns)}")
        
        print(f"\n✓ Test file: {test_file}")
        print(f"  - {len(test_df)} examples")
        print(f"  - Columns: {list(test_df.columns)}")
        
        # Verify required columns
        required_cols = {'question', 'answer'}
        if not required_cols.issubset(train_df.columns):
            print(f"✗ Missing required columns in training data")
            return False
        
        if not required_cols.issubset(test_df.columns):
            print(f"✗ Missing required columns in test data")
            return False
        
        print("\n✓ Dataset verification passed!")
        return True
        
    except Exception as e:
        print(f"✗ Error reading dataset files: {e}")
        return False


if __name__ == "__main__":
    """
    Main entry point for data preparation.
    
    Usage:
        python prepare_data.py              # Prepare dataset
        python prepare_data.py --verify     # Verify existing dataset
    """
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        # Verify existing dataset
        verify_dataset()
    else:
        # Prepare new dataset
        prepare_gsm8k_dataset()
        
        # Verify the prepared dataset
        print("\n")
        verify_dataset()