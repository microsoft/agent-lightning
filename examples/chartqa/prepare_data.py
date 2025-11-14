# Copyright (c) Microsoft. All rights reserved.

"""
Download and prepare ChartQA dataset from HuggingFace.

This script:
1. Downloads ChartQA from HuggingFace (HuggingFaceM4/ChartQA)
2. Saves images locally to data/images/
3. Converts to Parquet format for AgentLightning training

No HuggingFace token required (public dataset).
"""

from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


def prepare_chartqa_from_hf():
    """
    Download ChartQA from HuggingFace and prepare for training.

    Dataset: HuggingFaceM4/ChartQA (public, no token needed)
    Structure:
        - train: ~18k samples
        - test: ~2k samples
        - Each sample: {'image': PIL.Image, 'query': str, 'label': str/int}
    """
    print("=" * 60)
    print("ChartQA Dataset Preparation (HuggingFace)")
    print("=" * 60)
    print()
    print("Downloading from: HuggingFaceM4/ChartQA")
    print("This is a public dataset, no HF token required.")
    print()

    # Create directories
    data_dir = Path("data")
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    print("[1/3] Downloading dataset from HuggingFace...")
    print("This may take a few minutes on first run (images will be cached)...")
    try:
        dataset = load_dataset("HuggingFaceM4/ChartQA")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Install datasets: pip install datasets")
        print("3. Try: huggingface-cli login (if rate limited)")
        raise

    print("✓ Downloaded dataset")
    print(f"  Train samples: {len(dataset['train'])}")  # type: ignore
    print(f"  Test samples: {len(dataset['test'])}")  # type: ignore
    print()

    # Process train split
    print("[2/3] Processing training split...")
    train_tasks = []
    for idx, item in enumerate(tqdm(dataset["train"], desc="Saving train images")):  # type: ignore
        # Save image locally
        image_filename = f"train_{idx:06d}.png"
        image_path = images_dir / image_filename
        if not image_path.exists():
            item["image"].save(image_path)

        # Create task entry
        task = {
            "id": f"train_{idx}",
            "image_path": f"images/{image_filename}",  # Relative to data/
            "question": item["query"],
            "answer": str(item["label"]),  # Ensure string format
        }
        train_tasks.append(task)

    # Save to parquet
    train_df = pd.DataFrame(train_tasks)
    train_output = data_dir / "train_chartqa.parquet"
    train_df.to_parquet(train_output, index=False)
    print(f"✓ Saved {len(train_tasks)} training tasks to {train_output}")
    print(f"  Sample: {train_tasks[0]}")
    print()

    # Process test split
    print("[3/3] Processing test split...")
    test_tasks = []
    for idx, item in enumerate(tqdm(dataset["test"], desc="Saving test images")):  # type: ignore
        # Save image locally
        image_filename = f"test_{idx:06d}.png"
        image_path = images_dir / image_filename
        if not image_path.exists():
            item["image"].save(image_path)

        # Create task entry
        task = {
            "id": f"test_{idx}",
            "image_path": f"images/{image_filename}",  # Relative to data/
            "question": item["query"],
            "answer": str(item["label"]),
        }
        test_tasks.append(task)

    # Save to parquet
    test_df = pd.DataFrame(test_tasks)
    test_output = data_dir / "test_chartqa.parquet"
    test_df.to_parquet(test_output, index=False)
    print(f"✓ Saved {len(test_tasks)} test tasks to {test_output}")
    print(f"  Sample: {test_tasks[0]}")
    print()

    print("=" * 60)
    print("✓ Data preparation complete!")
    print("=" * 60)
    print()
    print("Dataset statistics:")
    print(f"  Training:   {len(train_tasks):,} tasks")
    print(f"  Test:       {len(test_tasks):,} tasks")
    print()
    print("Output files:")
    print(f"  - {train_output}")
    print(f"  - {test_output}")
    print(f"  - {images_dir}/ ({len(train_tasks) + len(test_tasks):,} images)")
    print()
    print("Next steps:")
    print("  1. Start vLLM: vllm serve Qwen/Qwen2-VL-2B-Instruct --port 8000")
    print("  2. Export: export OPENAI_API_BASE=http://localhost:8000/v1")
    print("  3. Train: python train_chartqa_agent.py fast")


def main():
    """Main entry point."""
    try:
        prepare_chartqa_from_hf()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have installed:")
        print("  pip install datasets pillow pandas pyarrow tqdm")
        raise


if __name__ == "__main__":
    main()
