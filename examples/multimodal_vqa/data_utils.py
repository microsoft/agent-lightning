# Copyright (c) Microsoft. All rights reserved.

"""
Data utilities for VQAv2 dataset processing.

This script converts VQAv2 JSON format to parquet files for efficient training.

VQAv2 Dataset Structure:
- Questions: {question_id, image_id, question}
- Annotations: {question_id, answers: [{answer, answer_confidence}]}
- Images: COCO train2014/val2014 directories

Output Parquet Schema:
- id: str (question_id)
- image_path: str (local path to COCO image)
- question: str
- answer: str (most common answer from multiple annotators)

Usage:
    # Convert train split
    python data_utils.py \\
        --questions data/vqav2/v2_OpenEnded_mscoco_train2014_questions.json \\
        --annotations data/vqav2/v2_mscoco_train2014_annotations.json \\
        --image-dir data/vqav2/images/train2014 \\
        --output data/vqav2_train.parquet

    # Convert val split
    python data_utils.py \\
        --questions data/vqav2/v2_OpenEnded_mscoco_val2014_questions.json \\
        --annotations data/vqav2/v2_mscoco_val2014_annotations.json \\
        --image-dir data/vqav2/images/val2014 \\
        --output data/vqav2_val.parquet
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def get_most_common_answer(answers: list[dict]) -> str:
    """
    Get the most common answer from VQAv2 annotations.

    VQAv2 has 10 annotators per question. We select the most common answer
    as the ground truth for training.

    Args:
        answers: List of answer dicts with "answer" and "answer_confidence" keys

    Returns:
        Most common answer string

    Example:
        >>> answers = [
        ...     {"answer": "cat", "answer_confidence": "yes"},
        ...     {"answer": "cat", "answer_confidence": "yes"},
        ...     {"answer": "kitten", "answer_confidence": "maybe"}
        ... ]
        >>> get_most_common_answer(answers)
        'cat'
    """
    answer_texts = [ans["answer"].lower().strip() for ans in answers]
    counter = Counter(answer_texts)
    most_common = counter.most_common(1)[0][0]
    return most_common


def convert_vqav2_to_parquet(
    questions_path: str,
    annotations_path: str,
    image_dir: str,
    output_path: str,
    limit: int | None = None,
) -> None:
    """
    Convert VQAv2 JSON files to parquet format.

    Args:
        questions_path: Path to questions JSON file
        annotations_path: Path to annotations JSON file
        image_dir: Directory containing COCO images
        output_path: Output parquet file path
        limit: Maximum number of samples (for debugging)

    Output Schema:
        - id: str (question_id)
        - image_path: str (absolute path to image)
        - question: str
        - answer: str (most common answer)

    Note:
        Images must exist in image_dir. Missing images will be skipped.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas pyarrow")

    print(f"Loading questions from {questions_path}...")
    with open(questions_path) as f:
        questions_data = json.load(f)

    print(f"Loading annotations from {annotations_path}...")
    with open(annotations_path) as f:
        annotations_data = json.load(f)

    # Build annotation lookup
    print("Building annotation lookup...")
    annotations_by_id = {
        ann["question_id"]: ann["answers"]
        for ann in annotations_data["annotations"]
    }

    # Process questions
    print("Processing questions...")
    image_dir_path = Path(image_dir)
    if not image_dir_path.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    rows = []
    skipped = 0

    for q in questions_data["questions"]:
        question_id = q["question_id"]
        image_id = q["image_id"]
        question = q["question"]

        # Get annotations
        if question_id not in annotations_by_id:
            skipped += 1
            continue

        answers = annotations_by_id[question_id]
        most_common_answer = get_most_common_answer(answers)

        # Construct image path
        # COCO image naming: COCO_train2014_000000123456.jpg
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        if "val2014" in str(image_dir):
            image_filename = f"COCO_val2014_{image_id:012d}.jpg"

        image_path = image_dir_path / image_filename

        # Check if image exists
        if not image_path.exists():
            skipped += 1
            continue

        # Add row
        rows.append({
            "id": str(question_id),
            "image_path": str(image_path.resolve()),
            "question": question,
            "answer": most_common_answer,
        })

        # Check limit
        if limit and len(rows) >= limit:
            break

    # Create DataFrame and save
    print(f"Creating parquet file with {len(rows)} samples...")
    df = pd.DataFrame(rows)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"Total questions processed: {len(questions_data['questions'])}")
    print(f"Valid samples: {len(rows)}")
    print(f"Skipped (missing annotations or images): {skipped}")
    print(f"Output file: {output_path}")
    print(f"File size: {output_path_obj.stat().st_size / 1024 / 1024:.2f} MB")
    print("=" * 60)

    # Show sample
    print("\nSample data:")
    print(df.head(3).to_string(index=False))


def verify_parquet(parquet_path: str, n_samples: int = 5) -> None:
    """
    Verify parquet file and show sample data.

    Args:
        parquet_path: Path to parquet file
        n_samples: Number of samples to display
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required. Install with: pip install pandas pyarrow")

    print(f"Loading parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)

    print("\n" + "=" * 60)
    print("Parquet File Info")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nSample data:\n")
    print(df.head(n_samples).to_string(index=False))
    print("=" * 60)

    # Check image paths
    print("\nVerifying image paths...")
    missing = 0
    for _, row in df.head(100).iterrows():  # Check first 100
        if not Path(row["image_path"]).exists():
            missing += 1

    if missing > 0:
        print(f"[WARNING] {missing}/100 image paths do not exist!")
    else:
        print("All checked image paths exist.")


def main():
    parser = argparse.ArgumentParser(description="Convert VQAv2 JSON to parquet format")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert JSON to parquet")
    convert_parser.add_argument(
        "--questions",
        type=str,
        required=True,
        help="Path to questions JSON file",
    )
    convert_parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to annotations JSON file",
    )
    convert_parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing COCO images",
    )
    convert_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path",
    )
    convert_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples (for debugging)",
    )

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify parquet file")
    verify_parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to parquet file to verify",
    )
    verify_parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of samples to display",
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_vqav2_to_parquet(
            questions_path=args.questions,
            annotations_path=args.annotations,
            image_dir=args.image_dir,
            output_path=args.output,
            limit=args.limit,
        )
    elif args.command == "verify":
        verify_parquet(
            parquet_path=args.file,
            n_samples=args.n_samples,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
