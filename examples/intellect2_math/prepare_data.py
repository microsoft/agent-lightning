# Copyright (c) Microsoft. All rights reserved.

import argparse
import json
from pathlib import Path

from datasets import load_dataset

DATASET = "PrimeIntellect/INTELLECT-2-only-math"
REVISION = "169e067d356aaca3cd8e860c7b900283df02f53d"


def format_row(row):
    return {
        "question": row["prompt"].strip(),
        "result": json.loads(row["verification_info"])["ground_truth"],
        "problem_id": row["problem_id"],
    }


def prepare_splits(dataset, val_size=500, seed=42):
    dataset = dataset.map(format_row, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda row: bool(row["result"]))
    return dataset.train_test_split(test_size=val_size, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("data"))
    parser.add_argument("--val-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = load_dataset(DATASET, revision=REVISION, split="train")
    splits = prepare_splits(dataset, args.val_size, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits["train"].to_parquet(args.output_dir / "train.parquet")
    splits["test"].to_parquet(args.output_dir / "test.parquet")


if __name__ == "__main__":
    main()
