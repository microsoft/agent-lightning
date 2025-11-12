# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd


def _ensure_outdir(base: pathlib.Path) -> None:
    (base / "data").mkdir(parents=True, exist_ok=True)


def _validate_records(records: List[Dict[str, Any]]) -> None:
    """
    Validate that records match the required schema for training.

    Required fields (mirroring patterns from SQL agent docs):
    - question: The user query/task instruction.
    - app_id: The application/environment identifier.
    - ground_truth: The expected action/output (analogous to a SQL query).
    Optional fields:
    - meta: Arbitrary metadata blob.
    """
    required = {"question", "app_id", "ground_truth"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(f"Record {i} is missing required fields: {sorted(missing)}")
        if not isinstance(rec["question"], str) or not isinstance(rec["app_id"], str):
            raise ValueError(f"Record {i} fields 'question' and 'app_id' must be strings")
        if not isinstance(rec["ground_truth"], str):
            raise ValueError(f"Record {i} field 'ground_truth' must be a string")


def _load_input(path: pathlib.Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        try:
            # Try JSONL
            df = pd.read_json(path, lines=True)
        except ValueError:
            # Fallback to normal JSON
            df = pd.read_json(path)
    elif path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".parquet"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return df.to_dict("records")


def _write_parquet(records: List[Dict[str, Any]], out_path: pathlib.Path) -> None:
    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


def create_toy_dataset(n_train: int = 8, n_test: int = 2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Create a small toy dataset suitable for CI or local smoke tests.
    """
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for i in range(n_train):
        train.append(
            {
                "question": f"Generate an ADK action for task {i}",
                "app_id": "sample_calendar_app",
                "ground_truth": f"create_event(title='Task {i}', date='2025-01-{(i%28)+1:02d}')",
                "meta": {"priority": "normal", "split": "train"},
            }
        )
    for j in range(n_test):
        test.append(
            {
                "question": f"Generate an ADK action for validation {j}",
                "app_id": "sample_calendar_app",
                "ground_truth": f"create_event(title='Validation {j}', date='2025-02-{(j%28)+1:02d}')",
                "meta": {"priority": "high", "split": "test"},
            }
        )
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ADK datasets (Parquet).")
    parser.add_argument(
        "--train",
        type=str,
        default="",
        help="Path to training data (jsonl/json/csv/parquet). If omitted with --generate-toy, a toy dataset is generated.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="Path to test/validation data (jsonl/json/csv/parquet). If omitted with --generate-toy, a toy dataset is generated.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Output base directory (Parquet files will be placed under OUTDIR/data).",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="data/train.parquet",
        help="Relative output path for training parquet under OUTDIR (default: data/train.parquet).",
    )
    parser.add_argument(
        "--test-out",
        type=str,
        default="data/test.parquet",
        help="Relative output path for test parquet under OUTDIR (default: data/test.parquet).",
    )
    parser.add_argument(
        "--generate-toy",
        action="store_true",
        help="Generate a small toy dataset if --train/--test not provided.",
    )
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir).resolve()
    _ensure_outdir(outdir)

    train_records: Optional[List[Dict[str, Any]]] = None
    test_records: Optional[List[Dict[str, Any]]] = None

    if args.train:
        train_records = _load_input(pathlib.Path(args.train))
    if args.test:
        test_records = _load_input(pathlib.Path(args.test))

    if (train_records is None or test_records is None) and args.generate_toy:
        toy_train, toy_test = create_toy_dataset()
        train_records = train_records or toy_train
        test_records = test_records or toy_test

    if train_records is None or test_records is None:
        raise SystemExit(
            "No input provided. Pass --train and --test, or use --generate-toy to create a toy dataset."
        )

    _validate_records(train_records)
    _validate_records(test_records)

    train_out = outdir / args.train_out
    test_out = outdir / args.test_out
    train_out.parent.mkdir(parents=True, exist_ok=True)
    test_out.parent.mkdir(parents=True, exist_ok=True)

    _write_parquet(train_records, train_out)
    _write_parquet(test_records, test_out)


if __name__ == "__main__":
    main()


