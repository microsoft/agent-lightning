# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import pathlib
import tempfile
import zipfile
from typing import Any, Dict, List, Tuple

import pandas as pd

# Here we use the same dataset as that used in Spider SQL Spider (examples/spider)
SPIDER_DATASET_URL = "https://drive.google.com/file/d/1oi9J1jZP9TyM35L85CL3qeGWl2jqlnL6/view"

def _ensure_outdir(base: pathlib.Path) -> None:
    (base / "data").mkdir(parents=True, exist_ok=True)


def _validate_records(records: list[dict[str, Any]]) -> None:
    required = {"question", "app_id", "ground_truth"}
    for i, rec in enumerate(records):
        missing = required - set(rec.keys())
        if missing:
            raise ValueError(f"Record {i} is missing required fields: {sorted(missing)}")
        if not isinstance(rec["question"], str) or not isinstance(rec["app_id"], str):
            raise ValueError(f"Record {i} fields 'question' and 'app_id' must be strings")
        if not isinstance(rec["ground_truth"], str):
            raise ValueError(f"Record {i} field 'ground_truth' must be a string")


def _load_input(path: pathlib.Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")
    return df.to_dict("records")


def _write_parquet(records: list[dict[str, Any]], out_path: pathlib.Path) -> None:
    df = pd.DataFrame.from_records(records)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


def _split_dataframe(df: pd.DataFrame, ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < ratio < 1.0:
        raise ValueError("--sample-train-ratio must be between 0 and 1")
    if len(df) < 2:
        raise ValueError("Sample dataset must contain at least two rows to split.")
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    split_idx = max(1, int(len(shuffled) * ratio))
    if split_idx >= len(shuffled):
        split_idx = len(shuffled) - 1
    train_df = shuffled.iloc[:split_idx].copy()
    test_df = shuffled.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Failed to split dataset into non-empty train/test partitions.")
    return train_df, test_df


def _convert_to_adk_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Spider dataset format to ADK format.
    
    Spider columns: db_id, question, query
    ADK columns: question, app_id, ground_truth
    """
    if not all(col in df.columns for col in ["db_id", "question", "query"]):
        raise ValueError("Spider dataset must contain columns: db_id, question, query")
    
    adk_df = pd.DataFrame({
        "question": df["question"],
        "app_id": df["db_id"],
        "ground_truth": df["query"],
    })
    
    if "meta" in df.columns:
        adk_df["meta"] = df["meta"]
    
    return adk_df


def _download_dataset(url: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Download Spider dataset zip, extract, and convert to ADK format."""
    print(f"Downloading Spider dataset from {url}")
    
    try:
        import gdown
    except ImportError:
        raise ImportError("gdown is required to download the Spider dataset. Install with: pip install gdown")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        zip_path = tmp_path / "spider-data.zip"
        
        # Download using gdown (same method as examples/spider)
        print("Downloading zip file...")
        gdown.download(url, str(zip_path), fuzzy=True, quiet=False)
        
        print("Extracting zip file...")
        extract_dir = tmp_path / "spider_data"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        data_dir = extract_dir
        if (extract_dir / "train_spider.json").exists():
            data_dir = extract_dir
        else:
            for subdir in ["spider", "data"]:
                candidate = extract_dir / subdir
                if candidate.exists() and (candidate / "train_spider.json").exists():
                    data_dir = candidate
                    break
        
        train_json = data_dir / "train_spider.json"
        dev_json = data_dir / "dev.json"
        
        if not train_json.exists():
            raise FileNotFoundError(f"train_spider.json not found in extracted archive")
        if not dev_json.exists():
            raise FileNotFoundError(f"dev.json not found in extracted archive")
        
        print(f"Loading {train_json}...")
        train_df = pd.read_json(train_json)
        print(f"Loaded {len(train_df)} training records")
        
        print(f"Loading {dev_json}...")
        dev_df = pd.read_json(dev_json)
        print(f"Loaded {len(dev_df)} validation records")

        train_adk = _convert_to_adk_format(train_df)
        test_adk = _convert_to_adk_format(dev_df)
        
        print(f"Converted to ADK format: {len(train_adk)} train, {len(test_adk)} test records")
        
        return train_adk.to_dict("records"), test_adk.to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ADK datasets (Parquet).")
    parser.add_argument(
        "--train",
        type=str,
        default="",
        help="Path to training data (jsonl/json/csv/parquet). Use --download if omitted.",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="",
        help="Path to test/validation data (jsonl/json/csv/parquet). Use --download if omitted.",
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
        "--download",
        action="store_true",
        help="Download the Spider dataset (same as examples/spider) and convert to ADK format when --train/--test are not provided.",
    )
    args = parser.parse_args()

    outdir = pathlib.Path(args.outdir).resolve()
    _ensure_outdir(outdir)

    train_records: List[Dict[str, Any]] | None = None
    test_records: List[Dict[str, Any]] | None = None

    if args.train:
        train_records = _load_input(pathlib.Path(args.train))
    if args.test:
        test_records = _load_input(pathlib.Path(args.test))

    if (train_records is None or test_records is None) and args.download:
        spider_train, spider_test = _download_dataset(SPIDER_DATASET_URL)
        train_records = train_records or spider_train
        test_records = test_records or spider_test

    if train_records is None or test_records is None:
        raise SystemExit(
            "No input provided. Pass --train and --test, or use --download to fetch the Spider dataset."
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


