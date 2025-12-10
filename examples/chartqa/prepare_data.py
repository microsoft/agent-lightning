# Copyright (c) Microsoft. All rights reserved.

"""Prepare ChartQA dataset from HuggingFace for training."""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset  # pyright: ignore[reportUnknownVariableType]


def prepare_chartqa():
    """Download ChartQA and convert to parquet format."""
    data_dir = Path("data")
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("HuggingFaceM4/ChartQA")

    for split in ["train", "test"]:
        tasks: List[Dict[str, Any]] = []
        dataset_length = len(dataset[split])  # type: ignore
        for idx, item in enumerate(dataset[split]):  # pyright: ignore[reportUnknownArgumentType]
            if idx % 1000 == 0:
                print(f"Processing {split} item {idx} (out of {dataset_length})")
            image_filename = f"{split}_{idx:06d}.png"
            image_path = images_dir / image_filename
            if not image_path.exists():
                item["image"].save(image_path)

            tasks.append(
                {
                    "id": f"{split}_{idx}",
                    "image_path": f"images/{image_filename}",
                    "question": item["query"],
                    "answer": str(item["label"]),
                }
            )

        pd.DataFrame(tasks).to_parquet(data_dir / f"{split}_chartqa.parquet", index=False)  # type: ignore


if __name__ == "__main__":
    prepare_chartqa()
