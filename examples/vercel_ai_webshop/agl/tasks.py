# Copyright (c) Microsoft. All rights reserved.

"""Task loading utilities for WebShop training.

This module provides utilities for loading WebShop tasks from various sources:
- Sample tasks matching the TypeScript sample-tasks.ts
- JSON files
- Parquet files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_sample_tasks() -> List[Dict[str, Any]]:
    """Load sample tasks matching src/data/sample-tasks.ts.

    Returns a list of task dictionaries with task_id, instruction, and target_attributes.
    """
    return [
        {
            "task_id": "ws_001",
            "instruction": "I need a red cotton t-shirt for men, size large, under $30",
            "target_attributes": {
                "color": "red",
                "material": "cotton",
                "size": "L",
                "priceMax": 30,
            },
        },
        {
            "task_id": "ws_002",
            "instruction": "Find me black running shorts for men, size medium, athletic style",
            "target_attributes": {
                "color": "black",
                "style": "athletic",
                "size": "M",
            },
        },
        {
            "task_id": "ws_003",
            "instruction": "I'm looking for a gray fleece hoodie for women, size small",
            "target_attributes": {
                "color": "gray",
                "material": "fleece",
                "size": "S",
            },
        },
        {
            "task_id": "ws_004",
            "instruction": "Find white canvas sneakers, size 9, preferably under $50",
            "target_attributes": {
                "color": "white",
                "material": "canvas",
                "size": "9",
                "priceMax": 50,
            },
        },
        {
            "task_id": "ws_005",
            "instruction": "I need navy blue slim fit chino pants, waist 32, length 32",
            "target_attributes": {
                "color": "navy",
                "style": "slim fit",
                "waist": "32",
                "length": "32",
            },
        },
        {
            "task_id": "ws_006",
            "instruction": "Looking for black yoga leggings for women, size medium",
            "target_attributes": {
                "color": "black",
                "style": "yoga",
                "size": "M",
            },
        },
        {
            "task_id": "ws_007",
            "instruction": "Find a white organic cotton blouse for women, size medium",
            "target_attributes": {
                "color": "white",
                "material": "organic cotton",
                "size": "M",
            },
        },
        {
            "task_id": "ws_008",
            "instruction": "I want a navy blue polo shirt for men, size XL, under $50",
            "target_attributes": {
                "color": "navy",
                "style": "polo",
                "size": "XL",
                "priceMax": 50,
            },
        },
    ]


def load_tasks_from_file(path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a JSON or Parquet file.

    Args:
        path: Path to the tasks file (JSON or Parquet format).

    Returns:
        List of task dictionaries.

    Raises:
        ValueError: If the file format is not supported.
    """
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")  # type: ignore
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .parquet")
