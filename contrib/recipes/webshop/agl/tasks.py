# Copyright (c) Microsoft. All rights reserved.

"""Task loading utilities for WebShop training.

This module provides utilities for loading WebShop tasks from various sources:
- Sample tasks matching the TypeScript sample-tasks.ts
- JSON files
- Parquet files

To generate a full task set from WebShop data, use `generate_tasks.py`:
    python agl/generate_tasks.py --output agl/webshop_tasks.json

Then load in training:
    python agl/run_training.py qwen --tasks-file agl/webshop_tasks.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]

# Default location for sample tasks file (same directory as this module)
_SAMPLE_TASKS_FILE = Path(__file__).parent / "sample_tasks.json"


def load_sample_tasks(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load sample tasks matching src/data/sample-tasks.ts.

    Args:
        path: Optional path to sample tasks JSON file. Defaults to
            `sample_tasks.json` in the same directory as this module.

    Returns:
        List of task dictionaries with task_id, instruction, and target_attributes.
    """
    tasks_path = path or _SAMPLE_TASKS_FILE
    return load_tasks_from_file(tasks_path)


def load_tasks_from_file(path: Path) -> List[Dict[str, Any]]:
    """Load tasks from a JSON or Parquet file.

    Args:
        path: Path to the tasks file (JSON or Parquet format).

    Returns:
        List of task dictionaries.

    Raises:
        ValueError: If the file format is not supported.
        ImportError: If loading Parquet but pandas is not installed.
    """
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    elif path.suffix == ".parquet":
        if pd is None:
            raise ImportError("pandas is required to load Parquet files. " "Install with: pip install pandas pyarrow")
        df = pd.read_parquet(path)  # type: ignore[reportUnknownMemberType]
        return df.to_dict(orient="records")  # type: ignore[return-value]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .parquet")
