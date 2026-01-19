# Copyright (c) Microsoft. All rights reserved.

"""Generate training tasks from WebShop product data.

This script extracts training tasks from the WebShop human instruction dataset.
It produces a JSON file compatible with the training pipeline's --tasks-file argument.

Usage:
    cd examples/vercel_ai_webshop
    python agl/generate_tasks.py --output agl/webshop_tasks.json

    # With custom options:
    python agl/generate_tasks.py --output agl/webshop_tasks.json --max-tasks 500 --shuffle

The output format matches sample_tasks.json:
    {
        "task_id": "ws_B09QKP7XQL_0",
        "instruction": "i'm looking for a blue wireless bluetooth headphones.",
        "target_attributes": {
            "attributes": ["wireless bluetooth"],
            "options": ["blue"]
        }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Default paths relative to this script's location
SCRIPT_DIR = Path(__file__).parent
WEBSHOP_DATA_DIR = SCRIPT_DIR.parent / "server" / "webshop" / "data"
DEFAULT_HUMAN_INS_PATH = WEBSHOP_DATA_DIR / "items_human_ins.json"
DEFAULT_OUTPUT_PATH = SCRIPT_DIR / "webshop_tasks.json"


def load_human_instructions(filepath: Path) -> dict[str, list[dict[str, Any]]]:
    """Load human instructions from WebShop data.

    Args:
        filepath: Path to items_human_ins.json file.

    Returns:
        Dictionary mapping ASINs to lists of instruction records.
    """
    logger.info(f"Loading human instructions from {filepath}")
    with open(filepath) as f:
        data = json.load(f)
    logger.info(f"Loaded instructions for {len(data)} products")
    return data


def convert_to_training_task(
    asin: str,
    instruction_record: dict[str, Any],
    task_index: int,
) -> dict[str, Any]:
    """Convert a WebShop instruction record to training task format.

    Args:
        asin: Product ASIN.
        instruction_record: Instruction dict from items_human_ins.json.
        task_index: Index for unique task ID.

    Returns:
        Task dictionary compatible with training pipeline.
    """
    instruction_text = instruction_record.get("instruction", "")
    instruction_attributes = instruction_record.get("instruction_attributes", [])
    instruction_options = instruction_record.get("instruction_options", [])

    return {
        "task_id": f"ws_{asin}_{task_index}",
        "instruction": instruction_text,
        "target_attributes": {
            "attributes": instruction_attributes,
            "options": instruction_options,
        },
        "asin": asin,
    }


def generate_tasks(
    human_instructions: dict[str, list[dict[str, Any]]],
    max_tasks: int | None = None,
    shuffle: bool = False,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate training tasks from human instruction data.

    Args:
        human_instructions: Dictionary mapping ASINs to instruction records.
        max_tasks: Maximum number of tasks to generate (None for all).
        shuffle: Whether to shuffle tasks before limiting.
        seed: Random seed for reproducibility when shuffling.

    Returns:
        List of training task dictionaries.
    """
    tasks: list[dict[str, Any]] = []
    task_index = 0

    for asin, instruction_records in human_instructions.items():
        for record in instruction_records:
            instruction = record.get("instruction", "")
            instruction_attrs = record.get("instruction_attributes", [])

            if not instruction or not instruction_attrs:
                continue

            task = convert_to_training_task(asin, record, task_index)
            tasks.append(task)
            task_index += 1

    logger.info(f"Generated {len(tasks)} tasks from human instructions")

    if shuffle:
        logger.info(f"Shuffling tasks with seed {seed}")
        random.seed(seed)
        random.shuffle(tasks)

    if max_tasks is not None and len(tasks) > max_tasks:
        logger.info(f"Limiting to {max_tasks} tasks")
        tasks = tasks[:max_tasks]

    return tasks


def main() -> int:
    """Main entry point for task generation."""
    parser = argparse.ArgumentParser(
        description="Generate training tasks from WebShop human instruction data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_HUMAN_INS_PATH,
        help=f"Path to items_human_ins.json (default: {DEFAULT_HUMAN_INS_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for generated tasks (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to generate (default: all)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle tasks before limiting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.error("Run setup.sh to download the WebShop data first.")
        return 1

    human_instructions = load_human_instructions(args.input)
    tasks = generate_tasks(
        human_instructions,
        max_tasks=args.max_tasks,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    if not tasks:
        logger.error("No valid tasks generated. Check input data.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(tasks, f, indent=2)

    logger.info(f"Wrote {len(tasks)} tasks to {args.output}")

    # Print summary statistics
    unique_asins = len(set(t["asin"] for t in tasks))
    logger.info(f"Summary: {len(tasks)} tasks from {unique_asins} unique products")

    return 0


if __name__ == "__main__":
    sys.exit(main())
