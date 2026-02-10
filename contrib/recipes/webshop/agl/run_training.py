# Copyright (c) Microsoft. All rights reserved.

"""Train a WebShop agent using Agent Lightning with external runners.

This script runs the training coordinator that:
1. Starts the Agent Lightning Store server
2. Enqueues rollouts for training
3. Waits for external runners (headless TypeScript or UI) to execute tasks
4. Collects traces and rewards for training

The key difference from the Spider example is that this uses n_runners=0
(external runner mode) because the agent is implemented in TypeScript/Vercel AI SDK
rather than Python.

Task Loading:
    By default, this script auto-generates tasks from WebShop human instruction data
    (items_human_ins.json) if available. If not found, it falls back to sample tasks.

Usage:
    python agl/run_training.py fast    # Fast training for CI
    python agl/run_training.py qwen    # Full Qwen training (auto-generates tasks)

    # Limit auto-generated tasks:
    python agl/run_training.py qwen --max-tasks 100

    # Disable auto-generation:
    python agl/run_training.py qwen --no-auto-generate

    # With custom tasks file:
    python agl/run_training.py qwen --tasks-file data/tasks.json

Prerequisites:
    1. Start the Agent Lightning Store server (will be started automatically)
    2. Start one or more headless runners:
       npm run headless -- --worker-id runner-1
    3. Or use the interactive UI which will also report to the Store
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import config_fast, config_qwen
from generate_tasks import generate_tasks, load_human_instructions
from tasks import load_sample_tasks, load_tasks_from_file

import agentlightning as agl
from agentlightning import LitAgent
from agentlightning.adapter.triplet import TracerTraceToTriplet
from agentlightning.types import NamedResources, Rollout, RolloutRawResult

# Paths where WebShop human instruction data may be found
WEBSHOP_DATA_PATHS = [
    Path(__file__).parent.parent / "server" / "webshop" / "data" / "items_human_ins.json",
    Path("/app/webshop/data/items_human_ins.json"),  # Docker path
]

# Vercel AI SDK produces spans with names like "ai.generateText", "ai.streamText", etc.
# The default adapter matches "openai.chat.completion" which doesn't match these.
# This pattern matches the AI SDK span names for LLM calls.
VERCEL_AI_SDK_LLM_CALL_PATTERN = r"ai\.(generateText|streamText|generateObject)(\.do(Generate|Stream))?"


def find_human_instructions_path() -> Path | None:
    """Find the items_human_ins.json file in known locations.

    Returns:
        Path to the file if found, None otherwise.
    """
    for path in WEBSHOP_DATA_PATHS:
        if path.exists():
            return path
    return None


def auto_generate_tasks(
    max_tasks: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
) -> list[Dict[str, Any]] | None:
    """Auto-generate tasks from WebShop human instruction data if available.

    Args:
        max_tasks: Maximum number of tasks to generate (None for all).
        shuffle: Whether to shuffle tasks before limiting.
        seed: Random seed for reproducibility when shuffling.

    Returns:
        List of generated tasks, or None if human instruction data not found.
    """
    human_ins_path = find_human_instructions_path()
    if human_ins_path is None:
        return None

    logger.info(f"[PROGRESS] Auto-generating tasks from {human_ins_path}")
    human_instructions = load_human_instructions(human_ins_path)
    tasks = generate_tasks(human_instructions, max_tasks=max_tasks, shuffle=shuffle, seed=seed)
    logger.info(f"[PROGRESS] Generated {len(tasks)} tasks")
    return tasks


class ExternalRunnerAgent(LitAgent[Dict[str, Any]]):
    """Placeholder agent that satisfies the Trainer API for external runner mode.

    In external runner mode (n_runners=0), the actual agent logic executes outside
    Python - typically in TypeScript/Node.js runners using the Vercel AI SDK.
    The Trainer API requires an agent object, so this placeholder fills that role
    while the Store server coordinates with external runners.

    This class should never have its `rollout` method invoked. If it is called,
    it indicates a configuration error (likely n_runners > 0 when it should be 0).
    """

    def rollout(self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Raise an error - external runners handle actual execution.

        Raises:
            RuntimeError: Always, since this method should never be called.
        """
        raise RuntimeError(
            "ExternalRunnerAgent.rollout() was called, but this agent is meant "
            "for external runner mode (n_runners=0). Ensure your Trainer is "
            "configured with n_runners=0 for external runner workflows."
        )


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "DEBUG").upper(), logging.DEBUG),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def train(
    config: Dict[str, Any],
    tasks: List[Dict[str, Any]],
    val_tasks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Run training with external runner mode.

    Args:
        config: Training configuration dictionary.
        tasks: List of training tasks.
        val_tasks: Optional list of validation tasks. Defaults to first 10 training tasks.
    """
    algorithm = agl.VERL(config)

    # Configure adapter to match span names from telemetry.
    # IMPORTANT: Pass adapter to Trainer, not algorithm.set_adapter() - the Trainer would overwrite it.
    # Match both LiteLLM spans (openai.chat.completion) and Vercel AI SDK spans.
    adapter = TracerTraceToTriplet(llm_call_match=r"(openai\.chat\.completion|" + VERCEL_AI_SDK_LLM_CALL_PATTERN + r")")

    # n_runners=0 means external runners will execute rollouts.
    # The Store server will be started and accept connections from:
    # - The headless TypeScript runner (scripts/headless-runner.ts)
    # - The interactive Next.js UI
    # server_host="0.0.0.0" binds to all interfaces so external runners can connect
    trainer = agl.Trainer(
        n_runners=0,  # External runner mode
        algorithm=algorithm,
        adapter=adapter,
        strategy={"type": "cs", "main_process": "algorithm", "server_host": "0.0.0.0"},
    )

    # Placeholder agent - actual execution happens in external TypeScript runners
    agent = ExternalRunnerAgent()

    logger.info("[PROGRESS] Starting training coordinator in external runner mode...")
    logger.info("[PROGRESS] Store server will be available at the configured endpoint")
    logger.info(f"[PROGRESS] Enqueuing {len(tasks)} tasks for training")

    if val_tasks is None:
        val_tasks = tasks[:10]
        logger.warning(
            "No validation tasks provided; defaulting to first %d training tasks. "
            "Consider providing --val-tasks-file for proper validation.",
            len(val_tasks),
        )
    logger.info(f"[PROGRESS] Using {len(val_tasks)} tasks for validation")

    # The trainer.fit() method will:
    # 1. Start the Store server
    # 2. Enqueue rollouts
    # 3. Wait for external runners to complete them
    # 4. Collect traces and update model
    trainer.fit(
        agent=agent,  # Placeholder agent for external runner mode
        train_dataset=tasks,
        val_dataset=val_tasks,
    )


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WebShop agent with Agent Lightning (external runner mode)")

    parser.add_argument(
        "config",
        choices=["fast", "qwen"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (full training)",
    )

    parser.add_argument(
        "--tasks-file",
        type=str,
        help="Path to tasks file (JSON or Parquet). Defaults to sample tasks.",
    )

    parser.add_argument(
        "--val-tasks-file",
        type=str,
        help="Path to validation tasks file. Defaults to first 10 training tasks.",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to use (applies to auto-generated tasks).",
    )

    parser.add_argument(
        "--no-auto-generate",
        action="store_true",
        help="Disable auto-generation of tasks; use sample tasks if no --tasks-file provided.",
    )

    args = parser.parse_args()

    # Load tasks
    if args.tasks_file:
        tasks = load_tasks_from_file(Path(args.tasks_file))
        logger.info(f"[PROGRESS] Loaded {len(tasks)} tasks from {args.tasks_file}")
    elif args.no_auto_generate:
        tasks = load_sample_tasks()
        logger.info(f"[PROGRESS] Using {len(tasks)} sample tasks (auto-generation disabled)")
    else:
        # Try auto-generation first, fall back to sample tasks
        tasks = auto_generate_tasks(max_tasks=args.max_tasks, shuffle=True)
        if tasks is None:
            tasks = load_sample_tasks()
            logger.info(f"[PROGRESS] Using {len(tasks)} sample tasks (WebShop data not found)")
        else:
            logger.info(f"[PROGRESS] Auto-generated {len(tasks)} tasks from WebShop data")

    # Get config
    config_functions = {
        "fast": config_fast,
        "qwen": config_qwen,
    }
    config = config_functions[args.config]()

    # Load validation tasks
    val_tasks = None
    if args.val_tasks_file:
        val_tasks = load_tasks_from_file(Path(args.val_tasks_file))
        logger.info(f"[PROGRESS] Loaded {len(val_tasks)} validation tasks from {args.val_tasks_file}")

    logger.info(f"[PROGRESS] Starting training with '{args.config}' configuration")
    train(config, tasks, val_tasks)


if __name__ == "__main__":
    main()
