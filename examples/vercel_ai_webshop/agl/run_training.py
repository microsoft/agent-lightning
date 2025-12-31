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

Usage:
    python agl/run_training.py fast    # Fast training for CI
    python agl/run_training.py qwen    # Full Qwen training
    python agl/run_training.py --dev   # Dev mode for CPU-only prototyping

    # With custom tasks file:
    python agl/run_training.py qwen --tasks-file data/tasks.json

Prerequisites:
    1. Start the Agent Lightning Store server (will be started automatically)
    2. Start one or more headless runners:
       npm run headless -- --worker-id runner-1
    3. Or use the interactive UI which will also report to the Store

Dev Mode:
    The --dev flag runs a lightweight dry-run using the Baseline algorithm
    instead of VERL. This is ideal for CPU-only prototyping and debugging
    the agent workflow without GPU resources. External runners still handle
    inference via OPENAI_API_BASE.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlightning as agl
from agentlightning import LitAgent

from config import config_fast, config_qwen
from tasks import load_sample_tasks, load_tasks_from_file


class ExternalRunnerAgent(LitAgent[Dict[str, Any]]):
    """Placeholder agent for external runner mode.

    When using external runners (e.g., TypeScript/Node.js runners), the actual
    agent logic runs outside of Python. This placeholder satisfies the Trainer
    API requirement for an agent object while the real work happens externally.
    """

    def rollout(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """No-op rollout - external runners handle the actual execution."""
        # This should never be called when n_runners=0
        raise RuntimeError(
            "ExternalRunnerAgent.rollout() was called, but this agent is meant "
            "for external runner mode (n_runners=0). Check your configuration."
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train(config: Dict[str, Any], tasks: List[Dict[str, Any]], val_tasks: Optional[List[Dict[str, Any]]] = None) -> None:
    """Run training with external runner mode.

    Args:
        config: Training configuration dictionary.
        tasks: List of training tasks.
        val_tasks: Optional list of validation tasks. Defaults to first 10 training tasks.
    """
    algorithm = agl.VERL(config)

    # n_runners=0 means external runners will execute rollouts.
    # The Store server will be started and accept connections from:
    # - The headless TypeScript runner (scripts/headless-runner.ts)
    # - The interactive Next.js UI
    # server_host="0.0.0.0" binds to all interfaces so external runners can connect
    trainer = agl.Trainer(
        n_runners=0,  # External runner mode
        algorithm=algorithm,
        strategy={"type": "cs", "main_process": "algorithm", "server_host": "0.0.0.0"},
    )

    # Placeholder agent - actual execution happens in external TypeScript runners
    agent = ExternalRunnerAgent()

    logger.info("[PROGRESS] Starting training coordinator in external runner mode...")
    logger.info("[PROGRESS] Store server will be available at the configured endpoint")
    logger.info(f"[PROGRESS] Enqueuing {len(tasks)} tasks for training")

    if val_tasks is None:
        val_tasks = tasks[:10]
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


def dev(tasks: List[Dict[str, Any]], max_tasks: int = 5) -> None:
    """Run dev mode for CPU-only prototyping and debugging.

    This uses the Baseline algorithm instead of VERL, making it suitable
    for local development without GPU resources. External runners still
    execute the actual agent logic via OPENAI_API_BASE.

    Args:
        tasks: List of tasks to run.
        max_tasks: Maximum number of tasks to run in dev mode.
    """
    # Limit tasks for quick iteration
    dev_tasks = tasks[:max_tasks]

    # Dev mode uses n_runners=0 for external runners.
    # Using client_server strategy (alias: "cs") to expose HTTP endpoints for external runners.
    # server_host="0.0.0.0" binds to all interfaces so external runners can connect
    trainer = agl.Trainer(
        n_runners=0,  # External runner mode
        strategy={"type": "cs", "main_process": "algorithm", "server_host": "0.0.0.0"},
    )

    # Placeholder agent - actual execution happens in external TypeScript runners
    agent = ExternalRunnerAgent()

    logger.info("[PROGRESS] " + "=" * 50)
    logger.info("[PROGRESS] DEV MODE - CPU-only prototyping with Baseline algorithm")
    logger.info("[PROGRESS] " + "=" * 50)
    logger.info(f"[PROGRESS] Running {len(dev_tasks)} tasks (max_tasks={max_tasks})")
    logger.info("[PROGRESS] External runners should connect to execute tasks")
    logger.info("[PROGRESS] Set OPENAI_API_BASE for the model endpoint")
    logger.info("[PROGRESS] " + "=" * 50)

    # trainer.dev() runs a lightweight loop with Baseline algorithm
    trainer.dev(
        agent=agent,  # Placeholder agent for external runner mode
        train_dataset=dev_tasks,
    )


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train WebShop agent with Agent Lightning (external runner mode)"
    )

    parser.add_argument(
        "config",
        nargs="?",
        choices=["fast", "qwen"],
        help="Training configuration: 'fast' (CI testing), 'qwen' (full training)",
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in dev mode for CPU-only prototyping (uses Baseline algorithm, no GPU required)",
    )

    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5,
        help="Maximum number of tasks to run in dev mode (default: 5)",
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

    args = parser.parse_args()

    # Validate arguments
    if not args.dev and not args.config:
        parser.error("Either --dev or a config (fast/qwen) is required")

    # Load tasks
    if args.tasks_file:
        tasks = load_tasks_from_file(Path(args.tasks_file))
        logger.info(f"[PROGRESS] Loaded {len(tasks)} tasks from {args.tasks_file}")
    else:
        tasks = load_sample_tasks()
        logger.info(f"[PROGRESS] Using {len(tasks)} sample tasks")

    # Dev mode: lightweight dry-run with Baseline algorithm
    if args.dev:
        dev(tasks, max_tasks=args.max_tasks)
        return

    # Training mode: requires config
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
