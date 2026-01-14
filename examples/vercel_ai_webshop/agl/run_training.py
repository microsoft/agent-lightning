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
from typing import Any, Dict, List, Literal, Optional

import agentlightning as agl
from agentlightning import LitAgent
from agentlightning.adapter.triplet import TracerTraceToTriplet
from agentlightning.verl.daemon import AgentModeDaemon

from config import config_fast, config_qwen
from tasks import load_sample_tasks, load_tasks_from_file

# Vercel AI SDK produces spans with names like "ai.generateText", "ai.streamText", etc.
# The default adapter matches "openai.chat.completion" which doesn't match these.
# This pattern matches the AI SDK span names for LLM calls.
VERCEL_AI_SDK_LLM_CALL_PATTERN = r"ai\.(generateText|streamText|generateObject)(\.do(Generate|Stream))?"


class WebShopDaemon(AgentModeDaemon):
    """Custom daemon that bypasses LLMProxy for external runner mode.

    The LLMProxy cannot be started reliably inside Ray actors due to event loop
    conflicts with LiteLLM. This daemon overrides `_async_set_up` to:
    1. Skip the LLMProxy startup entirely
    2. Register the vLLM server directly as an LLM resource

    External runners will connect directly to the vLLM HTTP server instead of
    going through the LLMProxy.

    Known Limitations (compared to using LLMProxy):
    - Token IDs are not captured (LLMProxy adds `return_token_ids=True` to requests)
    - LLM spans may not have rollout/attempt attribution from the proxy
    - Training may be less accurate due to retokenization issues

    The TypeScript runner still captures traces via Vercel AI SDK's experimental_telemetry,
    so basic tracing works. For production use with full token ID support, consider
    starting LLMProxy as a separate service outside the Ray actor.
    """

    async def _async_set_up(
        self,
        data: Dict[str, Any],
        server_addresses: List[str],
        is_train: bool = True,
    ) -> None:
        """Set up data and resources, bypassing LLMProxy.

        This override skips the LLMProxy startup and registers the vLLM server
        directly as an LLM resource. External runners will use the vLLM server
        directly for inference.
        """
        import uuid

        from agentlightning.types import (
            EnqueueRolloutRequest,
            LLM,
            RolloutConfig,
        )

        # Import the helper function from parent module
        from agentlightning.verl.daemon import _to_native

        self.clear_data_and_server()

        # Update backend addresses (skip LLMProxy startup)
        if server_addresses != self.backend_llm_server_addresses:
            self.backend_llm_server_addresses = server_addresses
            # NOTE: We intentionally skip self._update_proxy_server_v1() here
            # because LLMProxy cannot start inside Ray actors due to event loop conflicts.
            logger.info(
                f"[WebShopDaemon] Skipping LLMProxy startup. vLLM servers: {server_addresses}"
            )

        self.is_train = is_train

        # Create LLM resource pointing directly to vLLM server (bypass LLMProxy)
        if self.backend_llm_server_addresses:
            # Use the first vLLM server address
            vllm_address = self.backend_llm_server_addresses[0]
            # Ensure the address has http:// prefix and /v1 suffix for OpenAI compatibility
            if not vllm_address.startswith("http"):
                vllm_address = f"http://{vllm_address}"
            if not vllm_address.endswith("/v1"):
                vllm_address = f"{vllm_address}/v1"

            llm_resource = LLM(
                endpoint=vllm_address,
                model=self.train_information.get("model", "default-model"),
                sampling_parameters={
                    "temperature": self.train_information.get("temperature", 0.7 if is_train else 0.0)
                },
            )
            logger.info(f"[WebShopDaemon] Registered vLLM resource: {vllm_address}")
        else:
            # Fallback: create a placeholder resource (external runners will use OPENAI_API_BASE)
            llm_resource = LLM(
                endpoint="http://localhost:8000/v1",  # Placeholder
                model=self.train_information.get("model", "default-model"),
                sampling_parameters={
                    "temperature": self.train_information.get("temperature", 0.7 if is_train else 0.0)
                },
            )
            logger.warning("[WebShopDaemon] No vLLM servers available, using placeholder endpoint")

        resources = {"main_llm": llm_resource}

        # Register resources in the store
        resources_update = await self.store.add_resources(resources)
        resources_id = resources_update.resources_id

        # Queue tasks for agents to process
        keys = list(data.keys())
        num_samples = len(data[keys[0]])
        rollouts_per_sample = self.train_rollout_n if is_train else 1

        enqueue_rollout_requests = []

        for i in range(num_samples):
            data_id = str(uuid.uuid4())
            original_sample = {key: data[key][i] for key in keys}
            original_sample["data_id"] = data_id
            self._task_id_to_original_sample[data_id] = original_sample

            for _ in range(rollouts_per_sample):
                task_metadata = {"data_id": data_id, "is_train": is_train}
                enqueue_rollout_requests.append(
                    EnqueueRolloutRequest(
                        input=_to_native(original_sample),
                        mode="train" if is_train else "val",
                        resources_id=resources_id,
                        config=RolloutConfig(
                            unresponsive_seconds=self.llm_timeout_seconds,
                            timeout_seconds=self.llm_timeout_seconds,
                        ),
                        metadata=task_metadata,
                    )
                )

        # Enqueue all tasks in a single batch
        rollouts = await self.store.enqueue_many_rollouts(enqueue_rollout_requests)
        self._task_id_to_original_sample.update(
            {
                rollout.rollout_id: self._task_id_to_original_sample[rollout.metadata["data_id"]]
                for rollout in rollouts
            }
        )
        self._total_tasks_queued += len(rollouts)

        logger.info(f"[WebShopDaemon] Enqueued {len(rollouts)} rollouts")


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
    # Use custom daemon class that bypasses LLMProxy for external runner mode
    # The LLMProxy cannot be started reliably inside Ray actors due to event loop conflicts
    algorithm = agl.VERL(config, daemon_cls=WebShopDaemon)

    # Configure adapter to match span names from our custom emitLlmCallSpan() telemetry
    # The span name pattern matches "ai.generateText" which is what emitLlmCallSpan uses
    # IMPORTANT: Pass adapter to Trainer, not algorithm.set_adapter() - the Trainer would overwrite it
    adapter = TracerTraceToTriplet(llm_call_match=VERCEL_AI_SDK_LLM_CALL_PATTERN)

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
