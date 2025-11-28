# Copyright (c) Microsoft. All rights reserved.

"""Minimal Agent-lightning + Tinker training example.

The Hello agent fine-tunes a model so it repeats whatever identity string you
pass in (e.g., `"Say you are 42" -> "I'm 42."`). It mirrors the structure of
Tinker Cookbook RL recipes but drives rollouts through Agent-lightning tasks
instead of Tinker's built-in environments.

Environment setup:

1. Copy `examples/tinker/.env.example` to `examples/tinker/.env`.
2. Fill in `OPENAI_API_KEY` / `OPENAI_BASE_URL` so the helper completions
   can be routed via LiteLLM.
3. Provide `TINKER_API_KEY` if you plan to train against the hosted Tinker service.

This example does not support W&B logging.

CLI entry points:

```bash
# Integrated run that spawns store, algorithm, and runners
python hello.py oneclick
```

Distributed workflow across three terminals:

```bash
agl store  # <-- expect the store to be running on port 4747
python hello.py algo
python hello.py runner
```
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing
import socket

from agl_tinker.algo import Tinker
from agl_tinker.env import AGLDatasetBuilder
from agl_tinker.train import Config
from agl_tinker.train import main as entrypoint
from openai import OpenAI
from rich.console import Console

import agentlightning as agl

console = Console()


def _find_available_port() -> int:
    """Find an available port by binding to port 0.

    Returns:
        An available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@agl.rollout
def hello(task: str, llm: agl.LLM, rollout: agl.Rollout) -> None:
    """Agent rollout function that tests if the model claims the given identity.

    Prompts the model to say it is the given task/identity and assigns a reward
    based on whether the model's response matches the expected behavior.

    Args:
        task: The identity string the model should claim to be.
        llm: The LLM endpoint configuration.
        rollout: The rollout metadata containing rollout ID and mode.
    """
    openai_client = OpenAI(base_url=llm.endpoint, api_key="dummy")
    response = openai_client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": f"Let's play a game. Say you are {task}."}],
    )

    response_content = response.choices[0].message.content
    content_lower = response_content.lower() if response_content else ""
    if ("i am " + task) in content_lower or ("i'm " + task) in content_lower:
        rew = 1.0
    elif ("not " + task) in content_lower:
        rew = -1.0
    else:
        rew = 0.0

    console.print(
        f"[bold green]Runners ({rollout.rollout_id}, {rollout.mode}):[/bold green] "
        f"{task} -> {response_content} -> Reward: {rew}"
    )
    agl.emit_reward(rew)


def run_algo():
    """Run the training algorithm in standalone mode.

    Launches the Tinker training algorithm that connects to a separate store
    and rollout runners.
    """
    config = Config(
        learning_rate=1e-5,
        dataset_builder=AGLDatasetBuilder(
            train_dataset=[str(i) for i in range(1000)],
            val_dataset=[str(i) for i in range(1000, 1024)],
            batch_size=32,
            shuffle=True,
            group_size=4,
            seed=42,
        ),
        renderer_name="qwen3_instruct",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        log_path="logs/hello",
        max_tokens=32,
        store_address="http://localhost:4747",
    )
    asyncio.run(entrypoint(config))


def run_rollout(*, worker_id: int) -> None:
    """Rollout runner, single-process."""
    tracer = agl.AgentOpsTracer()

    runner = agl.LitAgentRunner[str](tracer=tracer)

    console.print(f"[bold green]Runners:[/bold green] Rollout runner {worker_id} started.")

    store = agl.LightningStoreClient("http://localhost:4747")
    with runner.run_context(agent=hello, store=store, worker_id=worker_id):
        asyncio.run(runner.iter())


def spawn_runners(*, n_runners: int) -> None:
    """Spawn a set of rollout runners in separate processes.

    Args:
        n_runners: The number of runners to spawn.
    """

    runners = [
        multiprocessing.Process(target=run_rollout, kwargs={"worker_id": worker_id}) for worker_id in range(n_runners)
    ]
    for runner in runners:
        runner.start()

    for runner in runners:
        runner.join()


def oneclick(ci: bool = False):
    """Run integrated training with algorithm and runners in one process.

    This is the simplest way to run the example, as it handles spawning
    the store, algorithm, and runners automatically.

    Args:
        ci: Whether to run in CI mode. Fast verification.
    """
    if ci:
        # Use smaller batch size and group size for faster verification.
        batch_size = 4
        group_size = 2
    else:
        batch_size = 16
        group_size = 4

    config = Config(
        learning_rate=1e-5,
        dataset_builder=AGLDatasetBuilder(
            batch_size=batch_size,
            group_size=group_size,
            seed=42,
            n_epochs=1,
        ),
        renderer_name="qwen3_instruct",
        model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
        log_path="logs/hello",
        max_tokens=32,
        llm_proxy_port=_find_available_port(),
    )
    trainer = agl.Trainer(
        algorithm=Tinker(config),
        llm_proxy=agl.LLMProxy(
            port=12306,
            num_retries=3,
            # Must use thread mode here because otherwise the Tinker sampling client will hang.
            launch_mode="thread",
        ),
        n_runners=8,
        port=_find_available_port(),
    )

    if ci:
        # For faster verification, use a smaller dataset.
        train_dataset = [str(i) for i in range(16)]
        val_dataset = [str(i) for i in range(100, 108)]
    else:
        train_dataset = [str(i) for i in range(1000)]
        val_dataset = [str(i) for i in range(1000, 1024)]
    trainer.fit(hello, train_dataset=train_dataset, val_dataset=val_dataset)


def main():
    """Entry point for the hello example script."""
    parser = argparse.ArgumentParser(description="Train a hello echo agent with Agent-lightning + Tinker.")
    parser.add_argument("mode", type=str, choices=["algo", "runner", "oneclick"])
    parser.add_argument("--ci", action="store_true", help="Run in CI mode. Fast verification.")

    args = parser.parse_args()

    if args.ci:
        if args.mode != "oneclick":
            raise ValueError("CI mode only supports oneclick mode.")

    agl.setup_logging()
    if args.mode == "algo":
        run_algo()
    elif args.mode == "runner":
        spawn_runners(n_runners=8)
    elif args.mode == "oneclick":
        oneclick(ci=args.ci)


if __name__ == "__main__":
    main()
