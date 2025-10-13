# Copyright (c) Microsoft. All rights reserved.

"""This sample spawns a set of rollout runners to collect data for SFT.

It communicates with the SFT algorithm via a store server.

Run the store server beforehand:

```bash
agl store --port 4747
```
"""

import asyncio
import multiprocessing
import time

from math_agent import GsmProblem, math_agent
from rich.console import Console

from agentlightning import configure_logger
from agentlightning.runner import AgentRunnerV2
from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient
from agentlightning.tracer import OtelTracer

console = Console()


def rollout_runner(*, store: LightningStore, worker_id: int) -> None:
    """A rollout runner.

    Args:
        store: The LightningStore instance.
    """

    # Since the server side has already used LiteLLM proxy to collect traces,
    # a simple OtelTracer to collect the rewards is enough.
    tracer = OtelTracer()

    runner = AgentRunnerV2[GsmProblem](tracer=tracer)

    console.print(f"[bold green]Runners: [/bold green] Rollout runner {worker_id} started.")

    with runner.run_context(agent=math_agent, store=store, worker_id=worker_id):
        asyncio.run(runner.iter())


def spawn_runners(*, store: LightningStore, n_runners: int) -> None:
    """Spawn a set of rollout runners.

    Args:
        store: The LightningStore instance.
    """

    runners = [
        multiprocessing.Process(target=rollout_runner, kwargs={"store": store, "worker_id": worker_id}, daemon=True)
        for worker_id in range(n_runners)
    ]
    try:
        for runner in runners:
            runner.start()

        for runner in runners:
            runner.join()
    except KeyboardInterrupt:
        console.print("[bold green]Runners:[/bold green] [red]KeyboardInterrupt[/red] received. Terminating runners...")
        for runner in runners:
            runner.terminate()
        deadline = time.time() + 10
        for runner in runners:
            if runner.is_alive():
                runner.join(timeout=max(deadline - time.time(), 0))
            for runner in runners:
                if runner.is_alive():
                    runner.kill()


if __name__ == "__main__":
    configure_logger()
    store = LightningStoreClient("http://localhost:4747")
    spawn_runners(store=store, n_runners=10)
