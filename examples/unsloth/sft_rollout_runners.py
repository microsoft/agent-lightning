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

from math_agent import GsmProblem, math_agent
from rich.console import Console

from agentlightning import setup_logging
from agentlightning.runner import LitAgentRunner
from agentlightning.store import LightningStore, LightningStoreClient
from agentlightning.tracer import OtelTracer

console = Console()


def run_rollout(*, store: LightningStore, worker_id: int) -> None:
    """A rollout runner.

    Args:
        store: The LightningStore instance.
    """

    # Since the server side has already used LiteLLM proxy to collect traces,
    # a simple OtelTracer to collect the rewards is enough.
    tracer = OtelTracer()

    runner = LitAgentRunner[GsmProblem](tracer=tracer)

    console.print(f"[bold green]Runners: [/bold green] Rollout runner {worker_id} started.")

    with runner.run_context(agent=math_agent, store=store, worker_id=worker_id):
        asyncio.run(runner.iter())


def spawn_runners(*, store: LightningStore, n_runners: int) -> None:
    """Spawn a set of rollout runners.

    It's just replicating the `run_rollout` function in multiple processes.
    You can also replace this function with a bash script.

    Args:
        store: The LightningStore instance.
        n_runners: The number of runners to spawn.
    """

    runners = [
        multiprocessing.Process(target=run_rollout, kwargs={"store": store, "worker_id": worker_id})
        for worker_id in range(n_runners)
    ]
    for runner in runners:
        runner.start()

    for runner in runners:
        runner.join()


if __name__ == "__main__":
    setup_logging()
    store = LightningStoreClient("http://localhost:4747")
    spawn_runners(store=store, n_runners=4)
