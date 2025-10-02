import asyncio
import logging
from typing import Any

from agentlightning.litagent import rollout
from agentlightning.logging import configure_logger
from agentlightning.runner import AgentRunnerV2
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer import AgentOpsTracer
from agentlightning.types.core import LLM


@rollout
def _agent_func(task: Any, llm: LLM) -> float:
    print(task)
    return 0.5


async def main():
    configure_logger(level=logging.DEBUG)
    runner = AgentRunnerV2[Any](tracer=AgentOpsTracer())

    store = InMemoryLightningStore()
    await store.update_resources(
        resources_id="res-123", resources={"llm": LLM(endpoint="http://localhost:8000", model="gpt-4o")}
    )
    runner.init(agent=_agent_func)
    runner.init_worker(worker_id=0, store=store)
    await runner.step(input={"task": "test"})

    runner.teardown_worker(worker_id=0)
    runner.teardown()

    print(store.__dict__)


if __name__ == "__main__":
    asyncio.run(main())
