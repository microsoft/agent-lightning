# Copyright (c) Microsoft. All rights reserved.

from contextlib import contextmanager
from typing import Any, Iterator, List, Optional, cast

import pytest

from agentlightning import LitAgent, ResourcesUpdate, Task
from agentlightning.adapter import TraceTripletAdapter
from agentlightning.client import AgentLightningClient
from agentlightning.runner import AgentRunner
from agentlightning.tracer import BaseTracer
from agentlightning.types import Rollout


class DummyTracer(BaseTracer):
    @contextmanager
    def trace_context(self, name: Optional[str] = None) -> Iterator[None]:
        yield

    def get_last_trace(self) -> List[Any]:
        return []


class DummyClient:
    def __init__(self) -> None:
        self.posted = None
        self.polled = False

    def poll_next_task(self) -> Optional[Task]:
        if self.polled:
            return None
        self.polled = True
        return Task(rollout_id="1", input={}, mode="train", resources_id=None)

    def get_latest_resources(self) -> ResourcesUpdate:
        return ResourcesUpdate(resources_id="r", resources={})

    def post_rollout(self, rollout: Any) -> None:
        self.posted = rollout


class DummyAsyncClient:
    def __init__(self) -> None:
        self.posted = None
        self.polled = False

    async def poll_next_task_async(self) -> Optional[Task]:
        if self.polled:
            return None
        self.polled = True
        return Task(rollout_id="1", input={}, mode="train", resources_id=None)

    async def get_latest_resources_async(self) -> ResourcesUpdate:
        return ResourcesUpdate(resources_id="r", resources={})

    async def post_rollout_async(self, rollout: Any) -> None:
        self.posted = rollout


class HookAgent(LitAgent[Any]):
    def __init__(self):
        super().__init__()
        self.start_called = False
        self.end_called = False
        self.end_rollout: Rollout | None = None

    def training_rollout(self, task: Any, resources: Any, rollout: Any) -> float:
        return 0.5

    async def training_rollout_async(self, task: Any, resources: Any, rollout: Any) -> float:
        return 0.5

    def on_rollout_start(self, task: Any, runner: Any, tracer: Any) -> None:
        self.start_called = True
        self.start_task = task

    def on_rollout_end(self, task: Any, rollout: Any, runner: Any, tracer: Any) -> None:
        self.end_called = True
        self.end_rollout = rollout


def test_runner_calls_hooks():
    agent = HookAgent()
    client = DummyClient()
    tracer = DummyTracer()
    runner = AgentRunner(agent, cast(AgentLightningClient, client), tracer, TraceTripletAdapter())

    assert runner.run() is True
    assert agent.start_called
    assert agent.end_called
    assert agent.end_rollout is not None
    assert agent.end_rollout.final_reward == 0.5


@pytest.mark.asyncio
async def test_runner_calls_hooks_async():
    agent = HookAgent()
    client = DummyAsyncClient()
    tracer = DummyTracer()
    runner = AgentRunner(agent, cast(AgentLightningClient, client), tracer, TraceTripletAdapter())

    assert await runner.run_async() is True
    assert agent.start_called
    assert agent.end_called
    assert agent.end_rollout is not None
    assert agent.end_rollout.final_reward == 0.5
