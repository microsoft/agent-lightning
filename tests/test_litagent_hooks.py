import pytest
from contextlib import contextmanager


from agentlightning import LitAgent, Task, ResourcesUpdate
from agentlightning.runner import AgentRunner
from agentlightning.tracer import BaseTracer, TripletExporter


class DummyTracer(BaseTracer):
    @contextmanager
    def trace_context(self, name=None):
        yield

    def get_last_trace(self):
        return []


class DummyClient:
    def __init__(self):
        self.posted = None
        self.polled = False

    def poll_next_task(self):
        if self.polled:
            return None
        self.polled = True
        return Task(rollout_id="1", input={}, mode="train", resources_id=None)

    def get_latest_resources(self):
        return ResourcesUpdate(resources_id="r", resources={})

    def post_rollout(self, rollout):
        self.posted = rollout


class DummyAsyncClient:
    def __init__(self):
        self.posted = None
        self.polled = False

    async def poll_next_task_async(self):
        if self.polled:
            return None
        self.polled = True
        return Task(rollout_id="1", input={}, mode="train", resources_id=None)

    async def get_latest_resources_async(self):
        return ResourcesUpdate(resources_id="r", resources={})

    async def post_rollout_async(self, rollout):
        self.posted = rollout


class HookAgent(LitAgent):
    def __init__(self):
        super().__init__()
        self.start_called = False
        self.end_called = False
        self.end_rollout = None

    def training_rollout(self, task, rollout_id, resources):
        return 0.5

    async def training_rollout_async(self, task, rollout_id, resources):
        return 0.5

    def on_rollout_start(self, task, runner, tracer):
        self.start_called = True
        self.start_task = task

    def on_rollout_end(self, task, rollout, runner, tracer):
        self.end_called = True
        self.end_rollout = rollout


def test_runner_calls_hooks():
    agent = HookAgent()
    client = DummyClient()
    tracer = DummyTracer()
    runner = AgentRunner(agent, client, tracer, TripletExporter())

    assert runner.run() is True
    assert agent.start_called
    assert agent.end_called
    assert agent.end_rollout.final_reward == 0.5


@pytest.mark.asyncio
async def test_runner_calls_hooks_async():
    agent = HookAgent()
    client = DummyAsyncClient()
    tracer = DummyTracer()
    runner = AgentRunner(agent, client, tracer, TripletExporter())

    assert await runner.run_async() is True
    assert agent.start_called
    assert agent.end_called
    assert agent.end_rollout.final_reward == 0.5
