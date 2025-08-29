import pytest

from agentlightning.litagent import LitAgent


def test_rollout_batch_defaults_to_single_rollout():
    class DummyAgent(LitAgent):
        def __init__(self):
            super().__init__()
            self.calls = []

        def training_rollout(self, task, rollout_id, resources):
            self.calls.append((task, rollout_id))
            return 1.0

    agent = DummyAgent()
    results = agent.training_rollout_batch([1, 2], ["r1", "r2"], [{}, {}])
    assert results == [1.0, 1.0]
    assert agent.calls == [(1, "r1"), (2, "r2")]

    val_results = agent.validation_rollout_batch([3], ["r3"], [{}])
    assert val_results == [1.0]
    assert agent.calls == [(1, "r1"), (2, "r2"), (3, "r3")]


@pytest.mark.asyncio
async def test_rollout_batch_async_defaults_to_single_rollout():
    class DummyAsyncAgent(LitAgent):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def training_rollout_async(self, task, rollout_id, resources):
            self.calls.append((task, rollout_id))
            return 1.0

    agent = DummyAsyncAgent()
    results = await agent.training_rollout_batch_async([1, 2], ["r1", "r2"], [{}, {}])
    assert results == [1.0, 1.0]

    val_results = await agent.validation_rollout_batch_async([3], ["r3"], [{}])
    assert val_results == [1.0]
    assert agent.calls == [(1, "r1"), (2, "r2"), (3, "r3")]
