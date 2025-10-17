# Copyright (c) Microsoft. All rights reserved.

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

from agentlightning.litagent import LitAgent
from agentlightning.runner import LitAgentRunner
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.base import Tracer
from agentlightning.types import LLM, Hook, Rollout

from ..common.tracer import clear_tracer_provider


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    """Setup the tracer provider for the tests."""

    clear_tracer_provider()

    trace_api.set_tracer_provider(TracerProvider())


class DummyTracer(Tracer):
    """Minimal tracer for testing."""

    def __init__(self) -> None:
        super().__init__()
        self._last_trace: List[ReadableSpan] = []
        self.init_called = False
        self.init_worker_called = False
        self.teardown_called = False
        self.teardown_worker_called = False

    def init(self, *args: Any, **kwargs: Any) -> None:
        self.init_called = True
        self._last_trace.clear()

    def init_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        self.init_worker_called = True

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        self.teardown_called = True
        self._last_trace.clear()

    def teardown_worker(self, worker_id: int, *args: Any, **kwargs: Any) -> None:
        self.teardown_worker_called = True

    def get_last_trace(self) -> List[ReadableSpan]:
        return list(self._last_trace)

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncGenerator[List[ReadableSpan], None]:
        self._last_trace = []
        try:
            yield self._last_trace
        finally:
            pass


class DummyAgent(LitAgent[Dict[str, Any]]):
    """Minimal agent for testing."""

    def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> None:
        return None


class RecordingHook(Hook):
    """Hook that records lifecycle events."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: List[str] = []

    async def on_rollout_start(self, *, agent: LitAgent[Any], runner: Any, rollout: Rollout) -> None:
        self.calls.append("on_rollout_start")

    async def on_trace_start(self, *, agent: LitAgent[Any], runner: Any, tracer: Tracer, rollout: Rollout) -> None:
        self.calls.append("on_trace_start")

    async def on_trace_end(self, *, agent: LitAgent[Any], runner: Any, tracer: Tracer, rollout: Rollout) -> None:
        self.calls.append("on_trace_end")

    async def on_rollout_end(self, *, agent: LitAgent[Any], runner: Any, rollout: Rollout, spans: Any) -> None:
        self.calls.append("on_rollout_end")


@pytest.mark.asyncio
async def test_run_context_basic_lifecycle() -> None:
    """Test that run_context properly initializes and tears down the runner."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    with runner.run_context(agent=agent, store=store):
        # Verify initialization happened
        assert tracer.init_called
        assert tracer.init_worker_called
        assert runner.worker_id == 0
        assert runner.get_agent() is agent
        assert runner.get_store() is store

    # Verify teardown happened
    assert tracer.teardown_worker_called
    assert tracer.teardown_called
    assert runner.worker_id is None


@pytest.mark.asyncio
async def test_run_context_yields_runner() -> None:
    """Test that run_context yields the runner instance itself."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    with runner.run_context(agent=agent, store=store) as yielded_runner:
        assert yielded_runner is runner


@pytest.mark.asyncio
async def test_run_context_with_hooks() -> None:
    """Test that run_context properly passes hooks to init()."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)
    hook = RecordingHook()

    with runner.run_context(agent=agent, store=store, hooks=[hook]):
        # Verify hooks were registered
        assert runner._hooks == [hook]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_run_context_teardown_on_exception_in_context() -> None:
    """Test that run_context properly tears down even when exception occurs in with block."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    with pytest.raises(RuntimeError, match="test error"):
        with runner.run_context(agent=agent, store=store):
            raise RuntimeError("test error")

    # Verify teardown still happened
    assert tracer.teardown_worker_called
    assert tracer.teardown_called


@pytest.mark.asyncio
async def test_run_context_no_teardown_worker_if_init_worker_fails() -> None:
    """Test that teardown_worker is not called if init_worker fails."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    # Mock init_worker to raise an exception
    original_init_worker = runner.init_worker

    def failing_init_worker(worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        raise RuntimeError("init_worker failed")

    runner.init_worker = failing_init_worker  # type: ignore[method-assign]

    try:
        with pytest.raises(RuntimeError, match="init_worker failed"):
            with runner.run_context(agent=agent, store=store):
                pass
    finally:
        # Restore original method
        runner.init_worker = original_init_worker  # type: ignore[method-assign]

    # Verify teardown was called but teardown_worker was not
    assert tracer.init_called
    assert not tracer.init_worker_called
    assert not tracer.teardown_worker_called
    assert tracer.teardown_called


@pytest.mark.asyncio
async def test_run_context_no_teardown_if_init_fails() -> None:
    """Test that teardown is not called if init fails."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    # Mock init to raise an exception
    original_init = runner.init

    def failing_init(agent: LitAgent[Any], **kwargs: Any) -> None:
        raise RuntimeError("init failed")

    runner.init = failing_init  # type: ignore[method-assign]

    try:
        with pytest.raises(RuntimeError, match="init failed"):
            with runner.run_context(agent=agent, store=store):
                pass
    finally:
        # Restore original method
        runner.init = original_init  # type: ignore[method-assign]

    # Verify neither init_worker nor teardown methods were called
    assert not tracer.init_called
    assert not tracer.init_worker_called
    assert not tracer.teardown_worker_called
    assert not tracer.teardown_called


@pytest.mark.asyncio
async def test_run_context_handles_teardown_worker_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Test that exceptions in teardown_worker are caught and logged."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    # Mock teardown_worker to raise an exception
    original_teardown_worker = runner.teardown_worker

    def failing_teardown_worker(worker_id: int, *args: Any, **kwargs: Any) -> None:
        original_teardown_worker(worker_id, *args, **kwargs)
        raise RuntimeError("teardown_worker failed")

    runner.teardown_worker = failing_teardown_worker  # type: ignore[method-assign]

    try:
        with runner.run_context(agent=agent, store=store):
            pass
    finally:
        # Restore original method
        runner.teardown_worker = original_teardown_worker  # type: ignore[method-assign]

    # Verify both teardown methods were attempted
    assert tracer.teardown_worker_called
    assert tracer.teardown_called

    # Verify error was logged
    assert any("Error during runner worker teardown" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_run_context_handles_teardown_exception(caplog: pytest.LogCaptureFixture) -> None:
    """Test that exceptions in teardown are caught and logged."""
    tracer = DummyTracer()
    agent = DummyAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[Dict[str, Any]](tracer=tracer)

    # Mock teardown to raise an exception
    original_teardown = runner.teardown

    def failing_teardown(*args: Any, **kwargs: Any) -> None:
        original_teardown(*args, **kwargs)
        raise RuntimeError("teardown failed")

    runner.teardown = failing_teardown  # type: ignore[method-assign]

    try:
        with runner.run_context(agent=agent, store=store):
            pass
    finally:
        # Restore original method
        runner.teardown = original_teardown  # type: ignore[method-assign]

    # Verify both teardown methods were attempted
    assert tracer.teardown_worker_called
    assert tracer.teardown_called

    # Verify error was logged
    assert any("Error during runner teardown" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_run_context_can_be_used_for_step() -> None:
    """Test that run_context can be used to execute runner.step()."""

    class CountingAgent(LitAgent[str]):
        def __init__(self) -> None:
            super().__init__()
            self.call_count = 0

        def validation_rollout(self, task: str, resources: Dict[str, Any], rollout: Any) -> float:
            self.call_count += 1
            return 0.5

    tracer = DummyTracer()
    agent = CountingAgent()
    store = InMemoryLightningStore()
    runner = LitAgentRunner[str](tracer=tracer)

    await store.update_resources("default", {"llm": LLM(endpoint="http://localhost", model="dummy")})

    with runner.run_context(agent=agent, store=store):
        await runner.step("test task")

    # Verify the agent was called
    assert agent.call_count == 1

    # Verify the rollout was recorded in the store
    rollouts = await store.query_rollouts()
    assert len(rollouts) == 1
    assert rollouts[0].status == "succeeded"
