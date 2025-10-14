# Copyright (c) Microsoft. All rights reserved.

import asyncio
import random
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import SpanContext, TraceFlags, TraceState
from opentelemetry.trace.status import Status, StatusCode

from agentlightning.execution.events import Event, ThreadingEvent
from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward, find_final_reward
from agentlightning.runner import AgentRunnerV2
from agentlightning.runner.base import BaseRunner
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.base import BaseTracer
from agentlightning.types import LLM, Hook, NamedResources, PromptTemplate, RolloutV2, Span, SpanNames

trace_api.set_tracer_provider(TracerProvider())


def create_readable_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> ReadableSpan:
    trace_id = random.getrandbits(128)
    span_id = random.getrandbits(64)
    context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=TraceState(),
    )
    status = Status(status_code=StatusCode.UNSET)
    return ReadableSpan(
        name=name,
        context=context,
        parent=None,
        resource=Resource.create({}),
        attributes=attributes or {},
        events=(),
        links=(),
        status=status,
    )


def create_agent_span(
    rollout_id: str,
    attempt_id: str,
    sequence_id: int,
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Span:
    readable = create_readable_span(name, attributes)
    return Span.from_opentelemetry(
        readable,
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
    )


class DummyTracer(BaseTracer):
    def __init__(self) -> None:
        super().__init__()
        self._last_trace: List[ReadableSpan] = []
        self._contexts: List[Dict[str, Any]] = []

    def init(self, *args: Any, **kwargs: Any) -> None:
        self._last_trace.clear()

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        self._last_trace.clear()

    def get_last_trace(self) -> List[ReadableSpan]:
        return list(self._last_trace)

    @contextmanager
    def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Iterator[List[ReadableSpan]]:
        previous = self._contexts[-1] if self._contexts else None
        current = {
            "name": name,
            "store": store,
            "rollout_id": rollout_id,
            "attempt_id": attempt_id,
        }
        self._contexts.append(current)
        self._last_trace = []
        try:
            yield self._last_trace
        finally:
            self._contexts.pop()
            if previous is None:
                self._contexts = []

    def record_span(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> ReadableSpan:
        span = create_readable_span(name, attributes)
        self._last_trace.append(span)
        return span


async def setup_runner(
    agent: LitAgent[Any],
    *,
    tracer: Optional[DummyTracer] = None,
    max_rollouts: Optional[int] = None,
    poll_interval: float = 0.01,
    hooks: Sequence[Hook] = (),
) -> tuple[AgentRunnerV2[Any], InMemoryLightningStore, DummyTracer]:
    tracer = tracer or DummyTracer()
    store = InMemoryLightningStore()
    await store.update_resources("default", {"llm": LLM(endpoint="http://localhost", model="dummy")})

    runner = AgentRunnerV2[Any](tracer=tracer, max_rollouts=max_rollouts, poll_interval=poll_interval)
    runner.init(agent=agent, hooks=hooks)
    runner.init_worker(worker_id=0, store=store)
    return runner, store, tracer


def teardown_runner(runner: AgentRunnerV2[Any]) -> None:
    runner.teardown_worker(worker_id=0)
    runner.teardown()


async def assert_single_attempt_succeeded(store: InMemoryLightningStore) -> tuple[str, str]:
    rollouts = await store.query_rollouts()
    assert len(rollouts) == 1
    rollout = rollouts[0]
    attempts = await store.query_attempts(rollout.rollout_id)
    assert attempts[-1].status == "succeeded"
    return rollout.rollout_id, attempts[-1].attempt_id


class RecordingHook(Hook):
    def __init__(self) -> None:
        super().__init__()
        self.calls: List[str] = []
        self.received_spans: Optional[List[ReadableSpan] | List[Span]] = None

    async def on_rollout_start(self, *, agent: LitAgent[Any], runner: BaseRunner[Any], rollout: RolloutV2) -> None:
        self.calls.append("on_rollout_start")

    async def on_trace_start(
        self, *, agent: LitAgent[Any], runner: BaseRunner[Any], tracer: BaseTracer, rollout: RolloutV2
    ) -> None:
        self.calls.append("on_trace_start")

    async def on_trace_end(
        self, *, agent: LitAgent[Any], runner: BaseRunner[Any], tracer: BaseTracer, rollout: RolloutV2
    ) -> None:
        self.calls.append("on_trace_end")

    async def on_rollout_end(
        self,
        *,
        agent: LitAgent[Any],
        runner: BaseRunner[Any],
        rollout: RolloutV2,
        spans: List[ReadableSpan] | List[Span],
    ) -> None:
        self.calls.append("on_rollout_end")
        self.received_spans = spans


@pytest.mark.asyncio
async def test_step_records_spans_for_none_result() -> None:
    tracer = DummyTracer()

    class AsyncSpanAgent(LitAgent[Dict[str, Any]]):
        async def validation_rollout_async(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> None:
            span = tracer.record_span("work", {"task_id": task["task_id"]})
            store = cast(AgentRunnerV2[Dict[str, Any]], self.runner).get_store()
            await store.add_otel_span(rollout.rollout_id, rollout.attempt.attempt_id, span)  # type: ignore[attr-defined]
            return None

    agent = AsyncSpanAgent()
    runner, store, _ = await setup_runner(agent, tracer=tracer)
    try:
        await runner.step({"task_id": 1})
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    assert [span.name for span in spans] == ["work"]
    assert find_final_reward(spans) is None


@pytest.mark.asyncio
async def test_step_emits_reward_for_float_result() -> None:
    class RewardAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 0.75

    agent = RewardAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        await runner.step({"prompt": "hello"})
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    rewards = [span.attributes.get("reward") for span in spans if span.name == SpanNames.REWARD.value]
    assert rewards == [0.75]


@pytest.mark.asyncio
async def test_step_handles_non_llm_resource() -> None:
    class PromptAgent(LitAgent[str]):
        def validation_rollout(self, task: str, resources: Dict[str, Any], rollout: Any) -> float:
            template = resources["template"]
            assert isinstance(template, PromptTemplate)
            rendered = template.template.format(name=task)
            assert task in rendered
            return 0.1

    agent = PromptAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        await store.update_resources(
            "prompt-resource",
            {"template": PromptTemplate(template="Hello {name}!", engine="f-string")},
        )
        await runner.step("Ada")
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    rewards = [span.attributes.get("reward") for span in spans if span.name == SpanNames.REWARD.value]
    assert rewards == [0.1]


@pytest.mark.asyncio
async def test_step_accepts_readable_span_list() -> None:
    class ReadableSpanAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> List[ReadableSpan]:
            return [create_readable_span(f"trace-{i}", {"idx": i}) for i in range(2)]

    agent = ReadableSpanAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        await runner.step({"payload": True})
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    assert [span.name for span in spans] == ["trace-0", "trace-1"]


@pytest.mark.asyncio
async def test_step_accepts_agent_span_list() -> None:
    class AgentSpanAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> List[Span]:
            return [
                create_agent_span(rollout.rollout_id, rollout.attempt.attempt_id, 1, "custom-1", {"order": 1}),
                create_agent_span(rollout.rollout_id, rollout.attempt.attempt_id, 2, "custom-2", {"order": 2}),
            ]

    agent = AgentSpanAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        await runner.step({"payload": False})
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    assert [span.name for span in spans] == ["custom-1", "custom-2"]


@pytest.mark.asyncio
async def test_iter_respects_max_rollouts() -> None:
    class CountingAgent(LitAgent[Dict[str, Any]]):
        def __init__(self) -> None:
            super().__init__()
            self.processed: List[int] = []

        def training_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            self.processed.append(task["idx"])
            return 0.0

    agent = CountingAgent()
    runner, store, _ = await setup_runner(agent, max_rollouts=2)

    for idx in range(3):
        await store.enqueue_rollout({"idx": idx}, mode="train")

    try:
        await asyncio.wait_for(runner.iter(), timeout=1)
    finally:
        teardown_runner(runner)

    assert agent.processed == [0, 1]
    rollouts = await store.query_rollouts()
    statuses = {rollout.rollout_id: rollout.status for rollout in rollouts}
    assert list(statuses.values()).count("succeeded") == 2


@pytest.mark.asyncio
async def test_iter_stops_when_event_is_set() -> None:
    stop_event = ThreadingEvent()

    class StoppableAgent(LitAgent[Dict[str, Any]]):
        def __init__(self) -> None:
            super().__init__()
            self.processed: List[int] = []

        async def training_rollout_async(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> None:
            self.processed.append(task["idx"])
            if len(self.processed) == 1:
                stop_event.set()
            await asyncio.sleep(0.05)
            return None

    agent = StoppableAgent()
    runner, store, _ = await setup_runner(agent)

    for idx in range(3):
        await store.enqueue_rollout({"idx": idx}, mode="train")

    iter_task = asyncio.create_task(runner.iter(event=stop_event))
    try:
        await asyncio.wait_for(asyncio.to_thread(stop_event.wait, timeout=1), timeout=2)
        await asyncio.wait_for(iter_task, timeout=1)
    finally:
        teardown_runner(runner)

    assert agent.processed == [0]
    rollouts = await store.query_rollouts()
    succeeded = [rollout for rollout in rollouts if rollout.status == "succeeded"]
    assert len(succeeded) == 1


@pytest.mark.asyncio
async def test_iter_waits_when_queue_empty_calls_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    stop_event = ThreadingEvent()

    class IdleAgent(LitAgent[Any]):
        def training_rollout(self, task: Any, resources: Any, rollout: Any) -> None:
            return None

    agent = IdleAgent()
    runner, _, _ = await setup_runner(agent, poll_interval=0.01)

    sleep_calls = 0

    async def fake_sleep(event: Optional[Event] = None) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if event is not None:
            event.set()

    monkeypatch.setattr(runner, "_sleep_until_next_poll", fake_sleep)

    try:
        await runner.iter(event=stop_event)
    finally:
        teardown_runner(runner)

    assert sleep_calls >= 1


@pytest.mark.asyncio
async def test_async_validation_rollout_used() -> None:
    class AsyncValidationAgent(LitAgent[Dict[str, Any]]):
        def __init__(self) -> None:
            super().__init__()
            self.validation_calls = 0
            self.training_calls = 0

        async def validation_rollout_async(
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> float:
            self.validation_calls += 1
            return 0.0

        async def training_rollout_async(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            self.training_calls += 1
            return 0.0

    agent = AsyncValidationAgent()
    runner, store, _ = await setup_runner(agent, max_rollouts=1)
    await store.enqueue_rollout({"idx": 1}, mode="val")

    try:
        await runner.iter()
    finally:
        teardown_runner(runner)

    assert agent.validation_calls == 1
    assert agent.training_calls == 0


@pytest.mark.asyncio
async def test_training_rollout_sync_used() -> None:
    class SyncTrainingAgent(LitAgent[Dict[str, Any]]):
        def __init__(self) -> None:
            super().__init__()
            self.training_calls = 0

        def training_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> None:
            self.training_calls += 1
            return None

    agent = SyncTrainingAgent()
    runner, store, _ = await setup_runner(agent, max_rollouts=1)
    await store.enqueue_rollout({"idx": 99}, mode="train")

    try:
        await runner.iter()
    finally:
        teardown_runner(runner)

    assert agent.training_calls == 1


@pytest.mark.asyncio
async def test_step_handles_agent_exception_marks_attempt_failed() -> None:
    class FailingAgent(LitAgent[Dict[str, Any]]):
        def training_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> None:
            raise RuntimeError("boom")

    agent = FailingAgent()
    runner, store, _ = await setup_runner(agent)
    with pytest.raises(RuntimeError):
        await runner.step({"task": "x"})

    rollouts = await store.query_rollouts()
    assert len(rollouts) == 1
    attempts = await store.query_attempts(rollouts[0].rollout_id)
    assert attempts[-1].status == "failed"
    teardown_runner(runner)


@pytest.mark.asyncio
async def test_agent_emits_multiple_rewards() -> None:
    class RewardListAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> List[ReadableSpan]:
            return [emit_reward(0.2), emit_reward(0.6)]

    agent = RewardListAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        await runner.step({"task": "reward"})
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    reward_values = [span.attributes.get("reward") for span in spans if span.name == SpanNames.REWARD.value]
    assert reward_values == [0.2, 0.6]


@pytest.mark.asyncio
async def test_hooks_triggered_in_order() -> None:
    hook = RecordingHook()

    class HookAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> List[ReadableSpan]:
            return [create_readable_span("hook-span")]

    agent = HookAgent()
    runner, store, _ = await setup_runner(agent, hooks=[hook])
    try:
        await runner.step({"task": "hook"})
    finally:
        teardown_runner(runner)

    assert hook.calls == ["on_rollout_start", "on_trace_start", "on_trace_end", "on_rollout_end"]
    assert hook.received_spans is not None
    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    assert [span.name for span in spans] == ["hook-span"]


@pytest.mark.asyncio
async def test_step_returns_completed_rollout() -> None:
    """Test that step() returns a RolloutV2 object after execution."""

    class SimpleAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 0.85

    agent = SimpleAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        result = await runner.step({"task": "test"})
    finally:
        teardown_runner(runner)

    # Verify the result is a RolloutV2 object
    assert isinstance(result, RolloutV2)
    assert result.status == "succeeded"
    assert result.input == {"task": "test"}

    # Verify the rollout was stored correctly
    rollouts = await store.query_rollouts()
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == result.rollout_id


@pytest.mark.asyncio
async def test_step_returns_rollout_with_spans() -> None:
    """Test that the returned rollout can be used to query spans."""

    class SpanAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> List[ReadableSpan]:
            return [create_readable_span("test-span-1"), create_readable_span("test-span-2")]

    agent = SpanAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        result = await runner.step({"task": "test"})
    finally:
        teardown_runner(runner)

    # Verify we can query spans using the returned rollout
    attempts = await store.query_attempts(result.rollout_id)
    assert len(attempts) > 0
    spans = await store.query_spans(result.rollout_id, attempts[-1].attempt_id)
    assert len(spans) == 2
    assert [span.name for span in spans] == ["test-span-1", "test-span-2"]


@pytest.mark.asyncio
async def test_step_raises_when_rollout_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that step() raises RuntimeError when completed rollout cannot be fetched."""

    class SimpleAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 0.5

    agent = SimpleAgent()
    runner, store, _ = await setup_runner(agent)

    # Mock get_rollout_by_id to return None
    original_get_rollout = store.get_rollout_by_id

    async def mock_get_rollout_by_id(rollout_id: str) -> Optional[RolloutV2]:
        return None

    monkeypatch.setattr(store, "get_rollout_by_id", mock_get_rollout_by_id)

    try:
        with pytest.raises(RuntimeError, match="Failed to fetch completed rollout by id after step"):
            await runner.step({"task": "test"})
    finally:
        monkeypatch.setattr(store, "get_rollout_by_id", original_get_rollout)
        teardown_runner(runner)


@pytest.mark.asyncio
async def test_step_impl_returns_rollout_id() -> None:
    """Test that _step_impl returns the rollout_id after execution."""

    class SimpleAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 0.9

    agent = SimpleAgent()
    runner, store, _ = await setup_runner(agent)

    # Create an attempted rollout
    attempted_rollout = await store.start_rollout(input={"task": "test"}, mode="val")

    try:
        # Call _step_impl directly and verify it returns rollout_id
        result = await runner._step_impl(  # pyright: ignore[reportPrivateUsage]
            attempted_rollout, raise_on_exception=True
        )
    finally:
        teardown_runner(runner)

    # Verify the result is a string (rollout_id)
    assert isinstance(result, str)
    assert result == attempted_rollout.rollout_id


@pytest.mark.asyncio
async def test_step_impl_returns_rollout_id_on_resource_failure() -> None:
    """Test that _step_impl returns rollout_id even when resources fail to fetch."""

    class SimpleAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 0.9

    agent = SimpleAgent()
    runner, store, _ = await setup_runner(agent)

    # Create an attempted rollout with invalid resources_id
    attempted_rollout = await store.start_rollout(input={"task": "test"}, mode="val", resources_id="invalid-id")

    try:
        # Call _step_impl with raise_on_exception=False (to test the early return path)
        result = await runner._step_impl(  # pyright: ignore[reportPrivateUsage]
            attempted_rollout, raise_on_exception=False
        )
    finally:
        teardown_runner(runner)

    # Verify the result is a string (rollout_id) even on early return
    assert isinstance(result, str)
    assert result == attempted_rollout.rollout_id


@pytest.mark.asyncio
async def test_step_with_custom_resources_returns_rollout() -> None:
    """Test that step() with custom resources returns a valid RolloutV2."""

    class ResourceAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            # Verify we received the custom LLM
            llm = resources.get("llm")
            assert llm is not None
            assert llm.model == "custom-model"
            return 0.95

    agent = ResourceAgent()
    runner, _store, _ = await setup_runner(agent)

    custom_resources: NamedResources = {"llm": LLM(endpoint="http://custom", model="custom-model")}

    try:
        result = await runner.step({"task": "test"}, resources=custom_resources)
    finally:
        teardown_runner(runner)

    # Verify the result is a valid RolloutV2
    assert isinstance(result, RolloutV2)
    assert result.status == "succeeded"
    assert result.input == {"task": "test"}

    # Verify the rollout has the correct resources_id
    assert result.resources_id is not None
