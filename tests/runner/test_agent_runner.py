# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Sequence, Tuple, cast

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.trace import SpanContext, TraceFlags, TraceState
from opentelemetry.trace.status import Status, StatusCode

from agentlightning.execution.events import ExecutionEvent, ThreadingEvent
from agentlightning.litagent import LitAgent
from agentlightning.reward import emit_reward, find_final_reward
from agentlightning.runner import LitAgentRunner
from agentlightning.runner.base import Runner
from agentlightning.semconv import AGL_ANNOTATION
from agentlightning.store.base import UNSET, LightningStore, Unset
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.base import Tracer
from agentlightning.types import LLM, Hook, NamedResources, PromptTemplate, Rollout, Span, SpanCoreFields, Worker


@pytest.fixture(scope="module", autouse=True)
def setup_module():
    trace_api.set_tracer_provider(TracerProvider())
    yield


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


class DummyTracer(Tracer):
    def __init__(self) -> None:
        super().__init__()
        self._last_trace: List[Span] = []
        self._contexts: List[Dict[str, Any]] = []
        self._sequence_id = 0

    def init(self, *args: Any, **kwargs: Any) -> None:
        self._last_trace.clear()

    def teardown(self, *args: Any, **kwargs: Any) -> None:
        self._last_trace.clear()

    def get_last_trace(self) -> List[Span]:
        return list(self._last_trace)

    @asynccontextmanager
    async def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncGenerator[List[Span], None]:
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
        rollout_id = "rollout-dummy"
        attempt_id = "attempt-dummy"
        sequence_id = self._sequence_id
        self._sequence_id += 1
        if self._contexts:
            current = self._contexts[-1]
            rollout_id = current["rollout_id"]
            attempt_id = current["attempt_id"]
        self._last_trace.append(Span.from_opentelemetry(span, rollout_id, attempt_id, sequence_id))
        return span


class RecordingStore(InMemoryLightningStore):
    """In-memory store that records worker heartbeat updates for inspection in tests."""

    def __init__(self) -> None:
        super().__init__()
        self.worker_updates: List[Tuple[str, Optional[Dict[str, Any]]]] = []

    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        payload = None if isinstance(heartbeat_stats, Unset) else heartbeat_stats
        self.worker_updates.append((worker_id, payload))
        return await super().update_worker(worker_id, heartbeat_stats=heartbeat_stats)


class HeartbeatAgent(LitAgent[Dict[str, Any]]):
    """Minimal agent used for heartbeat-only runner tests."""

    def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
        return 0.0


async def setup_heartbeat_runner(
    *,
    heartbeat_interval: float = 0.05,
    heartbeat_launch_mode: Literal["asyncio", "thread"] = "thread",
    heartbeat_include_gpu: bool = False,
) -> tuple[LitAgentRunner[Any], RecordingStore]:
    """Create a runner wired to a RecordingStore for heartbeat tests."""

    store = RecordingStore()
    runner = LitAgentRunner[Any](
        tracer=DummyTracer(),
        heartbeat_interval=heartbeat_interval,
        heartbeat_launch_mode=heartbeat_launch_mode,
        heartbeat_include_gpu=heartbeat_include_gpu,
    )
    agent = HeartbeatAgent()
    runner.init(agent)
    runner.init_worker(worker_id=0, store=store)
    return runner, store


async def setup_runner(
    agent: LitAgent[Any],
    *,
    tracer: Optional[DummyTracer] = None,
    max_rollouts: Optional[int] = None,
    poll_interval: float = 0.01,
    hooks: Sequence[Hook] = (),
) -> tuple[LitAgentRunner[Any], InMemoryLightningStore, DummyTracer]:
    tracer = tracer or DummyTracer()
    store = InMemoryLightningStore()
    await store.update_resources("default", {"llm": LLM(endpoint="http://localhost", model="dummy")})

    runner = LitAgentRunner[Any](tracer=tracer, max_rollouts=max_rollouts, poll_interval=poll_interval)
    runner.init(agent=agent, hooks=hooks)
    runner.init_worker(worker_id=0, store=store)
    return runner, store, tracer


def teardown_runner(runner: LitAgentRunner[Any]) -> None:
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

    async def on_rollout_start(self, *, agent: LitAgent[Any], runner: Runner[Any], rollout: Rollout) -> None:
        self.calls.append("on_rollout_start")

    async def on_trace_start(
        self, *, agent: LitAgent[Any], runner: Runner[Any], tracer: Tracer, rollout: Rollout
    ) -> None:
        self.calls.append("on_trace_start")

    async def on_trace_end(
        self, *, agent: LitAgent[Any], runner: Runner[Any], tracer: Tracer, rollout: Rollout
    ) -> None:
        self.calls.append("on_trace_end")

    async def on_rollout_end(
        self,
        *,
        agent: LitAgent[Any],
        runner: Runner[Any],
        rollout: Rollout,
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
            store = cast(LitAgentRunner[Dict[str, Any]], self.runner).get_store()
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
    rewards = [span.attributes.get("agentlightning.reward.0.value") for span in spans if span.name == AGL_ANNOTATION]
    assert rewards == [0.75]


@pytest.mark.asyncio
async def test_step_emits_reward_for_bool_result(caplog: pytest.LogCaptureFixture) -> None:
    class BoolRewardAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> bool:
            return True

    agent = BoolRewardAgent()
    runner, store, _ = await setup_runner(agent)
    caplog.set_level(logging.WARNING)
    try:
        await runner.step({"prompt": "hello"})
    finally:
        teardown_runner(runner)

    assert "Reward is not a number" in caplog.text
    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    rewards = [span.attributes.get("agentlightning.reward.0.value") for span in spans if span.name == AGL_ANNOTATION]
    assert rewards == [1.0]


@pytest.mark.asyncio
async def test_step_raises_for_invalid_result_type() -> None:
    class InvalidResultAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(  # type: ignore[reportIncompatibleMethodOverride]
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> Dict[str, Any]:
            return {"unexpected": True}

    agent = InvalidResultAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        with pytest.raises(TypeError, match="Invalid raw result type"):
            await runner.step({"task": "bad-result"})
    finally:
        teardown_runner(runner)

    rollouts = await store.query_rollouts()
    assert len(rollouts) == 1
    attempts = await store.query_attempts(rollouts[0].rollout_id)
    assert attempts[-1].status == "failed"


@pytest.mark.asyncio
async def test_readable_spans_return_skip_store_when_tracer_is_otel(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class DummyTracerWithOtel(DummyTracer):
        pass

    class RecordingStore(InMemoryLightningStore):
        def __init__(self) -> None:
            super().__init__()
            self.add_otel_span_calls = 0

        async def add_otel_span(
            self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
        ) -> Span | None:
            self.add_otel_span_calls += 1
            return await super().add_otel_span(rollout_id, attempt_id, readable_span, sequence_id)

    monkeypatch.setattr("agentlightning.runner.agent.OtelTracer", DummyTracerWithOtel)

    store = RecordingStore()
    await store.update_resources("default", {"llm": LLM(endpoint="http://localhost", model="dummy")})

    tracer = DummyTracerWithOtel()
    runner = LitAgentRunner[Any](tracer=tracer)
    agent = HeartbeatAgent()
    runner.init(agent)
    runner.init_worker(worker_id=0, store=store)

    attempted_rollout = await store.start_rollout(input={"task": "otel"}, mode="val")
    spans = [create_readable_span("otel-span")]

    result_spans = await runner._post_process_rollout_result(  # pyright: ignore[reportPrivateUsage]
        attempted_rollout, spans
    )

    assert all(isinstance(span, Span) for span in result_spans)
    # Warned, but still logged
    assert [span.name for span in result_spans] == ["otel-span"]
    assert store.add_otel_span_calls == 1
    assert "Tracer is already an OpenTelemetry tracer" in caplog.text

    teardown_runner(runner)


@pytest.mark.asyncio
async def test_post_process_readable_spans_adds_each_span() -> None:
    agent = HeartbeatAgent()
    runner, store, _ = await setup_runner(agent)
    attempted_rollout = await store.start_rollout(input={"task": "post-process"}, mode="val")

    spans = [create_readable_span("case-2-span-a"), create_readable_span("case-2-span-b")]

    try:
        result_spans = await runner._post_process_rollout_result(  # pyright: ignore[reportPrivateUsage]
            attempted_rollout, spans
        )
    finally:
        teardown_runner(runner)

    assert [span.name for span in result_spans] == ["case-2-span-a", "case-2-span-b"]
    stored_spans = await store.query_spans(attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id)
    assert [span.name for span in stored_spans] == ["case-2-span-a", "case-2-span-b"]


@pytest.mark.asyncio
async def test_post_process_span_core_fields_create_spans() -> None:
    agent = HeartbeatAgent()
    runner, store, _ = await setup_runner(agent)
    attempted_rollout = await store.start_rollout(input={"task": "reward-list"}, mode="val")

    span_core_fields = [emit_reward(0.5, propagate=False), emit_reward(-0.2, propagate=False)]

    try:
        result_spans = await runner._post_process_rollout_result(  # pyright: ignore[reportPrivateUsage]
            attempted_rollout, span_core_fields
        )
    finally:
        teardown_runner(runner)

    assert all(span.name == AGL_ANNOTATION for span in result_spans)
    stored_spans = await store.query_spans(attempted_rollout.rollout_id, attempted_rollout.attempt.attempt_id)
    assert [span.attributes.get("agentlightning.reward.0.value") for span in stored_spans] == [0.5, -0.2]


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
    rewards = [span.attributes.get("agentlightning.reward.0.value") for span in spans if span.name == AGL_ANNOTATION]
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

    async def fake_sleep(event: Optional[ExecutionEvent] = None) -> None:
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
        ) -> List[SpanCoreFields]:
            return [emit_reward(0.2, propagate=False), emit_reward(0.6, propagate=False)]

    agent = RewardListAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        await runner.step({"task": "reward"})
    finally:
        teardown_runner(runner)

    rollout_id, attempt_id = await assert_single_attempt_succeeded(store)
    spans = await store.query_spans(rollout_id, attempt_id)
    reward_values = [
        span.attributes.get("agentlightning.reward.0.value") for span in spans if span.name == AGL_ANNOTATION
    ]
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
    """Test that step() returns a Rollout object after execution."""

    class SimpleAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 0.85

    agent = SimpleAgent()
    runner, store, _ = await setup_runner(agent)
    try:
        result = await runner.step({"task": "test"})
    finally:
        teardown_runner(runner)

    # Verify the result is a Rollout object
    assert isinstance(result, Rollout)
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

    async def mock_get_rollout_by_id(rollout_id: str) -> Optional[Rollout]:
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
    """Test that step() with custom resources returns a valid Rollout."""

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

    # Verify the result is a valid Rollout
    assert isinstance(result, Rollout)
    assert result.status == "succeeded"
    assert result.input == {"task": "test"}

    # Verify the rollout has the correct resources_id
    assert result.resources_id is not None


@pytest.mark.asyncio
async def test_step_registers_worker_id_on_start_rollout(monkeypatch: pytest.MonkeyPatch) -> None:
    """runner.step should pass the formatted worker ID down to the store."""

    class WorkerAwareAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any) -> float:
            return 1.0

    agent = WorkerAwareAgent()
    runner, store, _ = await setup_runner(agent)

    expected_worker_label = runner.get_worker_id()
    captured: Dict[str, Optional[str]] = {}
    original_start_rollout = store.start_rollout

    async def wrapped_start_rollout(*args: Any, **kwargs: Any):
        captured["worker_id"] = kwargs.get("worker_id")
        return await original_start_rollout(*args, **kwargs)

    monkeypatch.setattr(store, "start_rollout", wrapped_start_rollout)

    try:
        await runner.step({"task": "worker-aware"})
    finally:
        teardown_runner(runner)

    assert captured["worker_id"] == expected_worker_label


@pytest.mark.asyncio
async def test_iter_passes_worker_id_to_dequeue(monkeypatch: pytest.MonkeyPatch) -> None:
    """iter() should poll the store with the formatted worker identifier."""

    class IdleAgent(LitAgent[Dict[str, Any]]):
        def validation_rollout(
            self, task: Dict[str, Any], resources: Dict[str, Any], rollout: Any
        ) -> float:  # pragma: no cover - not invoked
            return 0.0

    agent = IdleAgent()
    runner, store, _ = await setup_runner(agent, poll_interval=0.01)

    expected_worker_label = runner.get_worker_id()
    captured: Dict[str, Optional[str]] = {}
    event = ThreadingEvent()

    async def fake_dequeue(*, worker_id: Optional[str] = None):
        captured["worker_id"] = worker_id
        event.set()
        return None

    async def fast_sleep(self: LitAgentRunner[Any], event: Optional[ExecutionEvent] = None) -> None:
        if event is not None:
            event.set()

    monkeypatch.setattr(store, "dequeue_rollout", fake_dequeue)
    monkeypatch.setattr(LitAgentRunner, "_sleep_until_next_poll", fast_sleep)

    try:
        await runner.iter(event=event)
    finally:
        teardown_runner(runner)

    assert captured["worker_id"] == expected_worker_label


@pytest.mark.asyncio
async def test_emit_heartbeat_updates_worker_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"cpu_pct": 42.0, "mem_pct": 10.5}
    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", lambda include_gpu=False: snapshot)

    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.1)
    worker_label = runner.get_worker_id()
    try:
        await runner._emit_heartbeat(store)  # pyright: ignore[reportPrivateUsage]
    finally:
        teardown_runner(runner)

    assert store.worker_updates == [(worker_label, snapshot)]
    worker = await store.get_worker_by_id(worker_label)
    assert worker is not None
    assert worker.heartbeat_stats == snapshot
    assert worker.last_heartbeat_time is not None


@pytest.mark.asyncio
async def test_emit_heartbeat_passes_include_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"cpu_pct": 11.1}
    requested_flags: List[bool] = []

    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr("agentlightning.runner.agent.asyncio.to_thread", immediate_to_thread)

    def fake_system_snapshot(include_gpu: bool = False) -> Dict[str, Any]:
        requested_flags.append(include_gpu)
        return snapshot

    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", fake_system_snapshot)

    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.05, heartbeat_include_gpu=True)
    worker_label = runner.get_worker_id()
    try:
        await runner._emit_heartbeat(store)  # pyright: ignore[reportPrivateUsage]
    finally:
        teardown_runner(runner)

    assert requested_flags == [True]
    assert store.worker_updates == [(worker_label, snapshot)]


@pytest.mark.asyncio
async def test_emit_heartbeat_skips_when_snapshot_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"cpu_pct": 9.9}
    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.05)
    interval = runner._heartbeat_interval  # pyright: ignore[reportPrivateUsage]

    async def slow_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(interval + 0.05)
        return func(*args, **kwargs)

    monkeypatch.setattr("agentlightning.runner.agent.asyncio.to_thread", slow_to_thread)
    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", lambda include_gpu=False: snapshot)

    try:
        await runner._emit_heartbeat(store)  # pyright: ignore[reportPrivateUsage]
    finally:
        teardown_runner(runner)

    assert store.worker_updates == []


@pytest.mark.asyncio
async def test_emit_heartbeat_skips_when_store_update_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"cpu_pct": 8.8}

    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    monkeypatch.setattr("agentlightning.runner.agent.asyncio.to_thread", immediate_to_thread)
    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", lambda include_gpu=False: snapshot)

    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.05)
    interval = runner._heartbeat_interval  # pyright: ignore[reportPrivateUsage]
    original_update_worker = store.update_worker

    async def slow_update_worker(*args: Any, **kwargs: Any) -> Worker:
        await asyncio.sleep(interval + 0.05)
        return await original_update_worker(*args, **kwargs)

    monkeypatch.setattr(store, "update_worker", slow_update_worker)

    try:
        await runner._emit_heartbeat(store)  # pyright: ignore[reportPrivateUsage]
    finally:
        teardown_runner(runner)

    assert store.worker_updates == []


@pytest.mark.asyncio
async def test_heartbeat_loop_runs_until_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that heartbeat loop runs with the default launch mode until stopped."""
    snapshot = {"timestamp": 1234567890}
    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", lambda include_gpu=False: snapshot)

    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.05)
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        await asyncio.sleep(0.12)
    finally:
        await stop_heartbeat()

    update_count = len(store.worker_updates)
    assert update_count >= 1
    assert all(stats == snapshot for _, stats in store.worker_updates if stats is not None)

    await asyncio.sleep(0.06)
    assert len(store.worker_updates) == update_count

    teardown_runner(runner)


@pytest.mark.asyncio
async def test_asyncio_heartbeat_loop_runs_until_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that asyncio heartbeat loop runs until stopped (explicit asyncio mode test)."""
    snapshot = {"timestamp": 1234567890}
    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", lambda include_gpu=False: snapshot)

    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.05, heartbeat_launch_mode="asyncio")
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        await asyncio.sleep(0.12)
    finally:
        await stop_heartbeat()

    update_count = len(store.worker_updates)
    assert update_count >= 1
    assert all(stats == snapshot for _, stats in store.worker_updates if stats is not None)

    await asyncio.sleep(0.06)
    assert len(store.worker_updates) == update_count

    teardown_runner(runner)


@pytest.mark.asyncio
async def test_thread_heartbeat_loop_runs_until_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = {"timestamp": time.time()}
    include_flags: List[bool] = []

    def fake_system_snapshot(include_gpu: bool = False) -> Dict[str, Any]:
        include_flags.append(include_gpu)
        return snapshot

    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", fake_system_snapshot)

    runner, store = await setup_heartbeat_runner(
        heartbeat_interval=0.05,
        heartbeat_launch_mode="thread",
        heartbeat_include_gpu=True,
    )
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        await asyncio.sleep(0.3)
    finally:
        await stop_heartbeat()
        teardown_runner(runner)

    assert len(store.worker_updates) >= 1
    assert all(stats == snapshot for _, stats in store.worker_updates if stats is not None)
    assert include_flags and all(include_flags)


@pytest.mark.asyncio
async def test_thread_heartbeat_handles_producer_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that thread mode producer handles exceptions and continues running."""
    call_count = 0

    def fake_system_snapshot(_include_gpu: bool = False) -> Dict[str, Any]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated snapshot failure")
        return {"cpu_pct": 25.0}

    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", fake_system_snapshot)

    caplog.set_level(logging.WARNING)
    runner, store = await setup_heartbeat_runner(
        heartbeat_interval=0.02,
        heartbeat_launch_mode="thread",
    )
    runner._interval_jitter = 0.01  # pyright: ignore[reportPrivateUsage]
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        # Wait long enough for at least 2 cycles (2 * (interval + jitter) + buffer)
        await asyncio.sleep(0.15)
    finally:
        await stop_heartbeat()
        teardown_runner(runner)

    # Verify the producer logged the exception but continued
    assert "system_snapshot failed" in caplog.text
    assert call_count >= 2


@pytest.mark.asyncio
async def test_thread_heartbeat_handles_consumer_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that thread mode consumer handles store update exceptions and continues."""
    snapshot = {"cpu_pct": 33.0}
    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", lambda include_gpu=False: snapshot)

    runner, store = await setup_heartbeat_runner(
        heartbeat_interval=0.02,
        heartbeat_launch_mode="thread",
    )
    runner._interval_jitter = 0.01  # pyright: ignore[reportPrivateUsage]

    update_count = 0
    original_update_worker = store.update_worker

    async def failing_update_worker(*args: Any, **kwargs: Any) -> Worker:
        nonlocal update_count
        update_count += 1
        if update_count == 1:
            raise RuntimeError("Simulated store failure")
        return await original_update_worker(*args, **kwargs)

    monkeypatch.setattr(store, "update_worker", failing_update_worker)

    caplog.set_level(logging.WARNING)
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        # Wait long enough for at least 2 cycles
        await asyncio.sleep(0.15)
    finally:
        await stop_heartbeat()
        teardown_runner(runner)

    # Verify the consumer logged the exception but continued
    assert "update failed" in caplog.text
    assert update_count >= 2


@pytest.mark.asyncio
async def test_thread_heartbeat_waits_for_first_snapshot(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that thread mode consumer skips update when no snapshot is available yet."""
    snapshot_ready = False

    def fake_system_snapshot(_include_gpu: bool = False) -> Dict[str, Any]:
        nonlocal snapshot_ready
        if not snapshot_ready:
            time.sleep(0.1)  # Simulate slow first snapshot
            snapshot_ready = True
        return {"cpu_pct": 42.0}

    monkeypatch.setattr("agentlightning.runner.agent.system_snapshot", fake_system_snapshot)

    caplog.set_level(logging.DEBUG)
    runner, store = await setup_heartbeat_runner(
        heartbeat_interval=0.05,
        heartbeat_launch_mode="thread",
    )
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        await asyncio.sleep(0.25)
    finally:
        await stop_heartbeat()
        teardown_runner(runner)

    # Verify the consumer logged that no snapshot was available initially
    assert "no snapshot yet" in caplog.text


@pytest.mark.asyncio
async def test_heartbeat_disabled_when_interval_zero() -> None:
    """Test that heartbeat loop is not started when interval is 0 or negative."""
    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.0)
    try:
        stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
        assert stop_heartbeat is None
    finally:
        teardown_runner(runner)


@pytest.mark.asyncio
async def test_heartbeat_disabled_when_no_worker_id(caplog: pytest.LogCaptureFixture) -> None:
    """Test that heartbeat loop returns None when worker_id is not set."""
    store = RecordingStore()
    runner = LitAgentRunner[Any](
        tracer=DummyTracer(),
        heartbeat_interval=0.05,
    )
    agent = HeartbeatAgent()
    runner.init(agent)
    # Note: NOT calling init_worker, so worker_id remains None

    caplog.set_level(logging.WARNING)
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is None
    assert "Cannot start heartbeat loop without worker_id" in caplog.text

    teardown_runner(runner)


@pytest.mark.asyncio
async def test_emit_heartbeat_propagates_cancelled_error() -> None:
    """Test that CancelledError is properly propagated in _emit_heartbeat."""
    runner, store = await setup_heartbeat_runner(heartbeat_interval=0.1)

    async def cancelling_to_thread(_func: Any, *_args: Any, **_kwargs: Any) -> Any:
        raise asyncio.CancelledError()

    original_to_thread = asyncio.to_thread
    try:
        asyncio.to_thread = cancelling_to_thread  # type: ignore
        with pytest.raises(asyncio.CancelledError):
            await runner._emit_heartbeat(store)  # pyright: ignore[reportPrivateUsage]
    finally:
        asyncio.to_thread = original_to_thread  # type: ignore
        teardown_runner(runner)


@pytest.mark.asyncio
async def test_asyncio_heartbeat_continues_after_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that asyncio heartbeat loop continues running after exceptions."""
    call_count = 0

    async def failing_emit_heartbeat(_self: Any, _store: Any) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Simulated heartbeat failure")

    monkeypatch.setattr(LitAgentRunner, "_emit_heartbeat", failing_emit_heartbeat)

    caplog.set_level(logging.ERROR)
    runner, store = await setup_heartbeat_runner(
        heartbeat_interval=0.02,
        heartbeat_launch_mode="asyncio",
    )
    runner._interval_jitter = 0.01  # pyright: ignore[reportPrivateUsage]
    stop_heartbeat = runner._start_heartbeat_loop(store)  # pyright: ignore[reportPrivateUsage]
    assert stop_heartbeat is not None

    try:
        # Wait long enough for at least 2 cycles (2 * (interval + jitter) + buffer)
        await asyncio.sleep(0.15)
    finally:
        await stop_heartbeat()
        teardown_runner(runner)

    # Verify the loop logged the exception but continued
    assert "Heartbeat failed" in caplog.text
    assert call_count >= 2
