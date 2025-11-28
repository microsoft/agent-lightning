# Copyright (c) Microsoft. All rights reserved.

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, cast
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store.base import UNSET, LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.store.threading import LightningStoreThreaded
from agentlightning.types import (
    LLM,
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    OtelResource,
    ResourcesUpdate,
    Rollout,
    Span,
    SpanContext,
    TaskInput,
    TraceStatus,
    Worker,
)

from .dummy_store import DummyLightningStore


class SlowAttemptStore(LightningStore):
    def __init__(self) -> None:
        super().__init__()
        self.active_calls = 0
        self.max_active_calls = 0
        self.sequence = 0

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str,
        status: AttemptStatus | Any = UNSET,
        worker_id: str | Any = UNSET,
        last_heartbeat_time: float | Any = UNSET,
        metadata: Dict[str, Any] | Any = UNSET,
    ) -> Attempt:
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.01)
        self.active_calls -= 1

        self.sequence += 1
        status_value = status if status is not UNSET else "preparing"
        worker_value = worker_id if worker_id is not UNSET else None
        heartbeat_value = last_heartbeat_time if last_heartbeat_time is not UNSET else None
        metadata_value = metadata if metadata is not UNSET else {}

        return Attempt(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=self.sequence,
            start_time=float(self.sequence),
            status=status_value,
            worker_id=worker_value,
            last_heartbeat_time=heartbeat_value,
            metadata=metadata_value,
        )


class IncrementingResourceStore(LightningStore):
    def __init__(self) -> None:
        super().__init__()
        self.counter = 0

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        snapshot = self.counter
        await asyncio.sleep(0.01)
        self.counter = snapshot + 1
        return ResourcesUpdate(
            resources_id=f"res-{self.counter}",
            resources=resources,
            create_time=time.time(),
            update_time=time.time(),
            version=1,
        )


def make_span(rollout_id: str, attempt_id: str, sequence_id: int = 1) -> Span:
    return Span(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
        trace_id="0" * 32,
        span_id="0" * 16,
        parent_id=None,
        name="test-span",
        status=TraceStatus(status_code="OK"),
        attributes={},
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=SpanContext(trace_id="0" * 32, span_id="0" * 16, is_remote=False, trace_state={}),
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


@pytest.mark.asyncio
async def test_threaded_store_delegates_all_methods() -> None:
    task_input: TaskInput = {"foo": "bar"}
    rollout_id = "rollout-1"
    attempt_id = "attempt-1"

    base_attempt = Attempt(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=1,
        start_time=0.0,
    )
    base_rollout = Rollout(
        rollout_id=rollout_id,
        input=task_input,
        start_time=0.0,
        mode="train",
    )
    attempted_rollout = AttemptedRollout(
        rollout_id=rollout_id,
        input=task_input,
        start_time=0.0,
        mode="train",
        status="preparing",
        metadata={},
        attempt=base_attempt,
    )
    resources_update = ResourcesUpdate(
        resources_id="resources-1", resources={}, create_time=time.time(), update_time=time.time(), version=1
    )
    span = make_span(rollout_id, attempt_id)
    readable_span = MagicMock(spec=ReadableSpan)

    updated_rollout = Rollout(
        rollout_id=rollout_id,
        input=task_input,
        start_time=1.0,
        mode="val",
        status="succeeded",
        resources_id="resources-2",
        metadata={"note": "done"},
    )
    updated_attempt = Attempt(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=2,
        start_time=1.0,
        status="running",
        worker_id="worker-1",
        last_heartbeat_time=1.5,
        metadata={"idx": 0},
    )
    worker_list = [Worker(worker_id="worker-1", status="busy")]
    updated_worker = Worker(worker_id="worker-1", status="idle")

    return_values = {
        "start_rollout": attempted_rollout,
        "enqueue_rollout": base_rollout,
        "dequeue_rollout": attempted_rollout,
        "start_attempt": attempted_rollout,
        "query_rollouts": [base_rollout],
        "query_attempts": [base_attempt],
        "get_rollout_by_id": base_rollout,
        "get_latest_attempt": base_attempt,
        "update_resources": resources_update,
        "get_resources_by_id": resources_update,
        "get_latest_resources": resources_update,
        "add_span": span,
        "add_many_spans": [span],
        "add_otel_span": span,
        "wait_for_rollouts": [base_rollout],
        "get_next_span_sequence_id": 42,
        "get_many_span_sequence_ids": [101],
        "query_spans": [span],
        "update_rollout": updated_rollout,
        "update_attempt": updated_attempt,
        "query_workers": worker_list,
        "get_worker_by_id": worker_list[0],
        "update_worker": updated_worker,
    }

    dummy_store = DummyLightningStore(return_values)
    threaded_store = LightningStoreThreaded(dummy_store)

    assert (
        await threaded_store.start_rollout(task_input, mode="train", resources_id="resources-1", metadata={"a": 1})
        == attempted_rollout
    )
    assert (
        await threaded_store.enqueue_rollout(task_input, mode="train", resources_id="resources-1", metadata={"b": 2})
        == base_rollout
    )
    assert await threaded_store.dequeue_rollout() == attempted_rollout
    assert await threaded_store.start_attempt(rollout_id) == attempted_rollout
    assert await threaded_store.query_rollouts(status=["preparing", "running"], rollout_ids=[rollout_id]) == [
        base_rollout
    ]
    assert await threaded_store.query_attempts(rollout_id) == [base_attempt]
    assert await threaded_store.get_rollout_by_id(rollout_id) == base_rollout
    assert await threaded_store.get_latest_attempt(rollout_id) == base_attempt
    assert await threaded_store.update_resources("resources-1", {}) == resources_update
    assert await threaded_store.get_resources_by_id("resources-1") == resources_update
    assert await threaded_store.get_latest_resources() == resources_update
    assert await threaded_store.add_span(span) == span
    assert await threaded_store.add_many_spans([span]) == [span]
    assert await threaded_store.add_otel_span(rollout_id, attempt_id, readable_span, sequence_id=5) == span
    assert await threaded_store.wait_for_rollouts(rollout_ids=[rollout_id], timeout=1.0) == [base_rollout]
    assert await threaded_store.get_next_span_sequence_id(rollout_id, attempt_id) == 42
    assert await threaded_store.get_many_span_sequence_ids([(rollout_id, attempt_id)]) == [101]
    assert await threaded_store.query_spans(rollout_id, attempt_id="latest") == [span]
    assert (
        await threaded_store.update_rollout(
            rollout_id,
            input=task_input,
            mode="val",
            resources_id="resources-2",
            status="succeeded",
            metadata={"note": "done"},
        )
        == updated_rollout
    )
    assert (
        await threaded_store.update_attempt(
            rollout_id,
            attempt_id,
            status="running",
            worker_id="worker-1",
            last_heartbeat_time=1.5,
            metadata={"idx": 0},
        )
        == updated_attempt
    )
    assert await threaded_store.query_workers() == worker_list
    assert await threaded_store.get_worker_by_id("worker-1") == worker_list[0]
    assert await threaded_store.update_worker("worker-1", heartbeat_stats={"cpu": 0.5}) == updated_worker

    expected_order = [
        "start_rollout",
        "enqueue_rollout",
        "dequeue_rollout",
        "start_attempt",
        "query_rollouts",
        "query_attempts",
        "get_rollout_by_id",
        "get_latest_attempt",
        "update_resources",
        "get_resources_by_id",
        "get_latest_resources",
        "add_span",
        "add_many_spans",
        "add_otel_span",
        "wait_for_rollouts",
        "get_next_span_sequence_id",
        "get_many_span_sequence_ids",
        "query_spans",
        "update_rollout",
        "update_attempt",
        "query_workers",
        "get_worker_by_id",
        "update_worker",
    ]
    assert [name for name, *_ in dummy_store.calls] == expected_order


def test_threaded_store_serializes_update_attempt_calls() -> None:
    store = SlowAttemptStore()
    threaded_store = LightningStoreThreaded(store)
    rollout_id = "rollout-race"
    num_calls = 5

    def invoke(idx: int) -> Attempt:
        return asyncio.run(
            threaded_store.update_attempt(
                rollout_id,
                f"attempt-{idx}",
                status="running",
                worker_id=f"worker-{idx}",
                last_heartbeat_time=float(idx),
                metadata={"idx": idx},
            )
        )

    with ThreadPoolExecutor(max_workers=num_calls) as executor:
        results = list(executor.map(invoke, range(num_calls)))

    assert store.max_active_calls == 1
    assert len(results) == num_calls
    assert sorted(result.attempt_id for result in results) == [f"attempt-{i}" for i in range(num_calls)]


def test_threaded_store_prevents_race_conditions_on_resource_updates() -> None:
    store = IncrementingResourceStore()
    threaded_store = LightningStoreThreaded(store)
    num_updates = 20

    def invoke(idx: int) -> ResourcesUpdate:
        return asyncio.run(threaded_store.update_resources(f"resources-{idx}", {}))

    with ThreadPoolExecutor(max_workers=num_updates) as executor:
        updates = list(executor.map(invoke, range(num_updates)))

    assert store.counter == num_updates
    assert len(updates) == num_updates
    assert {update.resources_id for update in updates} == {f"res-{i + 1}" for i in range(num_updates)}


@pytest.mark.asyncio
async def test_threaded_store_add_resources_delegates() -> None:
    """Test that threaded store delegates add_resources calls correctly."""
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )
    resources: NamedResources = cast(NamedResources, {"main_llm": llm})
    resources_update = ResourcesUpdate(
        resources_id="resources-1", resources=resources, create_time=time.time(), update_time=time.time(), version=1
    )

    return_values = {
        "add_resources": resources_update,
    }

    dummy_store = DummyLightningStore(return_values)
    threaded_store = LightningStoreThreaded(dummy_store)

    result = await threaded_store.add_resources(resources)
    assert result == resources_update
    assert len(dummy_store.calls) == 1
    assert dummy_store.calls[0][0] == "add_resources"
    assert dummy_store.calls[0][1] == (resources,)


def test_threaded_store_serializes_add_resources_calls() -> None:
    """Test that concurrent add_resources calls are serialized by the thread lock."""
    store = InMemoryLightningStore()
    threaded_store = LightningStoreThreaded(store)
    num_adds = 10

    def invoke(idx: int) -> ResourcesUpdate:
        llm = LLM(
            resource_type="llm",
            endpoint=f"http://localhost:808{idx % 10}",
            model=f"model-{idx}",
        )
        resources: NamedResources = cast(NamedResources, {"llm": llm})
        return asyncio.run(threaded_store.add_resources(resources))

    with ThreadPoolExecutor(max_workers=num_adds) as executor:
        updates = list(executor.map(invoke, range(num_adds)))

    # Verify all calls were made and resources_ids are unique
    assert len(updates) == num_adds
    resource_ids = {update.resources_id for update in updates}
    assert len(resource_ids) == num_adds  # All IDs should be unique


@pytest.mark.asyncio
async def test_threaded_store_statistics_match_underlying(inmemory_store: InMemoryLightningStore) -> None:
    """Threaded store should proxy statistics from wrapped store."""
    rollout = await inmemory_store.start_rollout(input={"source": "threaded-stats"})
    await inmemory_store.add_span(make_span(rollout.rollout_id, rollout.attempt.attempt_id))
    await inmemory_store.add_resources(
        {
            "threaded_llm": LLM(
                resource_type="llm",
                endpoint="http://localhost:9000/v1",
                model="threaded-model",
            )
        }
    )
    await inmemory_store.update_worker("threaded-worker", heartbeat_stats={"cpu": 0.1})

    threaded_store = LightningStoreThreaded(inmemory_store)
    threaded_stats = await threaded_store.statistics()
    inmemory_stats = await inmemory_store.statistics()
    assert {k: v for k, v in threaded_stats.items() if k != "uptime"} == {
        k: v for k, v in inmemory_stats.items() if k != "uptime"
    }
    assert threaded_stats["uptime"] < inmemory_stats["uptime"]  # type: ignore
