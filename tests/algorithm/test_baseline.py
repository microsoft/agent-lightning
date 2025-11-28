# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pytest

from agentlightning.adapter import TraceAdapter
from agentlightning.algorithm import Baseline
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import (
    LLM,
    NamedResources,
    OtelResource,
    Span,
    SpanContext,
    TraceStatus,
)

LOGGER_NAME = "agentlightning.algorithm.fast"


class _AdapterStub(TraceAdapter[Dict[str, Any]]):
    def adapt(self, source: Sequence[Span], /) -> Dict[str, Any]:
        return {
            "count": len(source),
            "attempt_ids": sorted({span.attempt_id for span in source}),
        }


@dataclass
class _RolloutArtifacts:
    rollout_id: str
    attempt_id: str
    attempt_sequence: int
    span: Span


def _make_resources() -> NamedResources:
    return {
        "main_llm": LLM(endpoint="http://localhost", model="test-model"),
    }


def _build_span(rollout_id: str, attempt_id: str, *, sequence_id: int, index: int) -> Span:
    trace_hex = f"{index:032x}"
    span_hex = f"{index:016x}"
    # Minimal span that passes validation and keeps log output predictable.
    return Span(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
        trace_id=trace_hex,
        span_id=span_hex,
        parent_id=None,
        name="test-span",
        status=TraceStatus(status_code="OK"),
        attributes={"stage": "collect"},
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=SpanContext(trace_id=trace_hex, span_id=span_hex, is_remote=False, trace_state={}),
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


async def _mock_runner(
    *,
    store: InMemoryLightningStore,
    expected: int,
    artifacts: List[_RolloutArtifacts],
) -> None:
    """Simulate a runner consuming rollouts, adding spans, and marking them complete."""
    processed = 0
    while processed < expected:
        attempted = await store.dequeue_rollout()
        if attempted is None:
            await asyncio.sleep(0.001)
            continue

        attempt = attempted.attempt
        rollout_id = attempted.rollout_id
        await store.update_attempt(
            rollout_id,
            attempt.attempt_id,
            status="running",
            worker_id="runner-1",
        )

        span = _build_span(rollout_id, attempt.attempt_id, sequence_id=1, index=processed + 1)
        await store.add_span(span)
        await store.update_attempt(rollout_id, attempt.attempt_id, status="succeeded")
        await store.update_rollout(rollout_id, status="succeeded")

        artifacts.append(
            _RolloutArtifacts(
                rollout_id=rollout_id,
                attempt_id=attempt.attempt_id,
                attempt_sequence=attempt.sequence_id,
                span=span,
            )
        )
        processed += 1


@pytest.mark.asyncio
async def test_mock_algorithm_collects_rollout_logs(caplog: pytest.LogCaptureFixture) -> None:
    store = InMemoryLightningStore()
    await store.update_resources("default", _make_resources())
    algorithm = Baseline(polling_interval=0.01, span_verbosity="key_values")
    algorithm.set_store(store)
    adapter = _AdapterStub()
    algorithm.set_adapter(adapter)

    caplog.set_level(logging.INFO, logger=LOGGER_NAME)

    train_dataset = ["train-sample", "validation-sample"]
    expected_rollouts = len(train_dataset)
    artifacts: List[_RolloutArtifacts] = []

    runner_task = asyncio.create_task(_mock_runner(store=store, expected=expected_rollouts, artifacts=artifacts))
    try:
        await algorithm.run(train_dataset=train_dataset)
        await asyncio.wait_for(runner_task, timeout=2)
    finally:
        if not runner_task.done():
            runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await runner_task

    log_messages = [record.getMessage() for record in caplog.records if record.name == LOGGER_NAME]

    # Ensure final status, attempt details, span details, and adapter output are logged per rollout.
    for entry in artifacts:
        attempt_summary = (
            f"[Rollout {entry.rollout_id} | Attempt {entry.attempt_sequence}] "
            f"ID: {entry.attempt_id}. Status: succeeded. Worker: runner-1"
        )
        assert attempt_summary in log_messages

        span_prefix = (
            f"[Rollout {entry.rollout_id} | Attempt {entry.attempt_id} | Span {entry.span.span_id}] "
            f"#{entry.span.sequence_id} ({entry.span.name}) "
        )
        assert any(msg.startswith(span_prefix + "From") for msg in log_messages)
        assert any(f"Attributes: {entry.span.attributes}" in msg for msg in log_messages)
        assert any(
            msg.startswith(f"[Rollout {entry.rollout_id}] Finished with status succeeded") for msg in log_messages
        )
        assert any(
            msg.startswith(f"[Rollout {entry.rollout_id}] Adapted data: ")
            and "'count': 1" in msg
            and entry.attempt_id in msg
            for msg in log_messages
        )


@pytest.mark.asyncio
async def test_baseline_does_not_skip_samples_when_queue_full() -> None:
    """Test that Baseline waits and retries when queue is full instead of skipping samples.

    This is a regression test for a bug where samples would be skipped when the queue
    exceeded max_queue_length. The fix wraps the queue check in a while loop to ensure
    all samples are eventually processed.
    """
    store = InMemoryLightningStore()
    await store.update_resources("default", _make_resources())

    # Use a small max_queue_length and fast polling to test queue full behavior
    algorithm = Baseline(polling_interval=0.01, max_queue_length=1)
    algorithm.set_store(store)

    # Create a dataset with 5 samples
    train_dataset = [f"sample-{i}" for i in range(5)]
    expected_rollouts = len(train_dataset)

    # Track which samples were enqueued
    enqueued_samples: List[Any] = []
    artifacts: List[_RolloutArtifacts] = []

    async def _slow_runner() -> None:
        """A slow runner that creates backpressure by processing rollouts with delays."""
        processed = 0
        while processed < expected_rollouts:
            attempted = await store.dequeue_rollout()
            if attempted is None:
                await asyncio.sleep(0.01)
                continue

            attempt = attempted.attempt
            rollout_id = attempted.rollout_id
            rollout = await store.get_rollout_by_id(rollout_id)

            # Track the sample that was enqueued
            if rollout:
                enqueued_samples.append(rollout.input)

            await store.update_attempt(
                rollout_id,
                attempt.attempt_id,
                status="running",
                worker_id="slow-runner",
            )

            # Add a delay to create backpressure and cause queue to fill up
            await asyncio.sleep(0.05)

            span = _build_span(rollout_id, attempt.attempt_id, sequence_id=1, index=processed + 1)
            await store.add_span(span)
            await store.update_attempt(rollout_id, attempt.attempt_id, status="succeeded")
            await store.update_rollout(rollout_id, status="succeeded")

            artifacts.append(
                _RolloutArtifacts(
                    rollout_id=rollout_id,
                    attempt_id=attempt.attempt_id,
                    attempt_sequence=attempt.sequence_id,
                    span=span,
                )
            )
            processed += 1

    runner_task = asyncio.create_task(_slow_runner())
    try:
        await algorithm.run(train_dataset=train_dataset)
        await asyncio.wait_for(runner_task, timeout=5)
    finally:
        if not runner_task.done():
            runner_task.cancel()
            with suppress(asyncio.CancelledError):
                await runner_task

    # Verify that ALL samples were enqueued and processed (no samples skipped)
    assert (
        len(enqueued_samples) == expected_rollouts
    ), f"Expected {expected_rollouts} samples to be enqueued, but got {len(enqueued_samples)}"
    assert (
        len(artifacts) == expected_rollouts
    ), f"Expected {expected_rollouts} rollouts to be processed, but got {len(artifacts)}"

    # Verify that the enqueued samples match the dataset (in order)
    assert (
        enqueued_samples == train_dataset
    ), f"Enqueued samples {enqueued_samples} do not match dataset {train_dataset}"
