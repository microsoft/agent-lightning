# Copyright (c) Microsoft. All rights reserved.

"""Tests for the SqliteLightningStore implementation."""

import asyncio
import json
import multiprocessing as mp
import sqlite3
import threading
import time
from typing import List

import pytest

from agentlightning.store.sqlite import SqliteLightningStore
from agentlightning.types import PromptTemplate, Resource, RolloutConfig, Span, TraceStatus

_process_store: SqliteLightningStore | None = None


def _process_update_attempt(queue: "mp.Queue[str]", rollout_id: str, attempt_id: str) -> None:
    store = _process_store
    if store is None:
        raise RuntimeError("process store not initialized")

    async def runner() -> None:
        await store.update_attempt(rollout_id, attempt_id, status="succeeded")

    asyncio.run(runner())
    queue.put("done")


def _make_span(rollout_id: str, attempt_id: str, sequence_id: int, name: str = "span") -> Span:
    return Span(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
        trace_id=f"{sequence_id:032x}",
        span_id=f"{sequence_id:016x}",
        parent_id=None,
        name=name,
        status=TraceStatus(status_code="OK"),
        attributes={},
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=None,
        parent=None,
        resource=Resource(attributes={}, schema_url=""),
    )


@pytest.mark.asyncio
async def test_schema_has_three_tables(sqlite_store: SqliteLightningStore) -> None:
    def reader(connection: sqlite3.Connection) -> List[str]:
        rows: list[sqlite3.Row] = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        return sorted([str(row["name"]) for row in rows])

    tables = await sqlite_store._execute_read(reader)  # type: ignore[attr-defined]
    assert tables == ["attempts", "rollouts", "spans"]


@pytest.mark.asyncio
async def test_enqueue_dequeue_fifo(sqlite_store: SqliteLightningStore) -> None:
    first = await sqlite_store.enqueue_rollout(input={"idx": 1})
    second = await sqlite_store.enqueue_rollout(input={"idx": 2})

    dequeued_first = await sqlite_store.dequeue_rollout()
    assert dequeued_first is not None
    dequeued_second = await sqlite_store.dequeue_rollout()
    assert dequeued_second is not None

    assert dequeued_first.rollout_id == first.rollout_id
    assert dequeued_second.rollout_id == second.rollout_id


@pytest.mark.asyncio
async def test_attempt_propagation(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "propagate"})
    await sqlite_store.update_attempt(
        attempted.rollout_id,
        attempted.attempt.attempt_id,
        status="succeeded",
    )

    stored = await sqlite_store.get_rollout_by_id(attempted.rollout_id)
    assert stored is not None
    assert stored.status == "succeeded"


@pytest.mark.asyncio
async def test_add_span_updates_status(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "span"})
    span = _make_span(
        attempted.rollout_id,
        attempted.attempt.attempt_id,
        sequence_id=1,
        name="running-span",
    )

    await sqlite_store.add_span(span)

    latest_attempt = await sqlite_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "running"

    rollout = await sqlite_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "running"


@pytest.mark.asyncio
async def test_span_json_storage(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "span-json"})
    span = _make_span(
        attempted.rollout_id,
        attempted.attempt.attempt_id,
        sequence_id=1,
        name="json-span",
    )
    span.attributes = {"binary": b"\x00\x01"}
    await sqlite_store.add_span(span)

    def reader(connection: sqlite3.Connection) -> str:
        row = connection.execute(
            "SELECT attributes_json FROM spans WHERE rollout_id=? AND attempt_id=?",
            (span.rollout_id, span.attempt_id),
        ).fetchone()
        assert row is not None
        encoded = row[0]
        assert isinstance(encoded, str)
        return encoded

    encoded = await sqlite_store._execute_read(reader)  # type: ignore[attr-defined]
    data = json.loads(encoded)
    assert data["binary"]["__type__"] == "bytes"
    fetched = await sqlite_store.query_spans(span.rollout_id)
    assert fetched[0].attributes["binary"] == b"\x00\x01"


@pytest.mark.asyncio
async def test_wait_for_rollout_completion(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "wait"})

    await sqlite_store.update_attempt(
        attempted.rollout_id,
        attempted.attempt.attempt_id,
        status="succeeded",
    )

    completed = await sqlite_store.wait_for_rollouts(rollout_ids=[attempted.rollout_id], timeout=0.1)
    assert completed
    assert completed[0].status == "succeeded"


@pytest.mark.asyncio
async def test_healthcheck_promotes_running_on_heartbeat(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "heartbeat"})

    await sqlite_store.update_attempt(
        attempted.rollout_id,
        attempted.attempt.attempt_id,
        last_heartbeat_time=time.time(),
    )

    rollout = await sqlite_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "running"

    latest_attempt = await sqlite_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "running"


@pytest.mark.asyncio
async def test_healthcheck_marks_timeout(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "timeout"})

    await sqlite_store.update_rollout(
        attempted.rollout_id,
        config=RolloutConfig(timeout_seconds=0.01, max_attempts=1, retry_condition=[]),
    )

    def writer(connection: sqlite3.Connection) -> None:
        connection.execute(
            "UPDATE attempts SET start_time=? WHERE attempt_id=?",
            (time.time() - 5, attempted.attempt.attempt_id),
        )

    await sqlite_store._execute_write(writer)  # type: ignore[attr-defined]

    rollout = await sqlite_store.get_rollout_by_id(attempted.rollout_id)
    assert rollout is not None
    assert rollout.status == "failed"

    latest_attempt = await sqlite_store.get_latest_attempt(attempted.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.status == "timeout"


@pytest.mark.asyncio
async def test_resources_roundtrip(sqlite_store: SqliteLightningStore) -> None:
    await sqlite_store.update_resources("resources-1", {"prompt": PromptTemplate(template="Hello", engine="f-string")})
    latest = await sqlite_store.get_latest_resources()
    assert latest is not None
    assert isinstance(latest.resources["prompt"], PromptTemplate)

    created = await sqlite_store.add_resources({"prompt": PromptTemplate(template="World", engine="f-string")})
    assert isinstance(created.resources["prompt"], PromptTemplate)


@pytest.mark.asyncio
async def test_thread_uses_separate_connection(sqlite_store: SqliteLightningStore) -> None:
    attempted = await sqlite_store.start_rollout(input={"task": "thread"})

    def worker() -> None:
        asyncio.run(
            sqlite_store.update_attempt(
                attempted.rollout_id,
                attempted.attempt.attempt_id,
                status="succeeded",
            )
        )

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    stored = await sqlite_store.get_rollout_by_id(attempted.rollout_id)
    assert stored is not None and stored.status == "succeeded"


@pytest.mark.asyncio
async def test_process_reopens_connection(sqlite_store: SqliteLightningStore) -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method not available")

    attempted = await sqlite_store.start_rollout(input={"task": "process"})
    global _process_store
    _process_store = sqlite_store

    ctx = mp.get_context("fork")
    queue: mp.Queue[str] = ctx.Queue()
    process = ctx.Process(
        target=_process_update_attempt,
        args=(queue, attempted.rollout_id, attempted.attempt.attempt_id),
    )
    process.start()
    process.join(timeout=5)
    assert process.exitcode == 0
    assert queue.get(timeout=1) == "done"

    stored = await sqlite_store.get_rollout_by_id(attempted.rollout_id)
    assert stored is not None and stored.status == "succeeded"

    _process_store = None
