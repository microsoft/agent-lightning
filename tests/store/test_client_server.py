# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import socket
from typing import AsyncGenerator, Tuple
from unittest.mock import patch

import pytest
import pytest_asyncio
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store.base import UNSET
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer import Span
from agentlightning.tracer.types import Resource, TraceStatus


def _get_free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_span(rollout_id: str, attempt_id: str, sequence_id: int, name: str) -> Span:
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


@pytest_asyncio.fixture
async def server_client() -> AsyncGenerator[Tuple[LightningStoreServer, LightningStoreClient], None]:
    store = InMemoryLightningStore()
    port = _get_free_port()
    server = LightningStoreServer(store, "127.0.0.1", port)
    await server.start()
    client = LightningStoreClient(server.endpoint)
    try:
        yield server, client
    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_client_server_end_to_end(
    server_client: Tuple[LightningStoreServer, LightningStoreClient], mock_readable_span: ReadableSpan
) -> None:
    server, client = server_client

    # Server delegate coverage -------------------------------------------------
    await server.update_resources("server-resources", {})
    assert await server.get_resources_by_id("server-resources") is not None
    assert await server.get_latest_resources() is not None

    await server.start_rollout(input={"origin": "server"})
    queued_rollout = await server.enqueue_rollout(input={"origin": "server-queue"})
    dequeued = await server.dequeue_rollout()
    started_attempt = await server.start_attempt(queued_rollout.rollout_id)

    await server.query_rollouts()
    await server.query_attempts(queued_rollout.rollout_id)
    assert await server.get_latest_attempt(queued_rollout.rollout_id) is not None
    assert await server.get_rollout_by_id(queued_rollout.rollout_id) is not None

    assert dequeued is not None

    server_span = _make_span(dequeued.rollout_id, dequeued.attempt.attempt_id, 0, "server-span")
    await server.add_span(server_span)
    assert await server.get_next_span_sequence_id(dequeued.rollout_id, dequeued.attempt.attempt_id) == 1

    with patch("agentlightning.store.client_server.Span.from_opentelemetry", autospec=True) as mocked:
        mocked.side_effect = lambda readable, rollout_id, attempt_id, sequence_id: _make_span(  # pyright: ignore[reportUnknownLambdaType]
            rollout_id,  # pyright: ignore[reportUnknownArgumentType]
            attempt_id,  # pyright: ignore[reportUnknownArgumentType]
            sequence_id,  # pyright: ignore[reportUnknownArgumentType]
            f"server-otel-{sequence_id}",  # pyright: ignore[reportUnknownArgumentType]
        )
        await server.add_otel_span(dequeued.rollout_id, dequeued.attempt.attempt_id, mock_readable_span)

    await server.query_spans(dequeued.rollout_id)
    await server.update_rollout(queued_rollout.rollout_id, status="running")
    await server.update_attempt(
        queued_rollout.rollout_id,
        started_attempt.attempt.attempt_id,
        status="running",
        worker_id="server-worker",
        metadata={"phase": "warmup"},
    )
    await server.update_attempt(queued_rollout.rollout_id, "latest", status="succeeded")
    completed = await server.wait_for_rollouts(rollout_ids=[queued_rollout.rollout_id], timeout=0.1)
    assert completed and completed[0].status in {"succeeded", "failed", "cancelled"}

    # Client HTTP round trip ---------------------------------------------------
    resource_update = await client.update_resources("client-resources", {})
    assert resource_update.resources == {}
    assert await client.get_resources_by_id("client-resources") is not None
    assert await client.get_latest_resources() is not None

    _attempted = await client.start_rollout(input={"origin": "client"}, mode="train", metadata={"step": 0})
    enqueued = await client.enqueue_rollout(input={"origin": "client-queue"})
    dequeued_client = await client.dequeue_rollout()
    assert dequeued_client is not None
    started_client_attempt = await client.start_attempt(dequeued_client.rollout_id)

    all_rollouts = await client.query_rollouts()
    assert any(r.rollout_id == enqueued.rollout_id for r in all_rollouts)
    assert await client.query_rollouts(rollout_ids=[enqueued.rollout_id])
    attempts = await client.query_attempts(dequeued_client.rollout_id)
    assert attempts
    assert await client.get_latest_attempt(dequeued_client.rollout_id) is not None
    assert await client.get_rollout_by_id(dequeued_client.rollout_id) is not None

    client_span = _make_span(dequeued_client.rollout_id, dequeued_client.attempt.attempt_id, 101, "client-span")
    stored_span = await client.add_span(client_span)
    assert stored_span.name == "client-span"
    assert await client.get_next_span_sequence_id(dequeued_client.rollout_id, dequeued_client.attempt.attempt_id) == 102

    with patch("agentlightning.store.client_server.Span.from_opentelemetry", autospec=True) as mocked:
        mocked.side_effect = lambda readable, rollout_id, attempt_id, sequence_id: _make_span(  # pyright: ignore[reportUnknownLambdaType]
            rollout_id,  # pyright: ignore[reportUnknownArgumentType]
            attempt_id,  # pyright: ignore[reportUnknownArgumentType]
            sequence_id,  # pyright: ignore[reportUnknownArgumentType]
            f"client-otel-{sequence_id}",
        )
        await client.add_otel_span(dequeued_client.rollout_id, dequeued_client.attempt.attempt_id, mock_readable_span)

    spans = await client.query_spans(dequeued_client.rollout_id)
    assert spans

    await client.update_rollout(dequeued_client.rollout_id, mode="val", metadata={"step": 1})
    await client.update_attempt(
        dequeued_client.rollout_id,
        started_client_attempt.attempt.attempt_id,
        worker_id="client-worker",
        metadata={"info": "started"},
    )
    await client.update_attempt(dequeued_client.rollout_id, "latest", status="succeeded")
    await client.update_rollout(dequeued_client.rollout_id, status="succeeded")

    wait_result = await client.wait_for_rollouts(rollout_ids=[dequeued_client.rollout_id], timeout=0.05)
    assert wait_result and wait_result[0].status == "succeeded"


@pytest.mark.asyncio
async def test_update_rollout_none_vs_unset(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    _, client = server_client

    attempted = await client.start_rollout(input={"payload": True}, metadata={"keep": True})
    rollout_id = attempted.rollout_id

    await client.update_rollout(rollout_id, mode="train", metadata={"extra": 1})
    updated = await client.get_rollout_by_id(rollout_id)

    assert updated is not None
    assert updated.mode == "train"
    assert updated.metadata is not None
    assert updated.metadata["extra"] == 1

    await client.update_rollout(rollout_id, mode=None, metadata={"extra1": 2})
    cleared = await client.get_rollout_by_id(rollout_id)
    assert cleared is not None
    assert cleared.mode is None
    assert cleared.metadata is not None
    assert cleared.metadata == {"extra1": 2}

    await client.update_rollout(rollout_id, mode=UNSET, metadata=UNSET, status="running")
    preserved = await client.get_rollout_by_id(rollout_id)
    assert preserved is not None
    assert preserved.mode is None
    assert preserved.metadata == {"extra1": 2}
    assert preserved.status == "running"


@pytest.mark.asyncio
async def test_update_attempt_none_vs_unset(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    _, client = server_client

    attempted = await client.start_rollout(input={"payload": True})
    rollout_id = attempted.rollout_id
    attempt_id = attempted.attempt.attempt_id

    await client.update_attempt(rollout_id, attempt_id, worker_id="worker-1", metadata={"stage": "init"})
    initial = await client.get_latest_attempt(rollout_id)
    assert initial is not None
    assert initial.worker_id == "worker-1"
    assert initial.metadata is not None
    assert initial.metadata["stage"] == "init"

    await client.update_attempt(rollout_id, "latest", worker_id="", metadata={})
    cleared = await client.get_latest_attempt(rollout_id)
    assert cleared is not None
    assert cleared.worker_id == ""
    assert cleared.metadata == {}

    await client.update_attempt(rollout_id, "latest", status="running", worker_id=UNSET, metadata=UNSET)
    preserved = await client.get_latest_attempt(rollout_id)
    assert preserved is not None
    assert preserved.worker_id == ""
    assert preserved.metadata == {}
    assert preserved.status == "running"


@pytest.mark.asyncio
async def test_concurrent_add_otel_span_sequence_ids_unique(
    server_client: Tuple[LightningStoreServer, LightningStoreClient], mock_readable_span: ReadableSpan
) -> None:
    _, client = server_client

    attempted = await client.start_rollout(input={"payload": True})
    rollout_id = attempted.rollout_id
    attempt_id = attempted.attempt.attempt_id

    def _build_concurrent_span(readable: ReadableSpan, rollout_id: str, attempt_id: str, sequence_id: int) -> Span:
        return _make_span(rollout_id, attempt_id, sequence_id, f"concurrent-{sequence_id}")

    with patch("agentlightning.store.client_server.Span.from_opentelemetry", autospec=True) as mocked:
        mocked.side_effect = _build_concurrent_span
        spans = await asyncio.gather(
            *[client.add_otel_span(rollout_id, attempt_id, mock_readable_span) for _ in range(20)]
        )
    sequence_ids = [span.sequence_id for span in spans]
    assert len(set(sequence_ids)) == 20
    assert set(sequence_ids) == set(range(1, 21))

    stored_spans = await client.query_spans(rollout_id, attempt_id="latest")
    assert len(stored_spans) >= 2
