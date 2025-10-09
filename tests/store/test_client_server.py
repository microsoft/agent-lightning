# Copyright (c) Microsoft. All rights reserved.

import asyncio
import contextlib
import multiprocessing
import socket
import sys
from typing import Any, AsyncGenerator, Tuple, cast
from unittest.mock import patch

import aiohttp
import pytest
import pytest_asyncio
from _pytest.monkeypatch import MonkeyPatch
from aiohttp import ClientConnectorError, ClientResponseError, ServerDisconnectedError
from opentelemetry.sdk.trace import ReadableSpan
from yarl import URL

from agentlightning.store.base import UNSET
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import Resource, Span, TraceStatus


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


class MockResponse:
    """Wrapper that passes through to the original aiohttp context manager."""

    def __init__(self, context_manager: Any):
        self._cm = context_manager

    async def __aenter__(self) -> aiohttp.ClientResponse:
        return await self._cm.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self._cm.__aexit__(exc_type, exc_val, exc_tb)


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


@pytest.mark.asyncio
async def test_subprocess_operations_sync_via_http_automatically() -> None:
    """
    Test that LightningStoreServer automatically uses HTTP client in subprocesses.

    When LightningStoreServer is passed to a subprocess, it detects it's in a different
    process (via PID tracking) and automatically delegates to an HTTP client instead of
    the local store. This ensures operations in the subprocess are reflected in the
    main process via the HTTP server.
    """
    store = InMemoryLightningStore()
    port = _get_free_port()
    server = LightningStoreServer(store, "127.0.0.1", port)
    await server.start()

    try:
        # Record initial state
        initial_rollouts = await store.query_rollouts()
        initial_count = len(initial_rollouts)

        def subprocess_work(server_obj: LightningStoreServer) -> None:
            """Subprocess that performs operations via the server object."""

            async def do_work() -> None:
                # The server detects we're in a different process and automatically
                # uses HTTP client to communicate with the main process server
                await server_obj.enqueue_rollout(input={"origin": "subprocess"})

            asyncio.run(do_work())

        # Spawn a subprocess to perform operations
        ctx = multiprocessing.get_context()
        process = ctx.Process(target=subprocess_work, args=(server,))
        process.start()
        await asyncio.to_thread(process.join, timeout=5.0)

        assert process.exitcode == 0

        # Allow time for HTTP request to complete
        await asyncio.sleep(0.2)

        # Subprocess operations ARE reflected in main process store
        # because the server automatically used HTTP client in the subprocess
        main_process_rollouts = await store.query_rollouts()
        assert len(main_process_rollouts) == initial_count + 1, (
            "Subprocess operations should be reflected in main process store " "via automatic HTTP client delegation"
        )

    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_subprocess_client_operations_work_but_direct_store_access_fails() -> None:
    """
    Demonstrate that:
    1. Client operations via HTTP work correctly (data persists in main process)
    2. Direct store access in subprocess does NOT work (data isolated to subprocess)
    """
    store = InMemoryLightningStore()
    port = _get_free_port()
    server = LightningStoreServer(store, "127.0.0.1", port)
    await server.start()

    try:
        initial_rollouts = await store.query_rollouts()
        initial_count = len(initial_rollouts)

        def subprocess_client_work(endpoint: str) -> None:
            """Subprocess using HTTP client - this WORKS."""

            async def do_work() -> None:
                client = LightningStoreClient(endpoint)
                try:
                    await client.enqueue_rollout(input={"origin": "subprocess-client"})
                except Exception as e:
                    print(f"Client subprocess error: {e}", file=sys.stderr)
                    raise
                finally:
                    await client.close()

            asyncio.run(do_work())

        def subprocess_direct_store_work(server_obj: LightningStoreServer) -> None:
            """Subprocess using direct store access - this does NOT work."""

            async def do_work() -> None:
                # This operates on the subprocess's copy of the store
                await server_obj.enqueue_rollout(input={"origin": "subprocess-direct"})

            asyncio.run(do_work())

        # Test 1: Client operations via HTTP - should work
        ctx = multiprocessing.get_context()
        client_process = ctx.Process(target=subprocess_client_work, args=(server.endpoint,))
        client_process.start()
        await asyncio.to_thread(client_process.join, timeout=5.0)  # Add timeout

        # Handle timeout case
        if client_process.is_alive():
            client_process.terminate()
            client_process.join(timeout=1.0)
            pytest.fail("Client subprocess hung and had to be terminated")

        assert client_process.exitcode == 0, f"Client subprocess failed with exit code {client_process.exitcode}"

        await asyncio.sleep(0.2)
        after_client = await store.query_rollouts()
        # Client operations WORK - the rollout is in the main process store
        assert len(after_client) == initial_count + 1

        # Test 2: Server object in subprocess - ALSO works now (auto-delegates to HTTP)
        direct_process = ctx.Process(target=subprocess_direct_store_work, args=(server,))
        direct_process.start()
        await asyncio.to_thread(direct_process.join, timeout=5.0)

        # Handle timeout case
        if direct_process.is_alive():
            direct_process.terminate()
            direct_process.join(timeout=1.0)
            pytest.fail("Server subprocess hung and had to be terminated")

        assert direct_process.exitcode == 0, f"Server subprocess failed with exit code {direct_process.exitcode}"

        await asyncio.sleep(0.2)
        after_direct = await store.query_rollouts()
        # With the fix: server object in subprocess ALSO works via auto HTTP delegation
        # Both rollouts (client + server) should be in the store
        assert (
            len(after_direct) == initial_count + 2
        ), "Both explicit client and server object operations should work via HTTP"

    finally:
        await server.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,endpoint,make_app_error",
    [
        (400, "/enqueue_rollout", True),  # server-marked app error -> 400 -> no retry
        (404, "/update_rollout", False),  # non-408 4xx -> no retry
    ],
)
async def test_no_retry_on_4xx_application_and_non408(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
    monkeypatch: MonkeyPatch,
    status: int,
    endpoint: str,
    make_app_error: bool,
) -> None:
    server, client = server_client

    if make_app_error:
        # Force app-side exception so server returns 400 via exception handler.
        call_count = {"n": 0}
        original = server.store.enqueue_rollout

        async def boom(*args: Any, **kwargs: Any) -> Any:
            call_count["n"] += 1
            raise RuntimeError("synthetic app error")

        monkeypatch.setattr(server.store, "enqueue_rollout", boom, raising=True)

        with pytest.raises(ClientResponseError) as ei:
            await client.enqueue_rollout(input={"origin": "should-fail"})
        assert ei.value.status == 400
        assert call_count["n"] == 1

        monkeypatch.setattr(server.store, "enqueue_rollout", original, raising=True)
    else:
        # Raise 404 once for /update_rollout; client must not retry.
        original_post = aiohttp.ClientSession.post
        calls = {"n": 0}

        def post_404(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith(endpoint):
                calls["n"] += 1
                req_info = aiohttp.RequestInfo(
                    url=URL(str(url)), method="POST", headers=cast(Any, {}), real_url=URL(str(url))
                )
                raise ClientResponseError(request_info=req_info, history=(), status=status, message="not found")
            return MockResponse(original_post(self, url, *args, **kwargs))

        monkeypatch.setattr(aiohttp.ClientSession, "post", post_404, raising=True)

        with pytest.raises(ClientResponseError) as ei:
            await client.update_rollout("nonexistent", status="running")
        assert ei.value.status == 404
        assert calls["n"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("exc_cls", [ServerDisconnectedError, asyncio.TimeoutError])
async def test_retry_on_transient_network_errors_then_success(
    server_client: Tuple[LightningStoreServer, LightningStoreClient], monkeypatch: MonkeyPatch, exc_cls: type[Exception]
) -> None:
    _server, client = server_client

    original_post = aiohttp.ClientSession.post
    counters = {"post_calls": 0}

    def flaky_post(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
        if str(url).endswith("/start_rollout"):
            if counters["post_calls"] == 0:
                counters["post_calls"] += 1
                # raise chosen transient network exception
                raise exc_cls()
        return MockResponse(original_post(self, url, *args, **kwargs))

    monkeypatch.setattr(aiohttp.ClientSession, "post", flaky_post, raising=True)

    attempted = await client.start_rollout(input={"origin": f"retry-ok-{exc_cls.__name__}"})
    assert attempted.rollout_id
    assert counters["post_calls"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 408])
async def test_retry_on_transient_http_status_then_success(
    server_client: Tuple[LightningStoreServer, LightningStoreClient], monkeypatch: MonkeyPatch, status: int
) -> None:
    _server, client = server_client

    original_post = aiohttp.ClientSession.post
    fired = {"once": False}

    def post_then_ok(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
        if str(url).endswith("/enqueue_rollout") and not fired["once"]:
            fired["once"] = True
            req_info = aiohttp.RequestInfo(
                url=URL(str(url)), method="POST", headers=cast(Any, {}), real_url=URL(str(url))
            )
            raise ClientResponseError(request_info=req_info, history=(), status=status, message="transient")
        return MockResponse(original_post(self, url, *args, **kwargs))

    monkeypatch.setattr(aiohttp.ClientSession, "post", post_then_ok, raising=True)

    res = await client.enqueue_rollout(input={"origin": f"after-{status}"})
    assert res.rollout_id
    assert fired["once"] is True


@pytest.mark.asyncio
async def test_unhealthy_health_probe_stops_retries(
    server_client: Tuple[LightningStoreServer, LightningStoreClient], monkeypatch: MonkeyPatch
) -> None:
    _server, client = server_client

    original_post = aiohttp.ClientSession.post
    original_get = aiohttp.ClientSession.get
    post_calls = {"n": 0}

    def failing_post(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
        if str(url).endswith("/start_rollout"):
            post_calls["n"] += 1
            raise ServerDisconnectedError("synthetic disconnect")
        return MockResponse(original_post(self, url, *args, **kwargs))

    def failing_health_get(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
        if str(url).endswith("/health"):
            # health never becomes available
            raise ClientConnectorError(connection_key=None, os_error=None)  # type: ignore
        return MockResponse(original_get(self, url, *args, **kwargs))

    monkeypatch.setattr(aiohttp.ClientSession, "post", failing_post, raising=True)
    monkeypatch.setattr(aiohttp.ClientSession, "get", failing_health_get, raising=True)

    with pytest.raises(ServerDisconnectedError):
        await client.start_rollout(input={"origin": "unhealthy"})
    # Only the initial attempt, since health checks fail
    assert post_calls["n"] == 1


@pytest.mark.asyncio
async def test_wait_for_rollouts_timeout_guard_raises(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    _, client = server_client
    with pytest.raises(ValueError):
        await client.wait_for_rollouts(rollout_ids=["dummy"], timeout=0.2)
