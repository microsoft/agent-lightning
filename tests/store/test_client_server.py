# Copyright (c) Microsoft. All rights reserved.

import asyncio
import multiprocessing
import sys
from typing import Any, AsyncGenerator, Dict, Tuple, cast
from unittest.mock import patch

import aiohttp
import pytest
import pytest_asyncio
from _pytest.monkeypatch import MonkeyPatch
from aiohttp import ClientConnectorError, ClientResponseError, ServerDisconnectedError
from opentelemetry.sdk.trace import ReadableSpan
from portpicker import pick_unused_port
from yarl import URL

from agentlightning.store.base import UNSET, LightningStore
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import LLM, OtelResource, PaginatedResult, PromptTemplate, RolloutConfig, Span, TraceStatus
from agentlightning.utils.server_launcher import LaunchMode, PythonServerLauncherArgs


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
        resource=OtelResource(attributes={}, schema_url=""),
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
async def server_client(
    store_fixture: LightningStore,
) -> AsyncGenerator[Tuple[LightningStoreServer, LightningStoreClient], None]:
    port = pick_unused_port()
    server = LightningStoreServer(store_fixture, "127.0.0.1", port)
    await server.start()
    client = LightningStoreClient(server.endpoint)
    try:
        yield server, client
    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_mp_server_does_not_work_with_inmemory_store() -> None:
    store = InMemoryLightningStore()
    with pytest.raises(ValueError, match="The store does not support zero-copy."):
        LightningStoreServer(store, "127.0.0.1", pick_unused_port(), launch_mode="mp")


@pytest.mark.asyncio
@pytest.mark.parametrize("launch_mode", ["asyncio", "thread"])
async def test_server_start_rejects_port_conflict(caplog: pytest.LogCaptureFixture, launch_mode: LaunchMode) -> None:
    """Ensure startup fails loudly when the port is already owned by another store."""
    store_a = InMemoryLightningStore()
    port = pick_unused_port()
    server_a = LightningStoreServer(store_a, "127.0.0.1", port)
    await server_a.start()

    store_b = InMemoryLightningStore()
    server_b = LightningStoreServer(store_b, "127.0.0.1", port, launch_mode=launch_mode)

    with pytest.raises(RuntimeError, match="did not start up within"):
        await server_b.start()
    assert "address already in use" in caplog.text

    await server_a.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("launch_mode", ["asyncio", "thread"])
async def test_run_forever_rejects_port_conflict(caplog: pytest.LogCaptureFixture, launch_mode: LaunchMode) -> None:
    """Ensure run_forever also reports port conflicts with the friendly message."""
    store_a = InMemoryLightningStore()
    port = pick_unused_port()
    server_a = LightningStoreServer(store_a, "127.0.0.1", port, launch_mode=launch_mode)
    await server_a.start()

    store_b = InMemoryLightningStore()
    server_b = LightningStoreServer(store_b, "127.0.0.1", port, launch_mode=launch_mode)

    with pytest.raises(RuntimeError, match="did not start up within"):
        await server_b.run_forever()
    assert "address already in use" in caplog.text

    await server_a.stop()


@pytest.mark.asyncio
async def test_server_accepts_custom_launcher_args(store_fixture: LightningStore) -> None:
    """Ensure providing launcher_args works end-to-end and is propagated to the launcher."""
    port = pick_unused_port()
    launcher_args = PythonServerLauncherArgs(
        host="127.0.0.1",
        port=port,
        launch_mode="asyncio",
        healthcheck_url="/v1/agl/health",
    )
    server = LightningStoreServer(store_fixture, launcher_args=launcher_args)
    assert server.launcher_args is launcher_args
    assert server.server_launcher.args is launcher_args
    assert server.server_launcher.health_url == f"http://127.0.0.1:{port}/v1/agl/health"

    await server.start()
    client = LightningStoreClient(server.endpoint)
    try:
        rollout = await client.start_rollout(input={"source": "launcher-args"})
        assert rollout.rollout_id
    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_add_resources_via_server(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    """Test that add_resources works correctly via server."""
    server, _ = server_client

    # Add resources using add_resources
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:8080/v1",
        model="test-model",
        sampling_parameters={"temperature": 0.7},
    )
    prompt = PromptTemplate(resource_type="prompt_template", template="Hello {name}!", engine="f-string")

    resources_update = await server.add_resources(cast(Any, {"main_llm": llm, "greeting": prompt}))

    # Verify resources_id was auto-generated
    assert resources_update.resources_id.startswith("rs-")
    assert len(resources_update.resources_id) == 15  # "rs-" + 12 char hash

    # Verify resources can be retrieved
    retrieved = await server.get_resources_by_id(resources_update.resources_id)
    assert retrieved is not None
    assert retrieved.resources_id == resources_update.resources_id
    assert isinstance(retrieved.resources["main_llm"], LLM)
    assert retrieved.resources["main_llm"].model == "test-model"

    assert isinstance(retrieved.resources["greeting"], PromptTemplate)
    assert retrieved.resources["greeting"].template == "Hello {name}!"

    # Verify it's set as latest
    latest = await server.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == resources_update.resources_id


@pytest.mark.asyncio
async def test_add_resources_via_client(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    """Test that add_resources works correctly via HTTP client."""
    from typing import cast

    from agentlightning.types import LLM

    _, client = server_client

    # Add resources using add_resources via HTTP
    llm = LLM(
        resource_type="llm",
        endpoint="http://localhost:9000/v1",
        model="client-model",
        sampling_parameters={"temperature": 0.5},
    )

    resources_update = await client.add_resources(cast(Any, {"client_llm": llm}))

    # Verify resources_id was auto-generated
    assert resources_update.resources_id.startswith("rs-")

    # Verify resources can be retrieved via client
    retrieved = await client.get_resources_by_id(resources_update.resources_id)
    assert retrieved is not None
    assert isinstance(retrieved.resources["client_llm"], LLM)
    assert retrieved.resources["client_llm"].model == "client-model"

    # Verify it's set as latest
    latest = await client.get_latest_resources()
    assert latest is not None
    assert latest.resources_id == resources_update.resources_id


@pytest.mark.asyncio
async def test_query_resources_history(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    """Server and client should return identical resource history ordering."""
    server, client = server_client

    server_history_empty = await server.query_resources()
    assert isinstance(server_history_empty, PaginatedResult)
    assert len(server_history_empty) == 0

    client_history_empty = await client.query_resources()
    assert isinstance(client_history_empty, PaginatedResult)
    assert len(client_history_empty) == 0

    first = await server.add_resources(
        cast(
            Any,
            {
                "llm": LLM(
                    resource_type="llm",
                    endpoint="http://localhost:8000",
                    model="hist-model-1",
                    sampling_parameters={},
                )
            },
        )
    )
    second = await server.update_resources(
        "manual-id",
        cast(
            Any,
            {"prompt": PromptTemplate(resource_type="prompt_template", template="Hi {user}", engine="f-string")},
        ),
    )

    server_history = await server.query_resources()
    client_history = await client.query_resources()

    expected_ids = [first.resources_id, second.resources_id]
    assert sorted([item.resources_id for item in server_history]) == sorted(expected_ids)
    assert sorted([item.resources_id for item in client_history]) == sorted(expected_ids)


@pytest.mark.asyncio
async def test_client_query_resources_filters_and_pagination(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    _, client = server_client

    alpha = PromptTemplate(resource_type="prompt_template", template="alpha", engine="jinja")
    beta = PromptTemplate(resource_type="prompt_template", template="beta", engine="jinja")

    await client.update_resources("manual-alpha", cast(Any, {"prompt": alpha}))
    await client.update_resources("manual-beta", cast(Any, {"prompt": beta}))

    contains_beta = await client.query_resources(resources_id_contains="beta")
    assert [item.resources_id for item in contains_beta] == ["manual-beta"]

    sorted_ids = sorted(["manual-alpha", "manual-beta"], reverse=True)
    paged = await client.query_resources(sort_by="resources_id", sort_order="desc", limit=1, offset=1)
    assert [item.resources_id for item in paged] == sorted_ids[1:2]


@pytest.mark.asyncio
async def test_client_server_end_to_end(
    server_client: Tuple[LightningStoreServer, LightningStoreClient], mock_readable_span: ReadableSpan
) -> None:
    server, client = server_client

    # Server delegate coverage -------------------------------------------------
    await server.update_resources("server-resources", {})
    assert await server.get_resources_by_id("server-resources") is not None
    assert await server.get_latest_resources() is not None

    server_start_config = RolloutConfig(timeout_seconds=8.5)
    attempted_server = await server.start_rollout(input={"origin": "server"}, config=server_start_config)
    assert attempted_server.config.timeout_seconds == 8.5

    server_queue_config = RolloutConfig(unresponsive_seconds=4.2, max_attempts=2)
    queued_rollout = await server.enqueue_rollout(input={"origin": "server-queue"}, config=server_queue_config)
    assert queued_rollout.config.unresponsive_seconds == 4.2
    server_worker_id = "server-worker"
    dequeued = await server.dequeue_rollout(worker_id=server_worker_id)
    server_worker_after_dequeue = await server.get_worker_by_id(server_worker_id)
    assert server_worker_after_dequeue is not None
    assert server_worker_after_dequeue.status == "idle"
    assert server_worker_after_dequeue.last_dequeue_time is not None
    dequeue_time = server_worker_after_dequeue.last_dequeue_time
    started_attempt = await server.start_attempt(queued_rollout.rollout_id)

    await server.query_rollouts()
    await server.query_attempts(queued_rollout.rollout_id)
    assert await server.get_latest_attempt(queued_rollout.rollout_id) is not None
    stored_server_rollout = await server.get_rollout_by_id(queued_rollout.rollout_id)
    assert stored_server_rollout is not None
    assert stored_server_rollout.config.unresponsive_seconds == 4.2

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
        worker_id=server_worker_id,
        metadata={"phase": "warmup"},
    )
    server_worker_busy = await server.get_worker_by_id(server_worker_id)
    assert server_worker_busy is not None
    assert server_worker_busy.status == "busy"
    assert server_worker_busy.current_rollout_id == queued_rollout.rollout_id
    assert server_worker_busy.current_attempt_id == started_attempt.attempt.attempt_id
    assert server_worker_busy.last_busy_time is not None
    assert server_worker_busy.last_busy_time >= dequeue_time

    await server.update_attempt(queued_rollout.rollout_id, "latest", status="succeeded")
    server_worker_idle = await server.get_worker_by_id(server_worker_id)
    assert server_worker_idle is not None
    assert server_worker_idle.status == "idle"
    assert server_worker_idle.current_rollout_id is None
    assert server_worker_idle.current_attempt_id is None
    assert server_worker_idle.last_idle_time is not None
    assert server_worker_idle.last_idle_time >= server_worker_busy.last_busy_time
    completed = await server.wait_for_rollouts(rollout_ids=[queued_rollout.rollout_id], timeout=0.1)
    assert completed and completed[0].status in {"succeeded", "failed", "cancelled"}

    # Client HTTP round trip ---------------------------------------------------
    resource_update = await client.update_resources("client-resources", {})
    assert resource_update.resources == {}
    assert await client.get_resources_by_id("client-resources") is not None
    assert await client.get_latest_resources() is not None

    client_start_config = RolloutConfig(timeout_seconds=3.0, retry_condition=["timeout"])
    attempted_client = await client.start_rollout(
        input={"origin": "client"},
        mode="train",
        config=client_start_config,
        metadata={"step": 0},
    )
    assert attempted_client.config.timeout_seconds == 3.0

    client_queue_config = RolloutConfig(unresponsive_seconds=6.0)
    enqueued = await client.enqueue_rollout(input={"origin": "client-queue"}, config=client_queue_config)
    assert enqueued.config.unresponsive_seconds == 6.0
    client_worker_id = "client-worker"
    dequeued_client = await client.dequeue_rollout(worker_id=client_worker_id)
    assert dequeued_client is not None
    client_worker_after_dequeue = await client.get_worker_by_id(client_worker_id)
    assert client_worker_after_dequeue is not None
    assert client_worker_after_dequeue.status == "idle"
    assert client_worker_after_dequeue.last_dequeue_time is not None
    client_dequeue_time = client_worker_after_dequeue.last_dequeue_time
    started_client_attempt = await client.start_attempt(dequeued_client.rollout_id)

    all_rollouts = await client.query_rollouts()
    assert any(r.rollout_id == enqueued.rollout_id for r in all_rollouts)
    assert await client.query_rollouts(rollout_ids=[enqueued.rollout_id])
    # Test that attempt is present in the rollout
    assert any(hasattr(r, "attempt") and r.attempt is not None for r in all_rollouts)  # type: ignore
    attempts = await client.query_attempts(dequeued_client.rollout_id)
    assert attempts
    assert await client.get_latest_attempt(dequeued_client.rollout_id) is not None
    stored_client_rollout = await client.get_rollout_by_id(dequeued_client.rollout_id)
    assert stored_client_rollout is not None
    assert stored_client_rollout.config.unresponsive_seconds == 6.0
    # Test that attempt is present in the rollout
    assert hasattr(stored_client_rollout, "attempt") and stored_client_rollout.attempt is not None  # type: ignore

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
        worker_id=client_worker_id,
        metadata={"info": "started"},
    )
    client_worker_busy = await client.get_worker_by_id(client_worker_id)
    assert client_worker_busy is not None
    assert client_worker_busy.status == "busy"
    assert client_worker_busy.current_rollout_id == dequeued_client.rollout_id
    assert client_worker_busy.current_attempt_id == started_client_attempt.attempt.attempt_id
    assert client_worker_busy.last_busy_time is not None
    assert client_worker_busy.last_busy_time >= client_dequeue_time

    await client.update_attempt(dequeued_client.rollout_id, "latest", status="succeeded")
    await client.update_rollout(dequeued_client.rollout_id, status="succeeded")
    client_worker_idle = await client.get_worker_by_id(client_worker_id)
    assert client_worker_idle is not None
    assert client_worker_idle.status == "idle"
    assert client_worker_idle.current_rollout_id is None
    assert client_worker_idle.current_attempt_id is None
    assert client_worker_idle.last_idle_time is not None
    assert client_worker_idle.last_idle_time >= client_worker_busy.last_busy_time

    wait_result = await client.wait_for_rollouts(rollout_ids=[dequeued_client.rollout_id], timeout=0.05)
    assert wait_result and wait_result[0].status == "succeeded"


@pytest.mark.asyncio
async def test_client_query_rollouts_filters_and_pagination(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    _, client = server_client

    rollouts = [await client.enqueue_rollout(input={"idx": idx}) for idx in range(3)]
    await client.update_rollout(rollout_id=rollouts[0].rollout_id, status="failed")

    failed = await client.query_rollouts(status_in=["failed"])
    assert [rollout.rollout_id for rollout in failed] == [rollouts[0].rollout_id]

    substring = rollouts[2].rollout_id[-4:]
    contains = await client.query_rollouts(rollout_id_contains=substring)
    assert any(rollout.rollout_id == rollouts[2].rollout_id for rollout in contains)

    sorted_ids = sorted([rollout.rollout_id for rollout in rollouts], reverse=True)
    paged = await client.query_rollouts(sort_by="rollout_id", sort_order="desc", limit=1, offset=1)
    assert [rollout.rollout_id for rollout in paged] == sorted_ids[1:2]


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
@pytest.mark.parametrize(
    "bad_payload",
    [
        {"status": None},
        {"config": None},
    ],
)
async def test_update_rollout_rejects_none_values(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
    bad_payload: Dict[str, Any],
) -> None:
    _, client = server_client

    attempted = await client.start_rollout(input={"payload": "bad-none"})
    with pytest.raises((ClientResponseError, AttributeError)) as exc_info:
        await client.update_rollout(attempted.rollout_id, **bad_payload)

    if isinstance(exc_info.value, ClientResponseError):
        assert exc_info.value.status == 400


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
async def test_update_worker_records_heartbeat(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    _, client = server_client

    first = await client.update_worker("runner-1", heartbeat_stats={"cpu": 0.4})
    assert first.status == "unknown"
    assert first.heartbeat_stats == {"cpu": 0.4}
    assert first.last_heartbeat_time is not None

    second = await client.update_worker("runner-1")
    assert second.last_heartbeat_time is not None
    assert second.last_heartbeat_time >= first.last_heartbeat_time
    assert second.heartbeat_stats == {"cpu": 0.4}


@pytest.mark.asyncio
async def test_update_worker_rejects_none_stats(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    _, client = server_client
    with pytest.raises(ClientResponseError) as exc_info:
        await client.update_worker("runner-err", heartbeat_stats=cast(Any, None))
    assert exc_info.value.status == 400


@pytest.mark.asyncio
async def test_worker_status_transitions_via_attempts(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    _, client = server_client

    await client.enqueue_rollout(input={"payload": "worker"})
    claimed = await client.dequeue_rollout(worker_id="runner-auto")
    assert claimed is not None

    await client.update_attempt(claimed.rollout_id, claimed.attempt.attempt_id, worker_id="runner-auto")
    busy = await client.get_worker_by_id("runner-auto")
    assert busy is not None
    assert busy.status == "busy"
    assert busy.current_rollout_id == claimed.rollout_id
    assert busy.current_attempt_id == claimed.attempt.attempt_id
    assert busy.last_dequeue_time is not None
    assert busy.last_busy_time is not None

    await client.update_attempt(claimed.rollout_id, claimed.attempt.attempt_id, status="succeeded")
    idle = await client.get_worker_by_id("runner-auto")
    assert idle is not None
    assert idle.status == "idle"
    assert idle.current_rollout_id is None
    assert idle.current_attempt_id is None


@pytest.mark.asyncio
async def test_client_query_workers_filters(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    _, client = server_client

    await client.update_worker("alpha-worker", heartbeat_stats={"cpu": 0.2})
    await client.update_worker("beta-worker", heartbeat_stats={"cpu": 0.8})

    busy_rollout = await client.start_rollout(input={"worker": "alpha"})
    await client.update_attempt(
        busy_rollout.rollout_id,
        busy_rollout.attempt.attempt_id,
        worker_id="alpha-worker",
        status="running",
    )

    busy_workers = await client.query_workers(status_in=["busy"])
    assert [worker.worker_id for worker in busy_workers] == ["alpha-worker"]

    contains_beta = await client.query_workers(worker_id_contains="beta")
    assert [worker.worker_id for worker in contains_beta] == ["beta-worker"]

    or_filtered = await client.query_workers(
        status_in=["busy"],
        worker_id_contains="beta",
        filter_logic="or",
        sort_by="worker_id",
        sort_order="asc",
    )
    assert [worker.worker_id for worker in or_filtered] == ["alpha-worker", "beta-worker"]


@pytest.mark.asyncio
async def test_get_worker_by_id(server_client: Tuple[LightningStoreServer, LightningStoreClient]) -> None:
    server, client = server_client

    await server.update_worker("runner-lookup", heartbeat_stats={"cpu": 0.3})

    server_worker = await server.get_worker_by_id("runner-lookup")
    assert server_worker is not None
    assert server_worker.worker_id == "runner-lookup"
    assert await server.get_worker_by_id("missing") is None

    client_worker = await client.get_worker_by_id("runner-lookup")
    assert client_worker is not None
    assert client_worker.worker_id == "runner-lookup"
    assert await client.get_worker_by_id("missing") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_payload",
    [
        {"last_heartbeat_time": None},
        {"metadata": None},
    ],
)
async def test_update_attempt_rejects_none_values(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
    bad_payload: Dict[str, Any],
) -> None:
    _, client = server_client

    attempted = await client.start_rollout(input={"payload": "bad-none-attempt"})
    with pytest.raises(ClientResponseError) as exc_info:
        await client.update_attempt(attempted.rollout_id, attempted.attempt.attempt_id, **bad_payload)

    assert exc_info.value.status == 400


@pytest.mark.asyncio
async def test_client_query_spans_filters_and_pagination(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    server, client = server_client

    attempted = await server.start_rollout(input={"span": "filters"})
    attempt_id = attempted.attempt.attempt_id

    spans = [
        _make_span(attempted.rollout_id, attempt_id, 1, "planner"),
        _make_span(attempted.rollout_id, attempt_id, 2, "reward"),
        _make_span(attempted.rollout_id, attempt_id, 3, "tool-call"),
    ]
    for span in spans:
        await server.add_span(span)

    planner = await client.query_spans(attempted.rollout_id, attempt_id=attempt_id, name_contains="plan")
    assert [span.name for span in planner] == ["planner"]

    or_filtered = await client.query_spans(
        attempted.rollout_id,
        attempt_id=attempt_id,
        span_id=spans[0].span_id,
        trace_id_contains=spans[2].trace_id[-4:],
        filter_logic="or",
    )
    assert {span.span_id for span in or_filtered} == {spans[0].span_id, spans[2].span_id}

    paged = await client.query_spans(
        attempted.rollout_id,
        attempt_id=attempt_id,
        sort_by="sequence_id",
        sort_order="desc",
        limit=1,
        offset=1,
    )
    assert [span.span_id for span in paged] == [spans[1].span_id]


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
    port = pick_unused_port()
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
    port = pick_unused_port()
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
@pytest.mark.flaky(reruns=3, reruns_delay=2)
async def test_retry_on_400_application_error(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that client retries on app-side 400 that becomes a 500 due to server exception handling."""
    server, client = server_client

    # Force app-side exception so server returns 400 via exception handler.
    call_count = {"n": 0}
    original = server.store.enqueue_rollout  # type: ignore

    async def boom(*args: Any, **kwargs: Any) -> Any:
        call_count["n"] += 1
        raise RuntimeError("synthetic app error")

    monkeypatch.setattr(server.store, "enqueue_rollout", boom, raising=True)

    with pytest.raises(ClientResponseError) as ei:
        await client.enqueue_rollout(input={"origin": "should-fail"})

    assert ei.value.status == 500
    assert call_count["n"] == 4

    # Restore original method
    monkeypatch.setattr(server.store, "enqueue_rollout", original, raising=True)


@pytest.mark.asyncio
async def test_no_retry_on_non408_4xx(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
    monkeypatch: MonkeyPatch,
) -> None:
    """Test that client does not retry on non-408 4xx errors such as 404."""
    _, client = server_client

    original_post = aiohttp.ClientSession.post
    calls = {"n": 0}

    def post_404(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any):
        if str(url).endswith("/rollouts/nonexistent"):
            calls["n"] += 1
            req_info = aiohttp.RequestInfo(
                url=URL(str(url)),
                method="POST",
                headers=cast(Any, {}),
                real_url=URL(str(url)),
            )
            raise ClientResponseError(
                request_info=req_info,
                history=(),
                status=404,
                message="not found",
            )
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
        if str(url).endswith("/rollouts"):
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
        if str(url).endswith("/queues/rollouts/enqueue") and not fired["once"]:
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
        if str(url).endswith("/rollouts"):
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


@pytest.mark.asyncio
async def test_retry_mechanism_with_custom_delays_and_health_recovery(
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Test the complete retry mechanism including:
    - Custom retry delays are respected
    - Health checks are performed between retries
    - Multiple failures are recovered from
    - Final success after health recovery
    """
    store = InMemoryLightningStore()
    port = pick_unused_port()
    server = LightningStoreServer(store, "127.0.0.1", port)
    await server.start()

    # Client with custom short delays for faster testing
    client = LightningStoreClient(
        server.endpoint,
        retry_delays=(0.01, 0.02),
        health_retry_delays=(0.01,),
    )

    try:
        original_post = aiohttp.ClientSession.post
        original_get = aiohttp.ClientSession.get
        counters = {"post_attempts": 0, "health_checks": 0}
        timestamps: list[float] = []

        def monitored_post(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith("/rollouts"):
                import time

                counters["post_attempts"] += 1
                timestamps.append(time.time())

                # Fail first 2 attempts with network error
                if counters["post_attempts"] <= 2:
                    raise ServerDisconnectedError(f"synthetic disconnect #{counters['post_attempts']}")

            return MockResponse(original_post(self, url, *args, **kwargs))

        def monitored_get(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith("/health"):
                counters["health_checks"] += 1
            return MockResponse(original_get(self, url, *args, **kwargs))

        monkeypatch.setattr(aiohttp.ClientSession, "post", monitored_post, raising=True)
        monkeypatch.setattr(aiohttp.ClientSession, "get", monitored_get, raising=True)

        # Execute request that will fail twice then succeed
        attempted = await client.start_rollout(input={"origin": "retry-test"})

        # Verify success
        assert attempted.rollout_id is not None

        # Verify retry behavior
        assert counters["post_attempts"] == 3, "Should make initial attempt + 2 retries"
        assert counters["health_checks"] >= 2, "Should check health after each network failure"

        # Verify delays were respected (timestamps should be spaced)
        if len(timestamps) >= 3:
            delay1 = timestamps[1] - timestamps[0]
            delay2 = timestamps[2] - timestamps[1]
            # First delay should be ~0.01s (first retry_delay)
            assert delay1 >= 0.01 and delay1 < 0.05, "First retry delay not respected"
            # Second delay should be ~0.02s (second retry_delay)
            assert delay2 >= 0.02 and delay2 < 0.06, "Second retry delay not respected"

    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_client_response_error_with_different_status_codes(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
    monkeypatch: MonkeyPatch,
) -> None:
    """
    Test that client handles different HTTP status codes correctly:
    - 400-499 (except 408): no retry, immediate failure
    - 408: retry (request timeout is transient)
    - 500-599: retry (server errors are transient)
    """
    _server, client = server_client

    original_post = aiohttp.ClientSession.post

    # Test 403 Forbidden - should NOT retry
    def post_403(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
        if str(url).endswith("/rollouts"):
            req_info = aiohttp.RequestInfo(
                url=URL(str(url)), method="POST", headers=cast(Any, {}), real_url=URL(str(url))
            )
            raise ClientResponseError(request_info=req_info, history=(), status=403, message="forbidden")
        return MockResponse(original_post(self, url, *args, **kwargs))

    monkeypatch.setattr(aiohttp.ClientSession, "post", post_403, raising=True)

    with pytest.raises(ClientResponseError) as exc_info:
        await client.start_rollout(input={"test": "403"})
    assert exc_info.value.status == 403

    # Test 503 Service Unavailable - should retry and succeed
    call_count = {"n": 0}

    def post_503_then_ok(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
        if str(url).endswith("/queues/rollouts/enqueue"):
            call_count["n"] += 1
            if call_count["n"] == 1:
                req_info = aiohttp.RequestInfo(
                    url=URL(str(url)), method="POST", headers=cast(Any, {}), real_url=URL(str(url))
                )
                raise ClientResponseError(request_info=req_info, history=(), status=503, message="service unavailable")
        return MockResponse(original_post(self, url, *args, **kwargs))

    monkeypatch.setattr(aiohttp.ClientSession, "post", post_503_then_ok, raising=True)

    result = await client.enqueue_rollout(input={"test": "503"})
    assert result.rollout_id is not None
    assert call_count["n"] == 2  # Failed once, then succeeded


@pytest.mark.asyncio
async def test_get_next_span_sequence_id_returns_proper_int(
    server_client: Tuple[LightningStoreServer, LightningStoreClient],
) -> None:
    """Test that get_next_span_sequence_id correctly converts JSON number to int."""
    _, client = server_client

    attempted = await client.start_rollout(input={"test": True})

    # First call should return 1
    seq_id_1 = await client.get_next_span_sequence_id(attempted.rollout_id, attempted.attempt.attempt_id)
    assert isinstance(seq_id_1, int)
    assert seq_id_1 == 1

    # Second call should return 2
    seq_id_2 = await client.get_next_span_sequence_id(attempted.rollout_id, attempted.attempt.attempt_id)
    assert isinstance(seq_id_2, int)
    assert seq_id_2 == 2

    # Verify monotonic increment
    assert seq_id_2 == seq_id_1 + 1


@pytest.mark.asyncio
async def test_empty_retry_delays_disable_retries(monkeypatch: MonkeyPatch) -> None:
    """
    When retry_delays is empty, the client should perform only the initial attempt
    and not retry on transient network errors.
    """
    store = InMemoryLightningStore()
    port = pick_unused_port()
    server = LightningStoreServer(
        store,
        launcher_args=PythonServerLauncherArgs(
            port=port,
            host="127.0.0.1",
            healthcheck_url=None,
            launch_mode="thread",
        ),
    )
    await server.start()

    # retry_delays=() disables retries; health checks still enabled
    client = LightningStoreClient(
        server.endpoint,
        retry_delays=(),
        health_retry_delays=(0.01,),
    )

    try:
        original_post = aiohttp.ClientSession.post
        original_get = aiohttp.ClientSession.get

        calls = {"post": 0, "health": 0}

        def failing_post(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith("/rollouts"):
                calls["post"] += 1
                # Always raise a transient error
                raise ServerDisconnectedError("synthetic disconnect for empty retry_delays")
            return MockResponse(original_post(self, url, *args, **kwargs))

        def ok_health_get(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith("/health"):
                calls["health"] += 1
            # delegate to the real get() and wrap in MockResponse so it stays an async CM
            return MockResponse(original_get(self, url, *args, **kwargs))

        monkeypatch.setattr(aiohttp.ClientSession, "post", failing_post, raising=True)
        monkeypatch.setattr(aiohttp.ClientSession, "get", ok_health_get, raising=True)

        with pytest.raises(ServerDisconnectedError):
            await client.start_rollout(input={"origin": "empty-retry-delays"})

        # Only the initial attempt should be made
        assert calls["post"] == 1
        # Health should be probed at least once
        assert calls["health"] >= 1
    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_empty_health_retry_delays_skip_health_checks(monkeypatch: MonkeyPatch) -> None:
    """
    When health_retry_delays is empty, _wait_until_healthy should not perform any
    /health probes, but retries governed by retry_delays should still occur.
    """
    store = InMemoryLightningStore()
    port = pick_unused_port()
    server = LightningStoreServer(
        store,
        launcher_args=PythonServerLauncherArgs(
            port=port,
            host="127.0.0.1",
            healthcheck_url=None,
            launch_mode="thread",
        ),
    )
    await server.start()

    # health_retry_delays=() disables health probes; still allow one retry
    client = LightningStoreClient(
        server.endpoint,
        retry_delays=(0.01,),
        health_retry_delays=(),
    )

    try:
        original_post = aiohttp.ClientSession.post
        original_get = aiohttp.ClientSession.get

        calls = {"post": 0, "health": 0}

        def flaky_post(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith("/rollouts"):
                calls["post"] += 1
                # First call fails, second succeeds
                if calls["post"] == 1:
                    raise ServerDisconnectedError("synthetic disconnect for empty health_retry_delays")
            return MockResponse(original_post(self, url, *args, **kwargs))

        def counting_health_get(self: aiohttp.ClientSession, url: Any, *args: Any, **kwargs: Any) -> MockResponse:
            if str(url).endswith("/health"):
                calls["health"] += 1
            return MockResponse(original_get(self, url, *args, **kwargs))

        monkeypatch.setattr(aiohttp.ClientSession, "post", flaky_post, raising=True)
        monkeypatch.setattr(aiohttp.ClientSession, "get", counting_health_get, raising=True)

        # Should succeed after one retry, without ever calling /health
        attempted = await client.start_rollout(input={"origin": "empty-health-delays"})
        assert attempted.rollout_id

        # One failure + one success
        assert calls["post"] == 2
        # No health checks should have been performed
        assert calls["health"] == 0
    finally:
        await client.close()
        await server.stop()
