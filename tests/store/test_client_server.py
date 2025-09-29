import asyncio
import socket
from typing import AsyncGenerator, Tuple

import pytest
import pytest_asyncio

from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.types import Resource as TraceResource
from agentlightning.tracer.types import Span as TraceSpan
from agentlightning.tracer.types import TraceStatus
from agentlightning.types import LLM, NamedResources, PromptTemplate


def _make_span(rollout_id: str, attempt_id: str, sequence_id: int, name: str) -> TraceSpan:
    return TraceSpan(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id,
        trace_id="0" * 32,
        span_id="1" * 16,
        parent_id=None,
        name=name,
        status=TraceStatus(status_code="OK"),
        attributes={"key": "value"},
        events=[],
        links=[],
        start_time=0.0,
        end_time=None,
        context=None,
        parent=None,
        resource=TraceResource(attributes={}, schema_url=""),
    )


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def sample_resources() -> NamedResources:
    return {
        "main_llm": LLM(
            endpoint="http://localhost:8080/v1/chat/completions",
            model="gpt-4o",
            sampling_parameters={"temperature": 0.2},
        ),
        "system_prompt": PromptTemplate(template="You are a helper.", engine="f-string"),
    }


@pytest_asyncio.fixture
async def client_server_setup() -> (
    AsyncGenerator[Tuple[LightningStoreServer, LightningStoreClient, InMemoryLightningStore], None]
):
    store = InMemoryLightningStore()
    host = "127.0.0.1"
    port = _get_free_port()
    server = LightningStoreServer(store=store, host=host, port=port)
    await server.start()
    client = LightningStoreClient(server_address=server.endpoint)
    try:
        yield server, client, store
    finally:
        await client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_client_server_end_to_end_coverage(client_server_setup, sample_resources, mock_readable_span):
    server, client, _ = client_server_setup
    assert server.endpoint.endswith(str(server.port))

    resources_id = "resources-end-to-end"
    resource_update = await client.update_resources(resources_id, sample_resources)
    assert resource_update.resources_id == resources_id

    latest_resources = await client.get_latest_resources()
    assert latest_resources is not None
    assert latest_resources.resources_id == resources_id

    resources_by_id = await client.get_resources_by_id(resources_id)
    assert resources_by_id is not None
    assert resources_by_id.resources == sample_resources

    sample_task = {"prompt": "Explain concurrency."}
    attempted = await client.add_rollout(
        sample_task, mode="train", resources_id=resources_id, metadata={"source": "client"}
    )
    assert attempted.attempt.sequence_id == 1
    queued = await client.enqueue_rollout(sample_task, mode="val", resources_id=resources_id, metadata={"priority": 1})
    dequeued = await client.dequeue_rollout()
    assert dequeued is not None

    await client.add_attempt(dequeued.rollout_id)
    rollouts = await client.query_rollouts()
    assert any(r.rollout_id == queued.rollout_id for r in rollouts)

    attempts = await client.query_attempts(dequeued.rollout_id)
    latest_attempt = await client.get_latest_attempt(dequeued.rollout_id)
    assert latest_attempt is not None
    assert latest_attempt.attempt_id in {a.attempt_id for a in attempts}

    span_seq = await client.get_next_span_sequence_id(dequeued.rollout_id, dequeued.attempt.attempt_id)
    assert span_seq >= 1

    base_span = _make_span(dequeued.rollout_id, dequeued.attempt.attempt_id, span_seq, name="client-span")
    await client.add_span(base_span)
    otel_span = await client.add_otel_span(dequeued.rollout_id, dequeued.attempt.attempt_id, mock_readable_span)
    assert otel_span.rollout_id == dequeued.rollout_id

    spans_all = await client.query_spans(dequeued.rollout_id)
    spans_latest = await client.query_spans(dequeued.rollout_id, attempt_id="latest")
    spans_specific = await client.query_spans(dequeued.rollout_id, attempt_id=dequeued.attempt.attempt_id)
    assert spans_all and spans_latest and spans_specific

    await client.update_attempt(
        dequeued.rollout_id,
        dequeued.attempt.attempt_id,
        status="running",
        worker_id="worker-a",
        metadata={"stage": "start"},
    )
    await client.update_rollout(dequeued.rollout_id, status="running", metadata={"checkpoint": True})

    wait_task = asyncio.create_task(client.wait_for_rollouts([dequeued.rollout_id], timeout=1.0))
    await asyncio.sleep(0.05)
    await client.update_rollout(dequeued.rollout_id, status="succeeded", metadata={"final": True})
    await client.update_attempt(dequeued.rollout_id, attempt_id="latest", status="succeeded")
    completed = await wait_task
    assert len(completed) == 1
    assert completed[0].status == "succeeded"

    succeeded_rollouts = await client.query_rollouts(status=["succeeded"])
    assert any(r.rollout_id == dequeued.rollout_id for r in succeeded_rollouts)

    await client.close()

    # Delegate methods on the server itself
    server_resource_update = await server.update_resources("server-only-resources", sample_resources)
    assert server_resource_update.resources_id == "server-only-resources"
    assert await server.get_latest_resources() is not None
    assert await server.get_resources_by_id("server-only-resources") is not None

    direct_attempted = await server.add_rollout(sample_task, mode="test", resources_id=resources_id)
    assert direct_attempted.attempt.sequence_id == 1
    direct_rollout = await server.enqueue_rollout({"prompt": "Server run."}, mode="test", resources_id=resources_id)
    direct_dequeued = await server.dequeue_rollout()
    assert direct_dequeued is not None

    await server.add_attempt(direct_rollout.rollout_id)
    server_rollouts = await server.query_rollouts()
    assert any(r.rollout_id == direct_rollout.rollout_id for r in server_rollouts)

    server_attempts = await server.query_attempts(direct_rollout.rollout_id)
    direct_latest_attempt = await server.get_latest_attempt(direct_rollout.rollout_id)
    assert direct_latest_attempt is not None
    assert direct_latest_attempt.attempt_id in {a.attempt_id for a in server_attempts}

    server_seq = await server.get_next_span_sequence_id(direct_rollout.rollout_id, direct_dequeued.attempt.attempt_id)
    server_span = _make_span(
        direct_rollout.rollout_id, direct_dequeued.attempt.attempt_id, server_seq, name="server-span"
    )
    await server.add_span(server_span)
    await server.add_otel_span(direct_rollout.rollout_id, direct_dequeued.attempt.attempt_id, mock_readable_span)

    await server.update_attempt(
        direct_rollout.rollout_id, direct_dequeued.attempt.attempt_id, status="running", metadata={"phase": "server"}
    )
    await server.update_rollout(direct_rollout.rollout_id, status="running")

    server_wait_task = asyncio.create_task(server.wait_for_rollouts([direct_rollout.rollout_id], timeout=1.0))
    await asyncio.sleep(0.05)
    await server.update_rollout(direct_rollout.rollout_id, status="succeeded")
    server_completed = await server_wait_task
    assert len(server_completed) == 1

    server_spans = await server.query_spans(direct_rollout.rollout_id)
    assert server_spans
    assert await server.query_spans(direct_rollout.rollout_id, attempt_id="latest")


@pytest.mark.asyncio
async def test_client_session_reuse_under_concurrency(client_server_setup):
    _, client, _ = client_server_setup

    sessions = await asyncio.gather(*[client._get_session() for _ in range(5)])
    assert len({id(session) for session in sessions}) == 1

    await client.close()
    reopened = await client._get_session()
    assert not reopened.closed


@pytest.mark.asyncio
async def test_concurrent_enqueue_and_dequeue(client_server_setup, sample_resources):
    _, client, _ = client_server_setup
    resources_id = "resources-concurrency"
    await client.update_resources(resources_id, sample_resources)

    enqueue_tasks = [
        client.enqueue_rollout({"prompt": f"Job {idx}"}, mode="train", resources_id=resources_id, metadata={"idx": idx})
        for idx in range(5)
    ]
    enqueued = await asyncio.gather(*enqueue_tasks)
    assert len({r.rollout_id for r in enqueued}) == len(enqueued)

    dequeue_tasks = [client.dequeue_rollout() for _ in range(len(enqueued))]
    dequeued = await asyncio.gather(*dequeue_tasks)
    assert all(item is not None for item in dequeued)
    assert len({item.rollout_id for item in dequeued if item is not None}) == len(enqueued)


@pytest.mark.asyncio
async def test_wait_for_rollouts_race_conditions(client_server_setup, sample_resources):
    _, client, _ = client_server_setup
    resources_id = "resources-wait"
    await client.update_resources(resources_id, sample_resources)

    rollout = await client.enqueue_rollout({"prompt": "Race"}, mode="train", resources_id=resources_id)
    dequeued = await client.dequeue_rollout()
    assert dequeued is not None

    wait_task = asyncio.create_task(client.wait_for_rollouts([rollout.rollout_id], timeout=1.0))
    await asyncio.sleep(0.05)
    await client.update_rollout(rollout.rollout_id, status="succeeded")
    await client.update_attempt(rollout.rollout_id, attempt_id="latest", status="succeeded")

    completed = await wait_task
    assert len(completed) == 1
    assert completed[0].rollout_id == rollout.rollout_id
