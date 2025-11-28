# Copyright (c) Microsoft. All rights reserved.

"""
Tests for RESTful API pagination, sorting, and filtering functionality.

Test categories:
- Rollouts pagination, sorting, and filtering
- Attempts pagination and sorting
- Resources pagination and sorting
- Spans pagination, sorting, and filtering
"""

import asyncio
import contextlib
from typing import AsyncGenerator, List, Tuple

import aiohttp
import pytest
import pytest_asyncio
from portpicker import pick_unused_port

from agentlightning.store import LightningStore
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.types import (
    LLM,
    AttemptedRollout,
    OtelResource,
    PaginatedResult,
    PromptTemplate,
    Rollout,
    Span,
    TraceStatus,
)


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
        start_time=1.0,
        end_time=2.0,
        context=None,
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


@contextlib.asynccontextmanager
async def _run_server_with_cors(cors_origins: List[str] | str | None = None):
    store = InMemoryLightningStore()
    port = pick_unused_port()
    server = LightningStoreServer(store, "127.0.0.1", port, cors_allow_origins=cors_origins)
    await server.start()
    session = aiohttp.ClientSession()
    try:
        yield server, session
    finally:
        await session.close()
        await server.stop()


@pytest_asyncio.fixture
async def server_client(
    store_fixture: LightningStore,
) -> AsyncGenerator[Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str], None]:
    port = pick_unused_port()
    server = LightningStoreServer(store_fixture, "127.0.0.1", port)
    await server.start()
    client = LightningStoreClient(server.endpoint)
    session = aiohttp.ClientSession()
    # Get the full API endpoint with /v1/agl prefix
    api_endpoint = client.server_address
    try:
        yield server, client, session, api_endpoint
    finally:
        await session.close()
        await client.close()
        await server.stop()


# CORS configuration tests


@pytest.mark.asyncio
async def test_cors_allows_specific_origin() -> None:
    async with _run_server_with_cors(cors_origins=["http://localhost:3000"]) as (server, session):
        url = f"{server.endpoint}/v1/agl/health"
        origin = "http://localhost:3000"
        async with session.get(url, headers={"Origin": origin}) as resp:
            assert resp.status == 200
            assert resp.headers.get("access-control-allow-origin") == origin
            assert resp.headers.get("access-control-allow-credentials") == "true"

        async with session.options(
            url,
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "GET",
            },
        ) as resp:
            assert resp.status == 200
            assert resp.headers.get("access-control-allow-origin") == origin
            allow_methods = resp.headers.get("access-control-allow-methods") or ""
            assert "GET" in {method.strip() for method in allow_methods.split(",") if method}


@pytest.mark.asyncio
async def test_cors_disallows_unconfigured_origin() -> None:
    async with _run_server_with_cors(cors_origins=["http://localhost:3000"]) as (server, session):
        url = f"{server.endpoint}/v1/agl/health"
        async with session.get(url, headers={"Origin": "http://malicious.example"}) as resp:
            assert resp.status == 200
            assert "access-control-allow-origin" not in resp.headers


@pytest.mark.asyncio
async def test_cors_allows_wildcard_origin() -> None:
    async with _run_server_with_cors(cors_origins="*") as (server, session):
        url = f"{server.endpoint}/v1/agl/health"
        origin = "https://wildcard.example"
        async with session.get(url, headers={"Origin": origin}) as resp:
            assert resp.status == 200
            allow_origin = resp.headers.get("access-control-allow-origin")
            assert allow_origin in {origin, "*"}
            allow_credentials = resp.headers.get("access-control-allow-credentials")
            if allow_credentials is not None:
                assert allow_credentials == "true"


# Rollouts Pagination, Sorting, and Filtering Tests


@pytest.mark.asyncio
async def test_rollouts_pagination_basic(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test basic pagination for rollouts endpoint."""
    server, _client, session, api_endpoint = server_client

    # Create 10 rollouts
    for i in range(10):
        await server.enqueue_rollout(input={"index": i})

    # Get first page with limit=3
    async with session.get(f"{api_endpoint}/rollouts", params={"limit": 3, "offset": 0}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 10
        assert data["limit"] == 3
        assert data["offset"] == 0
        assert len(data["items"]) == 3

    # Get second page
    async with session.get(f"{api_endpoint}/rollouts", params={"limit": 3, "offset": 3}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 10
        assert data["limit"] == 3
        assert data["offset"] == 3
        assert len(data["items"]) == 3


@pytest.mark.asyncio
async def test_rollouts_pagination_disabled(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test pagination can be disabled with limit=-1."""
    server, _client, session, api_endpoint = server_client

    # Create 15 rollouts
    for i in range(15):
        await server.enqueue_rollout(input={"index": i})

    # Get all rollouts with limit=-1
    async with session.get(f"{api_endpoint}/rollouts", params={"limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 15
        assert data["limit"] == -1
        assert len(data["items"]) == 15


@pytest.mark.asyncio
async def test_rollouts_sorting_by_start_time(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting rollouts by start_time."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts with small delays
    rollouts: List[Rollout] = []
    for i in range(5):
        r = await server.enqueue_rollout(input={"index": i})
        rollouts.append(r)
        await asyncio.sleep(0.01)

    # Sort ascending by start_time
    async with session.get(
        f"{api_endpoint}/rollouts", params={"sort_by": "start_time", "sort_order": "asc", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert len(items) == 5
        # Should be in ascending order
        for i in range(len(items) - 1):
            assert items[i]["start_time"] <= items[i + 1]["start_time"]

    # Sort descending by start_time (default)
    async with session.get(
        f"{api_endpoint}/rollouts", params={"sort_by": "start_time", "sort_order": "desc", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        # Should be in descending order
        for i in range(len(items) - 1):
            assert items[i]["start_time"] >= items[i + 1]["start_time"]


@pytest.mark.asyncio
async def test_rollouts_filter_by_status(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test filtering rollouts by status."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts with different statuses
    r1 = await server.enqueue_rollout(input={"id": 1})
    r2 = await server.enqueue_rollout(input={"id": 2})
    _r3 = await server.enqueue_rollout(input={"id": 3})

    await server.update_rollout(rollout_id=r1.rollout_id, status="succeeded")
    await server.update_rollout(rollout_id=r2.rollout_id, status="failed")
    # r3 remains queuing

    # Filter by single status
    async with session.get(f"{api_endpoint}/rollouts", params={"status_in": ["succeeded"], "limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 1
        assert data["items"][0]["rollout_id"] == r1.rollout_id

    # Filter by multiple statuses
    async with session.get(
        f"{api_endpoint}/rollouts", params={"status_in": ["succeeded", "failed"], "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 2
        rollout_ids = {item["rollout_id"] for item in data["items"]}
        assert rollout_ids == {r1.rollout_id, r2.rollout_id}


@pytest.mark.asyncio
async def test_rollouts_filter_by_rollout_id_in(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test filtering rollouts by rollout_id_in."""
    server, _client, session, api_endpoint = server_client

    # Create multiple rollouts
    r1 = await server.enqueue_rollout(input={"id": 1})
    _r2 = await server.enqueue_rollout(input={"id": 2})
    r3 = await server.enqueue_rollout(input={"id": 3})

    # Filter by specific rollout IDs
    async with session.get(
        f"{api_endpoint}/rollouts", params={"rollout_id_in": [r1.rollout_id, r3.rollout_id], "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 2
        rollout_ids = {item["rollout_id"] for item in data["items"]}
        assert rollout_ids == {r1.rollout_id, r3.rollout_id}


@pytest.mark.asyncio
async def test_rollouts_filter_by_rollout_id_contains(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test filtering rollouts by rollout_id_contains."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts
    r1 = await server.enqueue_rollout(input={"id": 1})
    _r2 = await server.enqueue_rollout(input={"id": 2})
    _r3 = await server.enqueue_rollout(input={"id": 3})

    # Extract a substring from r1's ID
    substring = r1.rollout_id[3:8]  # Get middle part of ID

    # Filter by substring
    async with session.get(f"{api_endpoint}/rollouts", params={"rollout_id_contains": substring, "limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] >= 1
        # Verify all results contain the substring
        for item in data["items"]:
            assert substring in item["rollout_id"]


@pytest.mark.asyncio
async def test_rollouts_filter_logic_and(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test rollouts filtering with AND logic."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts
    r1 = await server.enqueue_rollout(input={"id": 1})
    r2 = await server.enqueue_rollout(input={"id": 2})
    r3 = await server.enqueue_rollout(input={"id": 3})

    await server.update_rollout(rollout_id=r1.rollout_id, status="succeeded")
    await server.update_rollout(rollout_id=r2.rollout_id, status="succeeded")
    await server.update_rollout(rollout_id=r3.rollout_id, status="failed")

    # Filter with AND logic: status=succeeded AND rollout_id in list
    async with session.get(
        f"{api_endpoint}/rollouts",
        params={
            "status_in": ["succeeded"],
            "rollout_id_in": [r1.rollout_id, r3.rollout_id],
            "filter_logic": "and",
            "limit": -1,
        },
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        # Only r1 matches both conditions (r3 is failed, r2 is not in the ID list)
        assert data["total"] == 1
        assert data["items"][0]["rollout_id"] == r1.rollout_id


@pytest.mark.asyncio
async def test_rollouts_filter_logic_or(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test rollouts filtering with OR logic."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts
    r1 = await server.enqueue_rollout(input={"id": 1})
    _r2 = await server.enqueue_rollout(input={"id": 2})
    r3 = await server.enqueue_rollout(input={"id": 3})

    await server.update_rollout(rollout_id=r1.rollout_id, status="succeeded")
    # r2 and r3 remain queuing

    # Filter with OR logic: status=succeeded OR rollout_id=r3
    async with session.get(
        f"{api_endpoint}/rollouts",
        params={"status_in": ["succeeded"], "rollout_id_in": [r3.rollout_id], "filter_logic": "or", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        # Both r1 (succeeded) and r3 (in ID list) should match
        assert data["total"] == 2
        rollout_ids = {item["rollout_id"] for item in data["items"]}
        assert rollout_ids == {r1.rollout_id, r3.rollout_id}


@pytest.mark.asyncio
async def test_rollouts_sorting_with_none_values(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting rollouts by fields that may have None values."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts - mode is optional and can be None
    _r1 = await server.enqueue_rollout(input={"id": 1}, mode="train")
    r2 = await server.enqueue_rollout(input={"id": 2})  # mode=None
    _r3 = await server.enqueue_rollout(input={"id": 3}, mode="test")

    # Sort by mode ascending (None values should be treated as empty string/0)
    async with session.get(
        f"{api_endpoint}/rollouts", params={"sort_by": "mode", "sort_order": "asc", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert len(items) == 3
        # Items with None mode should come first (treated as 0)
        assert items[0]["rollout_id"] == r2.rollout_id
        assert items[0]["mode"] is None

    # Sort by mode descending
    async with session.get(
        f"{api_endpoint}/rollouts", params={"sort_by": "mode", "sort_order": "desc", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert len(items) == 3
        # Items with actual mode should come first
        assert items[0]["mode"] is not None
        # Item with None should be last
        assert items[2]["rollout_id"] == r2.rollout_id
        assert items[2]["mode"] is None


@pytest.mark.asyncio
async def test_rollouts_sorting_by_unsupported_field(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting by a field that doesn't exist on the model."""
    server, _client, session, api_endpoint = server_client

    # Create rollouts
    for i in range(3):
        await server.enqueue_rollout(input={"id": i})

    # Try to sort by a non-existent field - should return 400 error
    async with session.get(
        f"{api_endpoint}/rollouts", params={"sort_by": "nonexistent_field", "sort_order": "asc", "limit": -1}
    ) as resp:
        assert resp.status == 400
        data = await resp.json()
        assert "Invalid sort_by: nonexistent_field, allowed fields are: " in data["detail"]


# Attempts Pagination and Sorting Tests


@pytest.mark.asyncio
async def test_attempts_pagination_basic(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test basic pagination for attempts endpoint."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and multiple attempts
    rollout = await server.enqueue_rollout(input={"test": "data"})
    for _ in range(5):
        await server.start_attempt(rollout.rollout_id)

    # Get first page
    async with session.get(
        f"{api_endpoint}/rollouts/{rollout.rollout_id}/attempts", params={"limit": 2, "offset": 0}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 5
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert len(data["items"]) == 2

    # Get all attempts
    async with session.get(f"{api_endpoint}/rollouts/{rollout.rollout_id}/attempts", params={"limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 5
        assert len(data["items"]) == 5


@pytest.mark.asyncio
async def test_attempts_sorting(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting attempts by start_time."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and attempts
    rollout = await server.enqueue_rollout(input={"test": "data"})
    attempts: List[AttemptedRollout] = []
    for i in range(3):
        attempt = await server.start_attempt(rollout.rollout_id)
        attempts.append(attempt)
        await asyncio.sleep(0.01)

    # Sort by start_time descending (default)
    async with session.get(
        f"{api_endpoint}/rollouts/{rollout.rollout_id}/attempts",
        params={"sort_by": "start_time", "sort_order": "desc", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["start_time"] >= items[i + 1]["start_time"]

    # Sort ascending
    async with session.get(
        f"{api_endpoint}/rollouts/{rollout.rollout_id}/attempts",
        params={"sort_by": "start_time", "sort_order": "asc", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["start_time"] <= items[i + 1]["start_time"]


@pytest.mark.asyncio
async def test_attempts_sorting_with_none_values(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting attempts by fields that may have None values."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and attempts
    rollout = await server.enqueue_rollout(input={"test": "data"})
    a1 = await server.start_attempt(rollout.rollout_id)
    a2 = await server.start_attempt(rollout.rollout_id)
    a3 = await server.start_attempt(rollout.rollout_id)

    # Set worker_id for some attempts, leave others with None
    await server.update_attempt(rollout_id=rollout.rollout_id, attempt_id=a1.attempt.attempt_id, worker_id="worker-1")
    await server.update_attempt(rollout_id=rollout.rollout_id, attempt_id=a3.attempt.attempt_id, worker_id="worker-2")
    # a2 remains with worker_id=None

    # Sort by worker_id ascending (None values should be treated as empty/0)
    async with session.get(
        f"{api_endpoint}/rollouts/{rollout.rollout_id}/attempts",
        params={"sort_by": "worker_id", "sort_order": "asc", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert len(items) == 3
        # Items with None worker_id should come first
        assert items[0]["attempt_id"] == a2.attempt.attempt_id
        assert items[0]["worker_id"] is None


# Resources Pagination and Sorting Tests


@pytest.mark.asyncio
async def test_resources_pagination_basic(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test basic pagination for resources endpoint."""
    server, _client, session, api_endpoint = server_client

    # Create multiple resources
    for i in range(7):
        llm = LLM(
            resource_type="llm",
            endpoint="http://localhost:8080",
            model=f"model-v{i}",
            sampling_parameters={"temperature": 0.7},
        )
        await server.add_resources({"llm": llm})

    # Get first page
    async with session.get(f"{api_endpoint}/resources", params={"limit": 3, "offset": 0}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 7
        assert data["limit"] == 3
        assert len(data["items"]) == 3

    # Get all resources
    async with session.get(f"{api_endpoint}/resources", params={"limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 7
        assert len(data["items"]) == 7


@pytest.mark.asyncio
async def test_resources_sorting_by_resources_id(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting resources by resources_id."""
    server, _client, session, api_endpoint = server_client

    # Create resources
    for i in range(5):
        llm = LLM(
            resource_type="llm",
            endpoint="http://localhost:8080",
            model=f"model-{i}",
            sampling_parameters={"temperature": 0.7},
        )
        await server.add_resources({"llm": llm})

    # Sort by resources_id ascending (default)
    async with session.get(
        f"{api_endpoint}/resources", params={"sort_by": "resources_id", "sort_order": "asc", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["resources_id"] <= items[i + 1]["resources_id"]

    # Sort descending
    async with session.get(
        f"{api_endpoint}/resources", params={"sort_by": "resources_id", "sort_order": "desc", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        for i in range(len(items) - 1):
            assert items[i]["resources_id"] >= items[i + 1]["resources_id"]


@pytest.mark.asyncio
async def test_resources_filter_by_resources_id_contains(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test filtering resources by resources_id_contains."""
    server, _client, session, api_endpoint = server_client

    # Create resources with specific IDs
    await server.update_resources("test-resource-001", {})
    await server.update_resources("test-resource-002", {})
    await server.update_resources("prod-resource-003", {})

    # Filter by "test-" prefix
    async with session.get(f"{api_endpoint}/resources", params={"resources_id_contains": "test-", "limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 2
        ids = [item["resources_id"] for item in data["items"]]
        assert "test-resource-001" in ids
        assert "test-resource-002" in ids
        assert "prod-resource-003" not in ids

    # Filter by "-003" suffix
    async with session.get(f"{api_endpoint}/resources", params={"resources_id_contains": "-003", "limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 1
        assert data["items"][0]["resources_id"] == "prod-resource-003"


@pytest.mark.asyncio
async def test_resources_combined_filter_sort_and_pagination(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test combining filter, sort, and pagination for resources."""
    server, _client, session, api_endpoint = server_client

    # Create resources with pattern
    await server.update_resources("prod-app-001", {})
    await asyncio.sleep(0.01)
    await server.update_resources("prod-app-002", {})
    await asyncio.sleep(0.01)
    await server.update_resources("test-app-001", {})
    await asyncio.sleep(0.01)
    await server.update_resources("prod-db-001", {})

    # Filter by "prod-" and sort by resources_id, then paginate
    async with session.get(
        f"{api_endpoint}/resources",
        params={
            "resources_id_contains": "prod-",
            "sort_by": "resources_id",
            "sort_order": "asc",
            "limit": 2,
            "offset": 0,
        },
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 3  # 3 resources contain "prod-"
        assert data["limit"] == 2
        assert data["offset"] == 0
        assert len(data["items"]) == 2
        # Should get first 2 when sorted by resources_id asc
        assert data["items"][0]["resources_id"] == "prod-app-001"
        assert data["items"][1]["resources_id"] == "prod-app-002"

    # Get the next page
    async with session.get(
        f"{api_endpoint}/resources",
        params={
            "resources_id_contains": "prod-",
            "sort_by": "resources_id",
            "sort_order": "asc",
            "limit": 2,
            "offset": 2,
        },
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 3
        assert len(data["items"]) == 1  # Only 1 item remaining
        assert data["items"][0]["resources_id"] == "prod-db-001"


# Spans Pagination, Sorting, and Filtering Tests


@pytest.mark.asyncio
async def test_spans_pagination_basic(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test basic pagination for spans endpoint."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and attempt
    rollout = await server.start_rollout(input={"test": "data"})

    # Add multiple spans
    for i in range(10):
        span = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, i + 1, f"span-{i}")
        await server.add_span(span)

    # Get first page
    async with session.get(
        f"{api_endpoint}/spans", params={"rollout_id": rollout.rollout_id, "limit": 3, "offset": 0}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 10
        assert data["limit"] == 3
        assert len(data["items"]) == 3

    # Get all spans
    async with session.get(f"{api_endpoint}/spans", params={"rollout_id": rollout.rollout_id, "limit": -1}) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 10
        assert len(data["items"]) == 10


@pytest.mark.asyncio
async def test_spans_sorting_by_start_time(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting spans by start_time."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans
    rollout = await server.start_rollout(input={"test": "data"})
    for i in range(5):
        span = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, i + 1, f"span-{i}")
        await server.add_span(span)

    # Sort descending (default)
    async with session.get(
        f"{api_endpoint}/spans",
        params={"rollout_id": rollout.rollout_id, "sort_by": "start_time", "sort_order": "desc", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        # All our test spans have the same start_time, so just verify the structure
        assert len(items) == 5


@pytest.mark.asyncio
async def test_spans_filter_by_trace_id(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test filtering spans by trace_id."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans with different trace IDs
    rollout = await server.start_rollout(input={"test": "data"})

    span1 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 1, "span1")
    span1.trace_id = "trace-123"
    await server.add_span(span1)

    span2 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 2, "span2")
    span2.trace_id = "trace-456"
    await server.add_span(span2)

    span3 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 3, "span3")
    span3.trace_id = "trace-123"
    await server.add_span(span3)

    # Filter by exact trace_id
    async with session.get(
        f"{api_endpoint}/spans", params={"rollout_id": rollout.rollout_id, "trace_id": "trace-123", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 2
        for item in data["items"]:
            assert item["trace_id"] == "trace-123"


@pytest.mark.asyncio
async def test_spans_filter_by_name_contains(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test filtering spans by name_contains."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans with different names
    rollout = await server.start_rollout(input={"test": "data"})

    span1 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 1, "api_call_fetch")
    await server.add_span(span1)

    span2 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 2, "database_query")
    await server.add_span(span2)

    span3 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 3, "api_call_update")
    await server.add_span(span3)

    # Filter by name_contains
    async with session.get(
        f"{api_endpoint}/spans", params={"rollout_id": rollout.rollout_id, "name_contains": "api_call", "limit": -1}
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["total"] == 2
        for item in data["items"]:
            assert "api_call" in item["name"]


@pytest.mark.asyncio
async def test_spans_filter_logic_and(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test spans filtering with AND logic."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans
    rollout = await server.start_rollout(input={"test": "data"})

    span1 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 1, "api_call")
    span1.trace_id = "trace-123"
    await server.add_span(span1)

    span2 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 2, "api_call")
    span2.trace_id = "trace-456"
    await server.add_span(span2)

    span3 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 3, "database")
    span3.trace_id = "trace-123"
    await server.add_span(span3)

    # Filter with AND logic: trace_id=trace-123 AND name contains "api"
    async with session.get(
        f"{api_endpoint}/spans",
        params={
            "rollout_id": rollout.rollout_id,
            "trace_id": "trace-123",
            "name_contains": "api",
            "filter_logic": "and",
            "limit": -1,
        },
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        # Only span1 matches both conditions
        assert data["total"] == 1
        assert data["items"][0]["name"] == "api_call"
        assert data["items"][0]["trace_id"] == "trace-123"


@pytest.mark.asyncio
async def test_spans_filter_logic_or(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test spans filtering with OR logic."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans
    rollout = await server.start_rollout(input={"test": "data"})

    span1 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 1, "api_call")
    span1.trace_id = "trace-123"
    await server.add_span(span1)

    span2 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 2, "api_call")
    span2.trace_id = "trace-456"
    await server.add_span(span2)

    span3 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 3, "database")
    span3.trace_id = "trace-123"
    await server.add_span(span3)

    span4 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 4, "other")
    span4.trace_id = "trace-789"
    await server.add_span(span4)

    # Filter with OR logic: trace_id=trace-123 OR name contains "api"
    async with session.get(
        f"{api_endpoint}/spans",
        params={
            "rollout_id": rollout.rollout_id,
            "trace_id": "trace-123",
            "name_contains": "api",
            "filter_logic": "or",
            "limit": -1,
        },
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        # span1, span2, and span3 should match
        assert data["total"] == 3


@pytest.mark.asyncio
async def test_spans_sorting_with_none_values(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting spans by fields that may have None values."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans
    rollout = await server.start_rollout(input={"test": "data"})

    # Create spans with different parent_id values (parent_id can be None)
    span1 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 1, "span1")
    span1.parent_id = None  # Root span
    await server.add_span(span1)

    span2 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 2, "span2")
    span2.parent_id = "parent-a"  # Child span
    await server.add_span(span2)

    span3 = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, 3, "span3")
    span3.parent_id = "parent-b"  # Child span
    await server.add_span(span3)

    # Sort by parent_id ascending (None values should be treated as empty/0)
    async with session.get(
        f"{api_endpoint}/spans",
        params={"rollout_id": rollout.rollout_id, "sort_by": "parent_id", "sort_order": "asc", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert len(items) == 3
        # Items with None parent_id should come first (treated as 0)
        assert items[0]["sequence_id"] == 1
        assert items[0]["parent_id"] is None

    # Sort by parent_id descending
    async with session.get(
        f"{api_endpoint}/spans",
        params={"rollout_id": rollout.rollout_id, "sort_by": "parent_id", "sort_order": "desc", "limit": -1},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert len(items) == 3
        # Items with actual parent_id should come first
        assert items[0]["parent_id"] is not None
        # Item with None should be last
        assert items[2]["sequence_id"] == 1
        assert items[2]["parent_id"] is None


@pytest.mark.asyncio
async def test_spans_sorting_by_unsupported_field(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test sorting spans by a field that doesn't exist on the model."""
    server, _client, session, api_endpoint = server_client

    # Create rollout and spans
    rollout = await server.start_rollout(input={"test": "data"})
    for i in range(3):
        span = _make_span(rollout.rollout_id, rollout.attempt.attempt_id, i + 1, f"span-{i}")
        await server.add_span(span)

    # Try to sort by a non-existent field
    async with session.get(
        f"{api_endpoint}/spans",
        params={"rollout_id": rollout.rollout_id, "sort_by": "invalid_field", "sort_order": "asc", "limit": -1},
    ) as resp:
        assert resp.status == 400
        data = await resp.json()
        assert "Invalid sort_by: invalid_field, allowed fields are: " in data["detail"]


# LightningStoreClient._request_json pagination metadata tests


@pytest.mark.asyncio
async def test_request_json_rollouts_returns_pagination_metadata(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    server, client, _session, _api_endpoint = server_client

    for idx in range(3):
        await server.enqueue_rollout(input={"idx": idx})

    params = [
        ("sort_by", "rollout_id"),
        ("sort_order", "asc"),
        ("limit", 1),
        ("offset", 1),
    ]
    data = await client._request_json("get", "/rollouts", params=params)  # pyright: ignore[reportPrivateUsage]
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert data["total"] == 3
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_request_json_attempts_returns_pagination_metadata(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    _server, client, _session, _api_endpoint = server_client

    attempted = await client.start_rollout(input={"payload": "attempts"})
    await client.start_attempt(attempted.rollout_id)
    await client.start_attempt(attempted.rollout_id)

    params = [
        ("sort_by", "sequence_id"),
        ("sort_order", "asc"),
        ("limit", 1),
        ("offset", 1),
    ]
    data = await client._request_json(  # pyright: ignore[reportPrivateUsage]
        "get", f"/rollouts/{attempted.rollout_id}/attempts", params=params
    )
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert data["total"] == 3
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_request_json_resources_returns_pagination_metadata(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    _server, client, _session, _api_endpoint = server_client

    alpha = PromptTemplate(resource_type="prompt_template", template="alpha", engine="jinja")
    beta = PromptTemplate(resource_type="prompt_template", template="beta", engine="jinja")

    await client.update_resources("manual-alpha", {"prompt": alpha})
    await client.update_resources("manual-beta", {"prompt": beta})

    params = [
        ("sort_by", "resources_id"),
        ("sort_order", "asc"),
        ("limit", 1),
        ("offset", 1),
    ]
    data = await client._request_json("get", "/resources", params=params)  # pyright: ignore[reportPrivateUsage]
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert data["total"] == 2
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_request_json_workers_returns_pagination_metadata(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    _server, client, _session, _api_endpoint = server_client

    for worker_id in ["worker-a", "worker-b", "worker-c"]:
        await client.update_worker(worker_id, heartbeat_stats={"cpu": 0.1})

    params = [
        ("sort_by", "worker_id"),
        ("sort_order", "asc"),
        ("limit", 1),
        ("offset", 1),
    ]
    data = await client._request_json("get", "/workers", params=params)  # pyright: ignore[reportPrivateUsage]
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert data["total"] == 3
    assert len(data["items"]) == 1


@pytest.mark.asyncio
async def test_request_json_spans_returns_pagination_metadata(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    server, client, _session, _api_endpoint = server_client

    attempted = await server.start_rollout(input={"span": "meta"})
    attempt_id = attempted.attempt.attempt_id
    for idx in range(1, 4):
        await server.add_span(_make_span(attempted.rollout_id, attempt_id, idx, f"span-{idx}"))

    params = [
        ("rollout_id", attempted.rollout_id),
        ("attempt_id", attempt_id),
        ("sort_by", "sequence_id"),
        ("sort_order", "asc"),
        ("limit", 1),
        ("offset", 1),
    ]
    data = await client._request_json("get", "/spans", params=params)  # pyright: ignore[reportPrivateUsage]
    assert data["limit"] == 1
    assert data["offset"] == 1
    assert data["total"] == 3
    assert len(data["items"]) == 1


# Client Compatibility Tests


@pytest.mark.asyncio
async def test_client_query_rollouts_extracts_items(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test that client correctly extracts items from PaginatedResponse."""
    server, client, _session, _api_endpoint = server_client

    # Create rollouts
    for i in range(5):
        await server.enqueue_rollout(input={"index": i})

    # Query via client (should return PaginatedResult that behaves like a sequence)
    rollouts = await client.query_rollouts()
    assert isinstance(rollouts, PaginatedResult)
    assert rollouts.total == 5
    assert len(rollouts) == 5
    for rollout in rollouts:
        assert isinstance(rollout, Rollout)


@pytest.mark.asyncio
async def test_client_query_with_filters(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    """Test that client correctly passes filters to server."""
    server, client, _session, _api_endpoint = server_client

    # Create rollouts with different statuses
    r1 = await server.enqueue_rollout(input={"id": 1})
    r2 = await server.enqueue_rollout(input={"id": 2})
    await server.update_rollout(rollout_id=r1.rollout_id, status="succeeded")

    # Query via client with status filter
    rollouts = await client.query_rollouts(status=["succeeded"])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == r1.rollout_id

    # Query via client with rollout_ids filter
    rollouts = await client.query_rollouts(rollout_ids=[r2.rollout_id])
    assert len(rollouts) == 1
    assert rollouts[0].rollout_id == r2.rollout_id


@pytest.mark.asyncio
async def test_workers_endpoint_supports_updates(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    _server, _client, session, api_endpoint = server_client

    async with session.post(
        f"{api_endpoint}/workers/worker-1",
        json={"heartbeat_stats": {"cpu": 0.7}},
    ) as resp:
        assert resp.status == 200
        created = await resp.json()
        assert created["worker_id"] == "worker-1"
        assert created["status"] == "unknown"
        assert created["heartbeat_stats"] == {"cpu": 0.7}
        first_heartbeat = created["last_heartbeat_time"]

    async with session.get(f"{api_endpoint}/workers") as resp:
        assert resp.status == 200
        data = await resp.json()
        workers = data["items"]
        assert len(workers) == 1
        assert workers[0]["worker_id"] == "worker-1"

    async with session.post(
        f"{api_endpoint}/workers/worker-1",
        json={"heartbeat_stats": {"cpu": 0.8}},
    ) as resp:
        assert resp.status == 200
        updated = await resp.json()
        assert updated["last_heartbeat_time"] >= first_heartbeat

    async with session.get(f"{api_endpoint}/workers") as resp:
        assert resp.status == 200
        data = await resp.json()
        workers = data["items"]
        assert workers[0]["status"] == "unknown"


@pytest.mark.asyncio
async def test_workers_endpoint_rejects_none_stats(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    _server, _client, session, api_endpoint = server_client

    async with session.post(
        f"{api_endpoint}/workers/worker-err",
        json={"heartbeat_stats": None},
    ) as resp:
        assert resp.status == 400


@pytest.mark.asyncio
async def test_get_worker_by_id_restful(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    server, _client, session, api_endpoint = server_client

    await server.update_worker("worker-fetch", heartbeat_stats={"cpu": 0.4})

    async with session.get(f"{api_endpoint}/workers/worker-fetch") as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["worker_id"] == "worker-fetch"

    async with session.get(f"{api_endpoint}/workers/missing") as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data is None


@pytest.mark.asyncio
async def test_workers_endpoint_filter_and_sort(
    server_client: Tuple[LightningStoreServer, LightningStoreClient, aiohttp.ClientSession, str],
) -> None:
    server, _client, session, api_endpoint = server_client

    # Worker A: finishes an attempt and becomes idle.
    await server.update_worker("worker-a", heartbeat_stats={"cpu": 0.1})
    await server.enqueue_rollout(input={"worker": "a"})
    claimed_a = await server.dequeue_rollout(worker_id="worker-a")
    assert claimed_a is not None
    await server.update_attempt(
        claimed_a.rollout_id, claimed_a.attempt.attempt_id, worker_id="worker-a", status="succeeded"
    )

    # Worker B: currently busy on an attempt.
    await server.update_worker("worker-b", heartbeat_stats={"cpu": 0.9})
    await server.enqueue_rollout(input={"worker": "b"})
    claimed_b = await server.dequeue_rollout(worker_id="worker-b")
    assert claimed_b is not None
    await server.update_attempt(claimed_b.rollout_id, claimed_b.attempt.attempt_id, worker_id="worker-b")

    # Worker C: also busy.
    await server.update_worker("worker-c", heartbeat_stats={"cpu": 0.2})
    await server.enqueue_rollout(input={"worker": "c"})
    claimed_c = await server.dequeue_rollout(worker_id="worker-c")
    assert claimed_c is not None
    await server.update_attempt(claimed_c.rollout_id, claimed_c.attempt.attempt_id, worker_id="worker-c")

    async with session.get(
        f"{api_endpoint}/workers",
        params={"status_in": ["busy"], "worker_id_contains": "worker", "sort_by": "worker_id", "sort_order": "desc"},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        items = data["items"]
        assert [w["worker_id"] for w in items] == ["worker-c", "worker-b"]

    async with session.get(
        f"{api_endpoint}/workers",
        params={"limit": 1, "offset": 1, "sort_by": "worker_id", "sort_order": "asc"},
    ) as resp:
        assert resp.status == 200
        data = await resp.json()
        assert data["limit"] == 1
        assert data["offset"] == 1
        assert len(data["items"]) == 1
