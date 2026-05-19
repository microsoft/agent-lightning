"""Tests for AglLiteClient — exercises client against real FastAPI app."""

from __future__ import annotations

import httpx
import pytest

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.schemas.api import (
    EnqueueRolloutRequest,
    PatchRolloutRequest,
    PostEventRequest,
    RegisterModelRequest,
)
from agl_lite.schemas.rollout import RolloutStatus
from agl_lite.server.app import create_app
from agl_lite.server.config import ServerSettings

AGL_KEY = "test-key"
ADMIN_KEY = "test-admin-key"


@pytest.fixture
def app():
    return create_app(ServerSettings(key=AGL_KEY, admin_key=ADMIN_KEY))


@pytest.fixture
async def client(app) -> AglLiteClient:
    """AglLiteClient wired to the FastAPI app via httpx ASGITransport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    http_client = httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"Authorization": f"Bearer {AGL_KEY}"},
    )
    # Trigger lifespan startup (creates store, router, etc.)
    async with app.router.lifespan_context(app):
        c = AglLiteClient(base_url="http://testserver", agl_key=AGL_KEY, client=http_client)
        yield c
        await c.close()


class TestRollouts:
    async def test_enqueue_and_query(self, client: AglLiteClient):
        rollouts = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={"task": "a"}, config={}),
                EnqueueRolloutRequest(input={"task": "b"}, config={}),
            ]
        )
        assert len(rollouts) == 2
        assert rollouts[0].status == RolloutStatus.QUEUING

        # Query all
        result = await client.query_rollouts()
        assert len(result) == 2

        # Query by IDs
        result = await client.query_rollouts(ids=[rollouts[0].rollout_id])
        assert len(result) == 1

        # Query by status
        result = await client.query_rollouts(status_in=[RolloutStatus.QUEUING])
        assert len(result) == 2
        result = await client.query_rollouts(status_in=[RolloutStatus.RUNNING])
        assert len(result) == 0

    async def test_get_rollout(self, client: AglLiteClient):
        [rollout] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={"task": "x"}, config={}),
            ]
        )
        fetched = await client.get_rollout(rollout.rollout_id)
        assert fetched.rollout_id == rollout.rollout_id

    async def test_get_rollout_not_found(self, client: AglLiteClient):
        with pytest.raises(AglLiteError, match="404"):
            await client.get_rollout("nonexistent")

    async def test_patch_rollout(self, client: AglLiteClient):
        [rollout] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={}, config={}),
            ]
        )
        updated = await client.patch_rollout(
            rollout.rollout_id,
            PatchRolloutRequest(status=RolloutStatus.RUNNING, job_name="agl-rollout-123"),
        )
        assert updated.status == RolloutStatus.RUNNING
        assert updated.job_name == "agl-rollout-123"

    async def test_patch_invalid_transition(self, client: AglLiteClient):
        [rollout] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={}, config={}),
            ]
        )
        # queuing → succeeded is not valid
        with pytest.raises(AglLiteError, match="409"):
            await client.patch_rollout(
                rollout.rollout_id,
                PatchRolloutRequest(status=RolloutStatus.SUCCEEDED),
            )

    async def test_cancel_rollout(self, client: AglLiteClient):
        [rollout] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={}, config={}),
            ]
        )
        cancelled = await client.cancel_rollout(rollout.rollout_id)
        assert cancelled.cancel_requested is True

    async def test_archive_rollouts(self, client: AglLiteClient, tmp_path):
        [r] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={}, config={}),
            ]
        )
        # Move to terminal state first
        await client.patch_rollout(r.rollout_id, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        await client.patch_rollout(r.rollout_id, PatchRolloutRequest(status=RolloutStatus.SUCCEEDED))

        from agl_lite.schemas.api import ArchiveBackend

        path = str(tmp_path / "archive.jsonl")
        result = await client.archive_rollouts([r.rollout_id], backend=ArchiveBackend(path=path))
        assert result.archived == 1
        assert result.purged == 1


class TestEvents:
    async def test_post_and_get_events(self, client: AglLiteClient):
        [r] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={}, config={}),
            ]
        )
        event = await client.post_event(
            r.rollout_id,
            "pod-uid-1",
            PostEventRequest(event_type="reward", data={"value": 0.9}),
        )
        assert event.event_type == "reward"

        events = await client.get_events(r.rollout_id, attempt_id="pod-uid-1")
        assert len(events) == 1
        assert events[0].data["value"] == 0.9

    async def test_get_events_with_filters(self, client: AglLiteClient):
        [r] = await client.enqueue_rollouts(
            [
                EnqueueRolloutRequest(input={}, config={}),
            ]
        )
        await client.post_event(r.rollout_id, "pod-1", PostEventRequest(event_type="reward", data={"v": 1}))
        await client.post_event(r.rollout_id, "pod-1", PostEventRequest(event_type="custom", data={"v": 2}))

        events = await client.get_events(r.rollout_id, attempt_id="pod-1", event_type="reward")
        assert len(events) == 1
        assert events[0].event_type == "reward"


class TestModels:
    async def test_register_and_list(self, client: AglLiteClient):
        servers = await client.register_models(
            [
                RegisterModelRequest(model="qwen-7b", endpoint="http://vllm-0:8000/v1", version=1),
                RegisterModelRequest(model="qwen-7b", endpoint="http://vllm-1:8000/v1", version=1),
            ]
        )
        assert len(servers) == 2

        all_models = await client.list_models()
        assert len(all_models) == 2

    async def test_delete_model(self, client: AglLiteClient):
        await client.register_models(
            [
                RegisterModelRequest(model="qwen-7b", endpoint="http://vllm-0:8000/v1", version=1),
                RegisterModelRequest(model="qwen-7b", endpoint="http://vllm-1:8000/v1", version=1),
            ]
        )
        # Delete one endpoint
        await client.delete_model("qwen-7b", endpoints=["http://vllm-0:8000/v1"])
        remaining = await client.list_models()
        assert len(remaining) == 1

        # Delete entire model
        await client.delete_model("qwen-7b")
        remaining = await client.list_models()
        assert len(remaining) == 0

    async def test_delete_all_models(self, client: AglLiteClient):
        await client.register_models(
            [
                RegisterModelRequest(model="qwen-7b", endpoint="http://a:8000/v1", version=1),
                RegisterModelRequest(model="llama-70b", endpoint="http://b:8000/v1", version=1),
            ]
        )
        await client.delete_all_models()
        assert await client.list_models() == []


class TestResources:
    async def test_add_and_get(self, client: AglLiteClient):
        res = await client.add_resources({"system_prompt": "You are helpful."})
        assert res.resources_id
        assert res.resources["system_prompt"] == "You are helpful."

        fetched = await client.get_resources(res.resources_id)
        assert fetched.resources_id == res.resources_id

    async def test_get_latest(self, client: AglLiteClient):
        # No resources yet
        result = await client.get_latest_resources()
        assert result is None

        await client.add_resources({"v": 1})
        res2 = await client.add_resources({"v": 2})

        latest = await client.get_latest_resources()
        assert latest is not None
        assert latest.resources_id == res2.resources_id

    async def test_get_resources_not_found(self, client: AglLiteClient):
        with pytest.raises(AglLiteError, match="404"):
            await client.get_resources("nonexistent")
