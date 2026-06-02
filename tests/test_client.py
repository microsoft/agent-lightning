"""Tests for AglLiteClient — exercises client against real FastAPI app."""

from __future__ import annotations

import httpx
import pytest

from agl_lite.client import AglLiteClient, AglLiteError
from agl_lite.schemas import EventCreate
from agl_lite.schemas import Model
from agl_lite.schemas import RolloutCreate, RolloutPatch, RolloutState
from agl_lite.server.app import create_app

AGL_KEY = "test-key"
ADMIN_KEY = "test-admin-key"


@pytest.fixture
def app():
    return create_app({"key": AGL_KEY, "admin_key": ADMIN_KEY})


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
                RolloutCreate(input={"task": "a"}, config={}),
                RolloutCreate(input={"task": "b"}, config={}),
            ]
        )
        assert len(rollouts) == 2
        assert rollouts[0].status == RolloutState.QUEUING

        # Query all
        result = await client.query_rollouts()
        assert len(result) == 2

        # Query by IDs
        result = await client.query_rollouts(ids=[rollouts[0].rollout_id])
        assert len(result) == 1

        # Query by status
        result = await client.query_rollouts(status_in=[RolloutState.QUEUING])
        assert len(result) == 2
        result = await client.query_rollouts(status_in=[RolloutState.RUNNING])
        assert len(result) == 0

    async def test_get_rollout(self, client: AglLiteClient):
        [rollout] = await client.enqueue_rollouts(
            [
                RolloutCreate(input={"task": "x"}, config={}),
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
                RolloutCreate(input={}, config={}),
            ]
        )
        updated = await client.patch_rollout(
            rollout.rollout_id,
            RolloutPatch(status=RolloutState.RUNNING, job_name="agl-rollout-123"),
        )
        assert updated.status == RolloutState.RUNNING
        assert updated.job_name == "agl-rollout-123"

    async def test_patch_invalid_transition(self, client: AglLiteClient):
        [rollout] = await client.enqueue_rollouts(
            [
                RolloutCreate(input={}, config={}),
            ]
        )
        # queuing → succeeded is not valid
        with pytest.raises(AglLiteError, match="409"):
            await client.patch_rollout(
                rollout.rollout_id,
                RolloutPatch(status=RolloutState.SUCCEEDED),
            )

class TestEvents:
    async def test_post_and_get_events(self, client: AglLiteClient):
        [r] = await client.enqueue_rollouts(
            [
                RolloutCreate(input={}, config={}),
            ]
        )
        event = await client.post_event(
            r.rollout_id,
            "pod-uid-1",
            EventCreate(event_type="reward", data={"value": 0.9}),
        )
        assert event.event_type == "reward"

        events = await client.get_events(r.rollout_id, attempt_id="pod-uid-1")
        assert len(events) == 1
        assert events[0].data["value"] == 0.9

    async def test_get_events_with_filters(self, client: AglLiteClient):
        [r] = await client.enqueue_rollouts(
            [
                RolloutCreate(input={}, config={}),
            ]
        )
        await client.post_event(r.rollout_id, "pod-1", EventCreate(event_type="reward", data={"v": 1}))
        await client.post_event(r.rollout_id, "pod-1", EventCreate(event_type="custom", data={"v": 2}))

        events = await client.get_events(r.rollout_id, attempt_id="pod-1", event_type="reward")
        assert len(events) == 1
        assert events[0].event_type == "reward"


class TestModels:
    async def test_register_and_list(self, client: AglLiteClient):
        servers = await client.register_models(
            [
                Model(model="qwen-7b", endpoint="http://vllm-0:8000/v1", version=1),
                Model(model="qwen-7b", endpoint="http://vllm-1:8000/v1", version=1),
            ]
        )
        assert len(servers) == 2

        all_models = await client.list_models()
        assert len(all_models) == 2

    async def test_delete_model(self, client: AglLiteClient):
        await client.register_models(
            [
                Model(model="qwen-7b", endpoint="http://vllm-0:8000/v1", version=1),
                Model(model="qwen-7b", endpoint="http://vllm-1:8000/v1", version=1),
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
                Model(model="qwen-7b", endpoint="http://a:8000/v1", version=1),
                Model(model="llama-70b", endpoint="http://b:8000/v1", version=1),
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


class _ScriptedTransport(httpx.AsyncBaseTransport):
    """Transport that raises a scripted sequence of errors before succeeding."""

    def __init__(self, errors: list[Exception]) -> None:
        self._errors = list(errors)
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        return httpx.Response(200, json={"ok": True})


class TestRetryingTransport:
    """Regression coverage for the keep-alive race that crashed step 92.

    Bug repro: under a long-lived ``httpx.AsyncClient`` polling
    ``/api/rollouts/{rid}``, uvicorn's default 5s ``timeout_keep_alive`` closes
    idle pooled sockets, so the next request raises ``httpx.ReadError`` /
    ``RemoteProtocolError``. Before the fix, this single transient error
    propagated up to the Ray driver and tore down the entire training job.
    The ``_RetryingTransport`` wrapper must transparently retry it.
    """

    async def test_retries_read_error_then_succeeds(self) -> None:
        from agl_lite.client import _RetryingTransport

        scripted = _ScriptedTransport(
            [
                httpx.ReadError("simulated stale keep-alive"),
                httpx.RemoteProtocolError("server disconnected without response"),
            ]
        )
        transport = _RetryingTransport(scripted, initial_backoff_seconds=0.0)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.get("/api/rollouts/x")
        assert resp.status_code == 200
        assert scripted.calls == 3

    async def test_retries_connect_error(self) -> None:
        from agl_lite.client import _RetryingTransport

        scripted = _ScriptedTransport([httpx.ConnectError("ECONNREFUSED")])
        transport = _RetryingTransport(scripted, initial_backoff_seconds=0.0)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.get("/healthz")
        assert resp.status_code == 200
        assert scripted.calls == 2

    async def test_raises_after_max_attempts(self) -> None:
        from agl_lite.client import _RetryingTransport

        scripted = _ScriptedTransport([httpx.ReadError("persistent")] * 10)
        transport = _RetryingTransport(scripted, max_attempts=3, initial_backoff_seconds=0.0)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            with pytest.raises(httpx.ReadError):
                await c.get("/api/rollouts/x")
        assert scripted.calls == 3

    async def test_does_not_retry_application_errors(self) -> None:
        """5xx must NOT be retried — that's an application-level signal."""
        from agl_lite.client import _RetryingTransport

        class _Status500Transport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.calls = 0

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.calls += 1
                return httpx.Response(500, json={"detail": "boom"})

        inner = _Status500Transport()
        transport = _RetryingTransport(inner, initial_backoff_seconds=0.0)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.get("/api/rollouts/x")
        assert resp.status_code == 500
        assert inner.calls == 1
