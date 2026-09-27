# Copyright (c) Microsoft. All rights reserved.

"""Gateway request ownership when agent connections disappear."""

import asyncio
import socket
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.types import Message

from agentlightning.server.routes.proxy import llm_proxy

from .conftest import MODEL_NAME


@asynccontextmanager
async def _serve(app: FastAPI) -> AsyncIterator[str]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", timeout_graceful_shutdown=2))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    await task
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


async def _register_rollout(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/rollouts", json=[{"input": "hello"}])
    response.raise_for_status()
    return response.json()[0]["rollout_id"]


def _proxy_path(rollout_id: str) -> str:
    return f"/proxy/rollout/{rollout_id}/attempt/0/mode/train/openai/v1/chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["upstream", "retry_backoff"])
async def test_disconnect_drains_only_abandoned_request(
    app: FastAPI, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    backend = FastAPI()
    release = asyncio.Event()
    orphan_started = asyncio.Event()
    orphan_cleaned = asyncio.Event()
    healthy_started = asyncio.Event()
    calls: Counter[str] = Counter()

    @backend.post("/v1/chat/completions")
    async def completion(request: Request):
        body = await request.json()
        name = body["messages"][0]["content"]
        calls[name] += 1
        if name == "orphan":
            if phase == "retry_backoff":
                return JSONResponse({"error": "temporarily busy"}, status_code=503)
            orphan_started.set()
            while not release.is_set():
                if await request.is_disconnected():
                    orphan_cleaned.set()
                    return JSONResponse({"error": "disconnected"})
                await asyncio.sleep(0.01)
        else:
            healthy_started.set()
            await release.wait()
        return {"prompt_token_ids": [1], "choices": [{"token_ids": [2], "message": {"content": "ok"}}]}

    if phase == "retry_backoff":

        async def backoff(**kwargs: object) -> None:
            orphan_started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                orphan_cleaned.set()
                raise

        monkeypatch.setattr("agentlightning.server.proxy._sleep_before_retry", backoff)

    async with (
        _serve(backend) as backend_url,
        _serve(app) as gateway_url,
        httpx.AsyncClient(base_url=gateway_url, headers=auth_headers, timeout=5) as client,
    ):
        response = await client.post("/api/models", json=[{"model": MODEL_NAME, "endpoint": f"{backend_url}/v1"}])
        response.raise_for_status()
        orphan = await _register_rollout(client)
        healthy = await _register_rollout(client)
        orphan_task = asyncio.create_task(
            client.post(_proxy_path(orphan), json={"messages": [{"role": "user", "content": "orphan"}]})
        )
        healthy_task = asyncio.create_task(
            client.post(_proxy_path(healthy), json={"messages": [{"role": "user", "content": "healthy"}]})
        )
        try:
            await asyncio.wait_for(orphan_started.wait(), 3)
            await asyncio.wait_for(healthy_started.wait(), 3)
            orphan_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await orphan_task
            await asyncio.wait_for(orphan_cleaned.wait(), 3)
            async with asyncio.timeout(3):
                while (await client.get("/proxy/state")).json()["inflight"] != 1:
                    await asyncio.sleep(0.01)

            paused = await client.post("/proxy/pause", json={"reason": "training update"})
            assert paused.json()["inflight"] == 1
            rejected = await client.post(_proxy_path(healthy), json={})
            assert rejected.status_code == 429
            assert not healthy_task.done()

            release.set()
            assert (await healthy_task).status_code == 200
            assert (await client.get("/proxy/state")).json()["inflight"] == 0
            assert (await client.get(f"/api/rollouts/{orphan}/events")).json() == []
            assert len((await client.get(f"/api/rollouts/{healthy}/events")).json()) == 1
            assert calls == {"orphan": 1, "healthy": 1}
        finally:
            release.set()
            orphan_task.cancel()
            healthy_task.cancel()
            await asyncio.gather(orphan_task, healthy_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_handler_joins_forwarding_and_disconnect_watcher(
    app: FastAPI, auth_headers: dict[str, str]
) -> None:
    forwarding_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    watcher_started = asyncio.Event()
    watcher_stopped = asyncio.Event()

    async def upstream(request: httpx.Request) -> httpx.Response:
        forwarding_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleanup_started.set()
            await release_cleanup.wait()
        raise AssertionError("unreachable")

    body_sent = False

    async def receive() -> Message:
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": b"{}", "more_body": False}
        watcher_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            watcher_stopped.set()
        raise AssertionError("unreachable")

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test", headers=auth_headers
        ) as client,
    ):
        response = await client.post("/api/models", json=[{"model": MODEL_NAME, "endpoint": "http://model/v1"}])
        response.raise_for_status()
        rollout_id = await _register_rollout(client)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as backend:
            app.state.http_client = backend
            request = Request({"type": "http", "app": app}, receive=receive)
            task = asyncio.create_task(llm_proxy(rollout_id, "0", "train", "chat/completions", request))
            try:
                await asyncio.wait_for(forwarding_started.wait(), 3)
                await asyncio.wait_for(watcher_started.wait(), 3)
                task.cancel()
                await asyncio.wait_for(cleanup_started.wait(), 3)
                assert not task.done()
                assert app.state.proxy_pause_state.inflight == 1
                release_cleanup.set()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert watcher_stopped.is_set()
                assert app.state.proxy_pause_state.inflight == 0
                assert (await client.get(f"/api/rollouts/{rollout_id}/events")).json() == []
            finally:
                release_cleanup.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
