# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import socket
import time
from contextlib import asynccontextmanager, closing
from typing import AsyncIterator

import aiohttp
import portpicker
import pytest
import uvicorn
from fastapi import FastAPI, Response

from agentlightning.utils.server_launcher import run_uvicorn_asyncio, shutdown_uvicorn_server


@asynccontextmanager
async def noop_context() -> AsyncIterator[None]:
    """A real async context manager that does nothing (satisfies serve_context)."""
    yield


def _make_app_health(always_ok: bool = True) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health():  # pyright: ignore[reportUnusedFunction]
        if always_ok:
            return {"ok": True}
        # Return non-200 to simulate failing health.
        return Response(status_code=503)

    @app.get("/")
    async def root():  # pyright: ignore[reportUnusedFunction]
        return {"hello": "world"}

    return app


def _new_server(app: FastAPI, host: str, port: int) -> uvicorn.Server:
    cfg = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="warning",  # keep test output quiet
        lifespan="on",  # exercise real lifespan
        loop="asyncio",
    )
    return uvicorn.Server(cfg)


@pytest.mark.asyncio
async def test_fastapi_health_ok_background_then_shutdown():
    """Server starts, health=200, wait_for_serve=False returns quickly; we then stop it."""
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=True)
    server = _new_server(app, host, port)

    serve_task = await run_uvicorn_asyncio(
        uvicorn_server=server,
        serve_context=noop_context(),
        timeout=10.0,
        health_url=f"http://{host}:{port}/health",
        wait_for_serve=False,  # return after health passes, leave server running
    )

    # After watcher returns, the server should be started and the task running
    assert isinstance(serve_task, asyncio.Task)
    assert not serve_task.done()
    assert server.started is True

    # Try the root endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{host}:{port}/") as resp:
            assert resp.status == 200
            assert await resp.json() == {"hello": "world"}

    # Clean shutdown
    await shutdown_uvicorn_server(server, serve_task)


@pytest.mark.asyncio
async def test_fastapi_no_health_background_then_shutdown():
    """Server starts without a health URL; watcher skips health and returns."""
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=True)
    server = _new_server(app, host, port)

    serve_task = await run_uvicorn_asyncio(
        uvicorn_server=server,
        serve_context=noop_context(),
        timeout=10.0,
        health_url=None,
        wait_for_serve=False,
    )

    assert isinstance(serve_task, asyncio.Task)
    assert not serve_task.done()
    assert server.started is True

    await shutdown_uvicorn_server(server, serve_task)


@pytest.mark.asyncio
async def test_fastapi_health_timeout_raises_and_server_is_stopped():
    """
    Server starts but /health never returns 200 -> watcher should raise RuntimeError.
    We still need to stop the running server task afterwards.
    """
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=False)  # returns 503
    server = _new_server(app, host, port)

    serve_task: asyncio.Task[None] | None = None
    with pytest.raises(RuntimeError) as ei:
        serve_task = await run_uvicorn_asyncio(
            uvicorn_server=server,
            serve_context=noop_context(),
            timeout=2.0,  # short to keep tests snappy
            health_url=f"http://{host}:{port}/health",
            wait_for_serve=False,  # only waits for watcher; it will raise
            kill_unhealthy_server=True,
        )

    assert "Server is not healthy" in str(ei.value)
    assert "has been killed" in str(ei.value)

    assert serve_task is None

    # The function raised before returning a serve_task, but server should be stopped.
    # Check if the port is still occupied.
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)
        assert s.fileno() is not None


@pytest.mark.asyncio
async def test_uvicorn_startup_failure_is_wrapped_as_runtimeerror():
    """
    Bind a port first to force uvicorn to fail binding the socket.
    That should cause run_uvicorn_asyncio to raise RuntimeError("Uvicorn server failed to serve").
    """
    host = "127.0.0.1"
    conflict_port = portpicker.pick_unused_port()

    # occupy the port to force bind failure
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, conflict_port))
        s.listen(1)

        app = _make_app_health(always_ok=True)
        server = _new_server(app, host, conflict_port)

        with pytest.raises(RuntimeError) as ei:
            # wait_for_serve=True so we propagate the serve exception path
            await run_uvicorn_asyncio(
                uvicorn_server=server,
                serve_context=noop_context(),
                timeout=3.0,
                health_url=None,
                wait_for_serve=True,
            )

    assert "Uvicorn server failed to serve" in str(ei.value)


@pytest.mark.asyncio
async def test_startup_deadline_if_server_never_starts():
    """
    Point health check to a valid URL but prevent server from flipping 'started' by
    never actually launching uvicorn (simulate by using a server configured for
    an unreachable host to fail before started). Alternatively, we can create a
    server that immediately raises before 'started' is set by using an invalid host.
    """
    # Using an invalid bind address triggers an early failure; the watcher should
    # treat it as startup deadline with the original exception chained.
    host = "203.0.113.123"  # TEST-NET-3, won't be a local addr to bind
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=True)
    server = _new_server(app, host, port)

    with pytest.raises(RuntimeError) as ei:
        await run_uvicorn_asyncio(
            uvicorn_server=server,
            serve_context=noop_context(),
            timeout=1.0,
            health_url=None,
            wait_for_serve=True,
        )
    # Either path is acceptable: wrapped serve failure or startup-timeout message
    msg = str(ei.value)
    assert ("Uvicorn server failed to serve" in msg) or ("Server did not start up within" in msg)


@pytest.mark.asyncio
async def test_port_already_used_conflict_raises_runtimeerror():
    """
    Explicitly occupy a port before starting uvicorn. The server should fail its bind,
    and run_uvicorn_asyncio should surface it as RuntimeError('Uvicorn server failed to serve').
    """
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()

    # Occupy the port
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, port))
        s.listen(1)

        app = _make_app_health(always_ok=True)
        server = _new_server(app, host, port)

        with pytest.raises(RuntimeError) as ei:
            await run_uvicorn_asyncio(
                uvicorn_server=server,
                serve_context=noop_context(),
                timeout=3.0,
                health_url=f"http://{host}:{port}/health",
                wait_for_serve=True,  # ensures we await the serve path that raises
            )
        assert "Uvicorn server failed to serve" in str(ei.value)


@pytest.mark.asyncio
async def test_wait_for_serve_true_graceful_shutdown():
    """
    Start the server with health check and wait_for_serve=True.
    We run run_uvicorn_asyncio in the background, wait for startup, then signal shutdown.
    Finally, we verify the returned serve task completed cleanly.
    """
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=True)
    server = _new_server(app, host, port)

    # Kick off run_uvicorn_asyncio in the background because wait_for_serve=True
    # will await the server's lifecycle (we'll signal shutdown below).
    run_task = asyncio.create_task(
        run_uvicorn_asyncio(
            uvicorn_server=server,
            serve_context=noop_context(),
            timeout=10.0,
            health_url=f"http://{host}:{port}/health",
            wait_for_serve=True,
        )
    )

    # Wait until uvicorn flips `started` or time out
    start = time.time()
    while not server.started and (time.time() - start) < 5.0:
        await asyncio.sleep(0.05)
    assert server.started is True, "Server did not report started in time"

    # Try the root endpoint
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{host}:{port}/") as resp:
            assert resp.status == 200
            assert await resp.json() == {"hello": "world"}

    # Trigger graceful shutdown and wait for run_task to finish
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task


@pytest.mark.asyncio
async def test_unhealthy_left_running_background_then_manual_shutdown(caplog: pytest.LogCaptureFixture):
    """
    When kill_unhealthy_server=False and wait_for_serve=False, the watcher raises
    while the server continues running. We verify it's still reachable, then shut it down.
    """
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=False)  # health returns 503
    server = _new_server(app, host, port)

    # run_uvicorn_asyncio will raise; server continues running in background
    caplog.set_level(logging.ERROR)
    serve_task = await run_uvicorn_asyncio(
        uvicorn_server=server,
        serve_context=noop_context(),
        timeout=2.0,
        health_url=f"http://{host}:{port}/health",
        wait_for_serve=False,
        kill_unhealthy_server=False,  # <-- leave server running
    )
    assert any("left running" in rec.message for rec in caplog.records)

    # Should still be started and serving:
    assert server.started is True

    # Root endpoint should be reachable even though /health is failing.
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{host}:{port}/") as resp:
            assert resp.status == 200
            assert await resp.json() == {"hello": "world"}

    # Now manually stop the server and wait for it to go down.
    await shutdown_uvicorn_server(server, serve_task)


@pytest.mark.asyncio
async def test_unhealthy_left_running_wait_for_serve_true(caplog: pytest.LogCaptureFixture):
    """
    With kill_unhealthy_server=False and wait_for_serve=True, the call raises,
    but uvicorn keeps running. Verify it's serving, then stop it.
    """
    host = "127.0.0.1"
    port = portpicker.pick_unused_port()
    app = _make_app_health(always_ok=False)
    server = _new_server(app, host, port)
    print("entered")

    caplog.set_level(logging.ERROR)
    run_task = asyncio.create_task(
        run_uvicorn_asyncio(
            uvicorn_server=server,
            serve_context=noop_context(),
            timeout=2.0,
            health_url=f"http://{host}:{port}/health",
            wait_for_serve=True,
            kill_unhealthy_server=False,  # leave server running
        )
    )

    # Wait until uvicorn flips `started` or time out
    start = time.time()
    while not server.started and (time.time() - start) < 5.0:
        await asyncio.sleep(0.05)
    print("server started", server.started)
    assert server.started is True, "Server did not report started in time"
    await asyncio.sleep(max(0.01, start + 3.0 - time.time()))
    print("after sleep")
    assert any("left running" in rec.message for rec in caplog.records)
    print("server started")

    # Verify we can still hit the root.
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{host}:{port}/") as resp:
            assert resp.status == 200
            assert await resp.json() == {"hello": "world"}

    # Trigger graceful shutdown and wait for run_task to finish
    run_task.cancel()
    print("cancelling run_task")
    with pytest.raises(asyncio.CancelledError):
        await run_task
