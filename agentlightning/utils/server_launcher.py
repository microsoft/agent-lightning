# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue
import signal
import socket
import threading
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Dict, Literal, Optional, Union

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI
from gunicorn.app.base import BaseApplication
from gunicorn.arbiter import Arbiter
from portpicker import pick_unused_port

__all__ = ["PythonServerLauncher", "PythonServerLauncherArgs"]


LaunchMode = Literal["asyncio", "thread", "mp"]


@dataclass
class PythonServerLauncherArgs:
    port: Optional[int] = None
    """The TCP port to listen on. If not provided, the server will use a random available port."""
    host: str = "127.0.0.1"
    """The hostname or IP address to bind the server to."""
    launch_mode: LaunchMode = "asyncio"
    """The launch mode. `asyncio` is the default mode to runs the server in the current thread.
    `thread` runs the server in a separate thread. `mp` runs the server in a separate process."""
    n_workers: int = 1
    """The number of workers to run in the server. Only applicable for `mp` mode.
    When `n_workers > 1`, the server will be run using Gunicorn.
    """
    healthcheck_url: Optional[str] = None
    """The health check URL to use.
    If not provided, the server will not be checked for healthiness after starting.
    """
    log_level: int = logging.INFO
    """The log level to use."""


@dataclass
class ChildEvent:
    """An event that occurred in a child process."""

    kind: Literal["ready", "error"]
    """The kind of message."""
    exc_type: Optional[str] = None
    """The type of the exception, only used for error messages."""
    message: Optional[str] = None
    """The message of the exception, only used for error messages."""
    traceback: Optional[str] = None
    """The traceback of the exception, only used for error messages."""


logger = logging.getLogger(__name__)


class GunicornApp(BaseApplication):
    """
    Programmatic Gunicorn application that:

    - Accepts a `FastAPI` app object and option dict.
    - Uses `uvicorn_worker.UvicornWorker`.
    """

    def __init__(self, app: FastAPI, options: Dict[str, Any]):
        super().__init__()  # type: ignore
        self.application = app
        self.options = options or {}

    def load_config(self):
        cfg = self.cfg
        valid_keys = cfg.settings.keys()  # type: ignore
        for k, v in (self.options or {}).items():
            if k in valid_keys and v is not None:
                cfg.set(k, v)  # type: ignore

    def load(self):
        return self.application


async def shutdown_uvicorn_server(server: uvicorn.Server, task: asyncio.Task[None], timeout: float = 5.0) -> None:
    """Shutdown a uvicorn server and await the serving task."""
    logger.debug("Requesting graceful shutdown of uvicorn server.")
    server.should_exit = True
    # Give uvicorn a brief window to shut down cleanly.
    try:
        logger.debug("Waiting for graceful shutdown of uvicorn server.")
        await asyncio.wait_for(task, timeout=timeout)
        logger.debug("Graceful shutdown of uvicorn server completed.")
    except asyncio.TimeoutError:
        logger.error("Graceful shutdown of uvicorn server timed out.")
        # As a last resort, cancel; this shouldn't happen under normal circumstances.
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        logger.warning("Uvicorn server forced to stop.")


async def run_uvicorn_asyncio(
    uvicorn_server: uvicorn.Server,
    serve_context: AsyncContextManager[Any],
    timeout: float = 60.0,
    health_url: Optional[str] = None,
    wait_for_serve: bool = True,
    kill_unhealthy_server: bool = True,
) -> asyncio.Task[None]:
    """Run two Asyncio tasks in parallel:

    - A watcher task that waits for the server to start up and then checks for healthiness.
    - A server task that serves the server.
    """
    server_start_exception: Optional[BaseException] = None

    # watcher: when server.started flips True, announce READY once
    async def _watch_server() -> None:
        start_time = time.monotonic()
        deadline = start_time + timeout  # child-side startup window
        logger.debug(f"Waiting for server to start up for {timeout:.2f} seconds...")
        # Wait for the server to start up or the deadline to be reached, or an exception to be raised.
        while time.monotonic() < deadline and not uvicorn_server.started and server_start_exception is None:
            await asyncio.sleep(0.1)

        if not uvicorn_server.started:
            # Normally, the program will not reach this point, as the server will throw the exception itself earlier.
            raise RuntimeError(f"Server did not start up within {timeout:.2f} seconds.") from server_start_exception

        logger.debug(f"Server started up in {time.monotonic() - start_time:.2f} seconds.")

        # Check for health endpoint status if provided
        if health_url is not None:
            async with aiohttp.ClientSession() as session:
                while time.monotonic() < deadline:
                    with suppress(Exception):
                        async with session.get(health_url) as resp:
                            if resp.status == 200:
                                logger.debug(
                                    f"Server is healthy at {health_url} in {time.monotonic() - start_time:.2f} seconds."
                                )
                                return
                    await asyncio.sleep(0.1)

            # If the server is not healthy, kill it if requested.
            health_failed_seconds = time.monotonic() - start_time
            if kill_unhealthy_server:
                logger.error(
                    f"Server is not healthy at {health_url} after {health_failed_seconds:.2f} seconds. Shutting down server gracefully."
                )
                uvicorn_server.should_exit = True
                await serve_task

                raise RuntimeError(
                    f"Server is not healthy at {health_url} after {health_failed_seconds:.2f} seconds. It has been killed."
                )
            else:
                logger.error(
                    f"Server is not healthy at {health_url} after {health_failed_seconds:.2f} seconds. It has been left running."
                )

        else:
            logger.debug("Server does not provide a health check endpoint. Skipping health check.")

    async def _serve_server() -> None:
        nonlocal server_start_exception
        async with serve_context:
            try:
                await uvicorn_server.serve()
            except (asyncio.CancelledError, KeyboardInterrupt):
                # Normal shutdown path; propagate without rewrapping
                raise
            except BaseException as exc:
                server_start_exception = exc
                if wait_for_serve:
                    # This probably sends out earlier than watcher exception; but either one is fine.
                    raise RuntimeError("Uvicorn server failed to serve") from exc
                else:
                    # If the caller is not waiting for this coroutine, we just log the error.
                    # It will be handled by the watch task.
                    logger.exception("Uvicorn server failed to serve. Inspect the logs for details.")

    watch_task = asyncio.create_task(_watch_server())
    serve_task = asyncio.create_task(_serve_server())

    if wait_for_serve:
        await asyncio.gather(watch_task, serve_task)
    else:
        # Wait for watch only, the serve task will run in the background.
        await watch_task
    return serve_task


def run_uvicorn_thread(
    uvicorn_server: uvicorn.Server,
    serve_context: AsyncContextManager[Any],
    event_queue: queue.Queue[ChildEvent],
    stop_event: threading.Event,
    timeout: float = 60.0,
    health_url: Optional[str] = None,
):
    """
    Run a uvicorn server in a thread.

    How to stop programmatically (from the main thread):

        uvicorn_server.should_exit = True

    This function:

    - starts the server and waits for startup/health (if provided),
    - then blocks until the server exits,
    - shuts down cleanly if an error happens during startup/health,
    - or if the thread is stopped by stop event.
    """

    async def _main() -> None:
        # Start server without waiting for full lifecycle; return once startup/health is done.
        serve_task: Optional[asyncio.Task[None]] = None
        try:
            serve_task = await run_uvicorn_asyncio(
                uvicorn_server=uvicorn_server,
                serve_context=serve_context,
                timeout=timeout,
                health_url=health_url,
                wait_for_serve=False,  # return after startup watcher finishes
                kill_unhealthy_server=True,  # raise if health fails within timeout
            )
            event_queue.put(ChildEvent(kind="ready"))
        except Exception as exc:
            # Startup/health failed; nothing is running in the background.
            logger.exception("Uvicorn failed to start or was unhealthy.")
            event_queue.put(
                ChildEvent(
                    kind="error", exc_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
                )
            )
            return

        logger.debug("Thread server started and ready.")
        try:
            # At this point, the server is up and serving in the same thread's loop.
            # Block here until it exits (caller can stop it via setting the stop_event).
            while not stop_event.is_set():
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            # Shutdown the server.
            logger.warning(
                "Thread server received asyncio cancellation signal. Shutting down gracefully. This is not the recommended way to stop the server."
            )
            raise
        except Exception as exc:
            logger.exception("Exception during the thread event waiting loop.")
            event_queue.put(
                ChildEvent(
                    kind="error", exc_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
                )
            )
        finally:
            logger.info("Requesting graceful shutdown of uvicorn server.")
            await shutdown_uvicorn_server(uvicorn_server, serve_task)
            logger.info("Uvicorn server shut down gracefully.")

    # Each thread needs its own event loop; use asyncio.run to manage it cleanly.
    try:
        asyncio.run(_main())
    except Exception:
        # Exceptions are already logged above; don't crash the process from a thread.
        # (Caller can inspect logs or add a queue/handler if they need to propagate.)
        logger.exception("Exception within the thread server loop. Inspect the logs for details.")


def run_uvicorn_subprocess(
    uvicorn_server: uvicorn.Server,
    serve_context: AsyncContextManager[Any],
    event_queue: multiprocessing.Queue[ChildEvent],
    timeout: float = 60.0,
    health_url: Optional[str] = None,
):
    """Run a uvicorn server in a subprocess.

    Behavior:

    - Start uvicorn and wait for startup/health (if provided).
    - Post ChildEvent(kind="ready") once the server is up.
    - Stay alive until a termination signal (SIGTERM/SIGINT).
    - On signal, request graceful shutdown and wait for the server to exit.

    This must be used with forked multiprocessing.Process.
    """

    async def _main() -> None:
        stop_event = asyncio.Event()

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop_event.set)
        logger.debug("Subprocess signal handlers registered.")

        serve_task: Optional[asyncio.Task[None]] = None

        try:
            # Start server but don't block on its full lifecycle; this returns once the watcher finishes.
            serve_task = await run_uvicorn_asyncio(
                uvicorn_server=uvicorn_server,
                serve_context=serve_context,
                timeout=timeout,
                health_url=health_url,
                wait_for_serve=False,  # return after startup/health passes
                kill_unhealthy_server=True,  # if unhealthy, fail fast in the child
            )

            # Announce readiness only after watcher success.
            event_queue.put(ChildEvent(kind="ready"))

            logger.debug("Subprocess server started and ready.")

            # Wait until we're told to stop.
            await stop_event.wait()

        except Exception as exc:
            # Propagate any startup/health errors to the parent.
            event_queue.put(
                ChildEvent(
                    kind="error",
                    exc_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
            logger.exception("Subprocess server failed to start or was unhealthy.")

        finally:
            # Request graceful shutdown if the server is running.
            if serve_task is not None:
                logger.info("Requesting graceful shutdown of subprocess server.")
                await shutdown_uvicorn_server(uvicorn_server, serve_task)
                logger.info("Subprocess server shut down gracefully.")
            else:
                logger.info("Subprocess server was not running. Nothing to stop.")

    try:
        asyncio.run(_main())
    except Exception as exc:
        # If something escapes _main(), make sure the parent hears about it.
        event_queue.put(
            ChildEvent(
                kind="error",
                exc_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        )


def run_gunicorn(
    gunicorn_app: GunicornApp,
    serve_context: AsyncContextManager[Any],
    event_queue: multiprocessing.Queue[ChildEvent],
    timeout: float = 60.0,
    health_url: Optional[str] = None,
):
    """Run a gunicorn server in a subprocess.

    The master arbiter will reside in a non-daemon subprocess,
    and the workers will be forked from the arbiter.

    Behavior:

    - Start Arbiter.run() (blocking) in this process.
    - A watchdog thread waits for workers to spawn, then (optionally) verifies a health URL.
    - On success: put ChildEvent(kind="ready").
    - On failure/timeout: put ChildEvent(kind="error") and request a graceful shutdown.
    """
    # Create the arbiter up-front so the watchdog can inspect it.
    try:
        arbiter = Arbiter(gunicorn_app)
    except Exception as exc:
        logger.exception("Failed to initialize Gunicorn Arbiter.")
        event_queue.put(
            ChildEvent(
                kind="error",
                exc_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        )
        return

    runtime_error: Optional[BaseException] = None

    def _watchdog() -> None:
        start = time.monotonic()
        deadline = start + timeout

        # First, wait for arbiter.workers to get populated
        while time.monotonic() < deadline and not arbiter.WORKERS:  # type: ignore
            # If arbiter died early, abort quickly.
            if runtime_error is not None:
                logger.error("Gunicorn arbiter exited during startup. Watchdog exiting.")
                return
            time.sleep(0.1)

        if not arbiter.WORKERS:  # type: ignore
            elapsed_time = time.monotonic() - start
            logger.error("Gunicorn workers did not start within %.2f seconds.", elapsed_time)
            if runtime_error is None:
                event_queue.put(
                    ChildEvent(
                        kind="error",
                        exc_type="RuntimeError",
                        message=f"Gunicorn workers did not start within {elapsed_time:.2f} seconds.",
                        traceback=None,
                    )
                )
                # Ask arbiter to stop if it's still alive.
                # It will make the watchdog exit too.
                arbiter.halt()  # type: ignore
            else:
                logger.error("Gunicorn arbiter exited during startup. Watchdog exiting.")
            return

        # Second, check for health endpoint status if provided
        if health_url:
            while time.monotonic() < deadline:
                if runtime_error is not None:
                    logger.error("Gunicorn arbiter exited during health check. Watchdog exiting.")
                    return
                try:
                    resp = requests.get(health_url, timeout=2.0)
                    if resp.status_code == 200:
                        logger.debug(f"Server is healthy at {health_url} in {time.monotonic() - start:.2f} seconds.")
                        if runtime_error is None:
                            event_queue.put(ChildEvent(kind="ready"))
                        else:
                            logger.error("Gunicorn arbiter exited unexpectedly during health check. Watchdog exiting.")
                        return
                except Exception:
                    logger.debug(
                        f"Server is still not healthy at {health_url} in {time.monotonic() - start:.2f} seconds.",
                        exc_info=True,
                    )
                time.sleep(0.1)

            # Health failed: report and shut down.
            elapsed = time.time() - start
            logger.error(
                "Server is not healthy at %s after %.2f seconds. Shutting down.",
                health_url,
                elapsed,
            )
            if runtime_error is None:
                event_queue.put(
                    ChildEvent(
                        kind="error",
                        exc_type="RuntimeError",
                        message=(
                            f"Server is not healthy at {health_url} after "
                            f"{elapsed:.2f} seconds. It will be killed by the watchdog."
                        ),
                        traceback=None,
                    )
                )
                # Ask arbiter to stop if it's still alive.
                arbiter.halt()  # type: ignore
            else:
                logger.error("Gunicorn arbiter exited during health check. Watchdog exiting.")
            return
        else:
            # No health check; workers up => ready.
            if runtime_error is None:
                event_queue.put(ChildEvent(kind="ready"))
            else:
                logger.error("Gunicorn arbiter exited unexpectedly before health check. Watchdog exiting.")

    watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
    watchdog_thread.start()

    async def _serve() -> None:
        nonlocal runtime_error
        try:
            async with serve_context:
                arbiter.run()
        except Exception as exc:
            runtime_error = exc
            event_queue.put(
                ChildEvent(
                    kind="error",
                    exc_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                )
            )
            logger.exception("Gunicorn server failed to start.")

    try:
        asyncio.run(_serve())
        # Most exceptions should have been caught within the _serve() coroutine.
    finally:
        # Ensure watchdog doesn't try to act on a dead arbiter for long.
        watchdog_thread.join(timeout=5.0)


class PythonServerLauncher:
    """Unified launcher for FastAPI, using uvicorn or gunicorn per mode/worker count."""

    def __init__(
        self, app: FastAPI, args: PythonServerLauncherArgs, serve_context: Optional[AsyncContextManager[Any]] = None
    ):
        self.app = app
        self.args = args
        self.serve_context = serve_context
        self._port: Optional[int] = self.args.port

        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._serving_thread: Optional[threading.Thread] = None
        self._server_start_exception: Optional[BaseException] = None

        # gunicorn/uvicorn state: master inside an arbiter process; workers are forked (if gunicorn)
        self._serving_process: Optional[multiprocessing.Process] = None
        self._child_error_queue: Optional[multiprocessing.Queue[ChildEvent]] = None
        self._gunicorn_app: Optional[_GunicornApp] = None
        self._gunicorn_thread: Optional[threading.Thread] = None

    @property
    def endpoint(self) -> str:
        """The endpoint of the server."""
        return f"http://{self.args.host}:{self._port}"

    @property
    def health_url(self) -> Optional[str]:
        """The health check URL of the server, if available."""
        if self.args.healthcheck_url is not None:
            return f"{self.endpoint}{self.args.healthcheck_url}"
        return None

    def _update_endpoint(self, port: int) -> None:
        self._endpoint = f"http://{self.args.host}:{port}"

    def _ensure_port(self) -> int:
        """Ensure we have a concrete TCP port and update endpoint accordingly."""
        if self._port is None:
            self._port = pick_unused_port()
        return self._port

    @staticmethod
    def _normalize_app_ref(app: FastAPI) -> str:
        """Normalizes the app reference to a string like "module:app"."""
        # Best effort: derive "module:varname" if possible
        module = getattr(app, "__module__", None)
        if module and module != "__main__":
            # We can't reliably get the variable name; fall back to "module:app"
            # The caller should pass a string if they need something else.
            return f"{module}:app"
        return "unknown:app"

    async def start(self):
        """Starts the server."""
        logger.info(f"Starting server {self._normalize_app_ref(self.app)}...")
        if self.args.launch_mode == "mp":
            # Multi-process mode, starts a new process first
            # Then decides whether to use uvicorn or gunicorn
            await self._start_serving_process()
        elif self.args.launch_mode in ("asyncio", "thread"):
            # Always in-proc uvicorn (never subprocess)
            if self.args.launch_mode == "asyncio":
                await self._start_uvicorn_asyncio()
            else:
                await self._start_uvicorn_thread()
        else:
            raise ValueError(f"Unsupported launch mode: {self.args.launch_mode}")
        logger.info(f"Server {self._normalize_app_ref(self.app)} started at {self.endpoint}")

    async def stop(self):
        """Stops the server."""
        logger.info(f"Stopping server {self._normalize_app_ref(self.app)}...")
        if self.args.launch_mode == "mp":
            await self._stop_serving_process()
        else:
            await self._stop_uvicorn_inproc()
        logger.info(f"Server {self._normalize_app_ref(self.app)} stopped")

    async def reload(self):
        """Reloads the server."""
        if self.is_running():
            logger.info(f"Reloading running server {self._normalize_app_ref(self.app)}...")
            await self.stop()
            await self.start()
        else:
            logger.info(f"Starting server {self._normalize_app_ref(self.app)} directly because it is not running.")
            await self.start()

    async def _start_serving_process(self):
        """Starts a server in a separate process (uvicorn) or via gunicorn."""
        if self.args.n_workers > 1:
            await self._start_gunicorn()
            return

        if self._serving_process is not None and self._serving_process.is_alive():
            await self._stop_serving_process()

        port = self._ensure_port()
        self._child_error_queue = multiprocessing.Queue(maxsize=1)

        process = multiprocessing.Process(
            target=_serve_uvicorn_in_process,
            args=(self.app, self.args.host, port, self.args.log_level, self._child_error_queue),
            daemon=True,
        )
        logger.info(f"Starting uvicorn server process on port {port}...")
        process.start()
        self._serving_process = process

        if not await self._server_health_check():
            await self._stop_serving_process()
            raise RuntimeError("Server failed to start within the 10 seconds.")

    async def _stop_serving_process(self):
        """Stops the server that was started in a separate process."""
        if self.args.n_workers > 1:
            await self._stop_gunicorn()
            return

        if self._serving_process is None:
            return

        logger.info("Stopping uvicorn server process...")
        process = self._serving_process
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
            if process.is_alive():
                logger.error("Uvicorn process did not stop within timeout.")
        self._serving_process = None

    async def run_forever(self):
        """Block until the server exits."""
        mode = self.args.launch_mode

        if mode == "asyncio":
            await self._run_uvicorn_forever()
            return

        if mode == "thread":
            await self._wait_for_thread()
            return

        if mode == "mp":
            if self.args.n_workers > 1:
                await self._wait_for_gunicorn()
            else:
                await self._wait_for_process()
            return

        raise ValueError(f"Unsupported launch mode: {mode}")

    def is_running(self) -> bool:
        mode = self.args.launch_mode

        if mode == "mp":
            if self.args.n_workers > 1:
                return self._gunicorn_thread is not None and self._gunicorn_thread.is_alive()
            return self._serving_process is not None and self._serving_process.is_alive()

        return self._uvicorn_server is not None and self._uvicorn_server.started

    async def _run_uvicorn_forever(self):
        """Share the same uvicorn serving loop pattern used elsewhere in the project."""
        if self._serving_thread is not None and self._serving_thread.is_alive():
            raise RuntimeError("run_forever cannot be used while the server is already running in a thread.")

        if self._uvicorn_server is None:
            self._uvicorn_server = self._create_uvicorn_server()

        uvicorn_server = self._uvicorn_server

        async def _wait_till_healthy():
            health = await self._server_health_check()
            if not health:
                raise RuntimeError("Server did not become healthy within the 10 seconds.")
            logger.info("Server is online at %s", self.endpoint)

        async def _serve_capture():
            try:
                await uvicorn_server.serve()
            except KeyboardInterrupt:
                raise
            except (SystemExit, Exception) as exc:
                logger.debug("uvicorn serve() raised %s", exc, exc_info=exc)
                self._server_start_exception = exc
                raise RuntimeError("uvicorn server failed to serve") from exc

        try:
            await asyncio.gather(_wait_till_healthy(), _serve_capture())
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            startup_failed = not uvicorn_server.started or isinstance(
                self._server_start_exception, (SystemExit, OSError)
            )
            if startup_failed:
                self._handle_failed_start()
                raise RuntimeError(self._format_start_failure_reason())
            raise

    async def _wait_for_thread(self):
        if not await self._server_health_check():
            raise RuntimeError("Server did not become healthy within the 10 seconds.")
        logger.info("Server is online at %s", self.endpoint)
        while True:
            thread = self._serving_thread
            if thread is None or not thread.is_alive():
                if self._server_start_exception:
                    raise RuntimeError(self._format_start_failure_reason())
                return
            await asyncio.sleep(0.5)

    async def _wait_for_gunicorn(self):
        if not await self._server_health_check():
            raise RuntimeError("Server did not become healthy within the 10 seconds.")
        logger.info("Server is online at %s", self.endpoint)
        while True:
            thread = self._gunicorn_thread
            if thread is None or not thread.is_alive():
                return
            await asyncio.sleep(0.5)

    async def _wait_for_process(self):
        if not await self._server_health_check():
            raise RuntimeError("Server did not become healthy within the 10 seconds.")
        logger.info("Server is online at %s", self.endpoint)
        while True:
            process = self._serving_process
            if process is None or not process.is_alive():
                return
            await asyncio.sleep(0.5)

    # ---------------
    # Uvicorn (in-process)
    # ---------------
    def _create_uvicorn_server(self) -> uvicorn.Server:
        port = self._ensure_port()
        config = uvicorn.Config(
            app=self.app,
            host=self.args.host,
            port=port,
            log_level=self.args.log_level,
        )
        return uvicorn.Server(config)

    async def _start_uvicorn_asyncio(self):
        if self.is_running():
            await self._stop_uvicorn_inproc()

        self._uvicorn_server = self._create_uvicorn_server()
        uvicorn_server = self._uvicorn_server

        logger.info(f"Starting server at {self.endpoint}")
        self._server_start_exception = None

        def run_server_forever():
            try:
                asyncio.run(uvicorn_server.serve())
            except (SystemExit, Exception) as exc:
                logger.debug("Server thread exiting due to %s", exc, exc_info=exc)
                self._server_start_exception = exc

        self._serving_thread = threading.Thread(target=run_server_forever, daemon=True)
        self._serving_thread.start()

        # Wait for uvicorn started flag
        start_deadline = time.monotonic() + 10
        while time.monotonic() < start_deadline:
            if uvicorn_server.started:
                break
            if self._server_start_exception is not None or not self._serving_thread.is_alive():
                self._handle_failed_start()
                raise RuntimeError(self._format_start_failure_reason())
            await asyncio.sleep(0.05)
        else:
            self._handle_failed_start()
            raise RuntimeError("Server failed to start within the 10 seconds.")

        if not await self._server_health_check():
            self._handle_failed_start()
            raise RuntimeError("Server failed to start within the 10 seconds.")

        if (
            not uvicorn_server.started
            or not self._serving_thread.is_alive()
            or self._server_start_exception is not None
        ):
            self._handle_failed_start()
            raise RuntimeError(self._format_start_failure_reason())

    async def _start_uvicorn_thread(self):
        if self.is_running():
            await self._stop_uvicorn_inproc()

        self._uvicorn_server = self._create_uvicorn_server()

        def run_server():
            assert self._uvicorn_server is not None
            asyncio.run(self._uvicorn_server.serve())

        logger.info("Starting uvicorn server thread...")
        self._server_start_exception = None
        self._serving_thread = threading.Thread(target=run_server, daemon=True)
        self._serving_thread.start()

        if not await self._server_health_check():
            self._handle_failed_start()
            raise RuntimeError("Server failed to start within the 10 seconds.")

    async def _stop_uvicorn_inproc(self):
        if self._uvicorn_server is not None and self._uvicorn_server.started:
            logger.info("Stopping uvicorn server...")
            self._uvicorn_server.should_exit = True
            if self._serving_thread is not None:
                self._serving_thread.join(timeout=10.0)
                if self._serving_thread.is_alive():
                    logger.error("Uvicorn thread did not stop within timeout.")
            self._serving_thread = None
        self._uvicorn_server = None
        self._server_start_exception = None

    # ---------------
    # Gunicorn (programmatic)
    # ---------------
    async def _start_gunicorn(self):
        if self.is_running():
            await self._stop_gunicorn()

        port = self._ensure_port()
        options = {
            "bind": f"{self.args.host}:{port}",
            "workers": int(self.args.n_workers),
            # IMPORTANT: use uvicorn_worker.UvicornWorker (preferred over deprecated class)
            "worker_class": "uvicorn_worker.UvicornWorker",
            "loglevel": self._gunicorn_loglevel(),
        }

        app = self._expect_fastapi_instance(self.app)
        self._gunicorn_app = _GunicornApp(app, options)

        def run_gunicorn():
            assert self._gunicorn_app is not None
            # Blocks until stopped; safe inside a daemon thread
            self._gunicorn_app.run()

        logger.info("Starting gunicorn master thread...")
        self._gunicorn_thread = threading.Thread(target=run_gunicorn, daemon=True)
        self._gunicorn_thread.start()

        # Wait for health
        if not await self._server_health_check():
            await self._stop_gunicorn()
            raise RuntimeError("Server failed to start within the 10 seconds.")

    async def _stop_gunicorn(self):
        if self._gunicorn_app is not None:
            logger.info("Stopping gunicorn (graceful)...")
            # Ask arbiter to halt gracefully
            self._gunicorn_app.stop(graceful=True)
        if self._gunicorn_thread is not None:
            self._gunicorn_thread.join(timeout=10.0)
            if self._gunicorn_thread.is_alive():
                logger.error("Gunicorn thread did not stop within timeout.")
        self._gunicorn_app = None
        self._gunicorn_thread = None

    # ---------------
    # Helpers
    # ---------------
    async def _restart_fallback(self):
        await self.stop()
        await self.start()

    async def _server_health_check(self, timeout_s: float = 10.0) -> bool:
        if self.health_url is None:
            # If a health URL is not provided, assume success once the server reports started.
            return True

        deadline = time.monotonic() + timeout_s
        async with aiohttp.ClientSession() as session:
            while time.monotonic() < deadline:
                try:
                    async with session.get(self.health_url) as resp:
                        if resp.status == 200:
                            return True
                except Exception:
                    pass
                await asyncio.sleep(0.1)
        return False

    def _handle_failed_start(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._serving_thread is not None:
            self._serving_thread.join(timeout=0.1)
            self._serving_thread = None

    def _format_start_failure_reason(self) -> str:
        base_message = f"Server failed to start on {self.endpoint}."
        if isinstance(self._server_start_exception, SystemExit):
            return f"{base_message} Another process may already be using this port."
        if isinstance(self._server_start_exception, OSError):
            return f"{base_message} {self._server_start_exception.strerror}."
        if self._server_start_exception is not None:
            return f"{base_message} Reason: {self._server_start_exception}."
        return f"{base_message} Another process may already be using this port."

    def _gunicorn_loglevel(self) -> str:
        level_name = logging.getLevelName(self.args.log_level)
        return level_name.lower() if isinstance(level_name, str) else "info"

    @staticmethod
    def _expect_fastapi_instance(app: Union[FastAPI, str]) -> FastAPI:
        if isinstance(app, FastAPI):
            return app
        raise ValueError(
            "When using Gunicorn (mp with n_workers>1), pass a FastAPI instance to PythonServerLauncher "
            "(Gunicorn wrapper runs the object directly)."
        )
