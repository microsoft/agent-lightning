# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import inspect
import logging
import multiprocessing
import queue
import signal
import socket
import threading
import time
import traceback
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Any, AsyncContextManager, AsyncIterator, Dict, Literal, Optional, cast

import aiohttp
import requests
import uvicorn
from fastapi import FastAPI
from gunicorn.app.base import BaseApplication
from gunicorn.arbiter import Arbiter
from portpicker import pick_unused_port

__all__ = ["PythonServerLauncher", "PythonServerLauncherArgs", "LaunchMode"]


LaunchMode = Literal["asyncio", "thread", "mp"]
"""The launch mode for the server."""


@dataclass
class PythonServerLauncherArgs:
    port: Optional[int] = None
    """The TCP port to listen on. If not provided, the server will use a random available port."""
    host: Optional[str] = None
    """The hostname or IP address to bind the server to."""
    access_host: Optional[str] = None
    """The hostname or IP address to advertise to the client. If not provided, the server will use the default outbound IPv4 address for this machine."""
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
    access_log: bool = False
    """Whether to turn on access logs."""
    startup_timeout: float = 60.0
    """The timeout to wait for the server to start up."""
    kill_unhealthy_server: bool = True
    """Whether to kill the server if it is not healthy after startup.
    This setting is ignored when `launch_mode` is not `asyncio`.
    """
    thread_join_timeout: float = 10.0
    """The timeout to wait for the thread to join."""
    process_join_timeout: float = 10.0
    """The timeout to wait for the process to join."""
    timeout_keep_alive: int = 30
    """The timeout to keep the connection alive."""


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
        self.application = app
        self.options = options
        super().__init__()  # type: ignore

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


@asynccontextmanager
async def noop_context() -> AsyncIterator[None]:
    """A real async context manager that does nothing (satisfies serve_context)."""
    yield


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
        start_time = time.time()
        deadline = start_time + timeout  # child-side startup window
        logger.debug(f"Waiting for server to start up for {timeout:.2f} seconds...")
        # Wait for the server to start up or the deadline to be reached, or an exception to be raised.
        while time.time() < deadline and not uvicorn_server.started and server_start_exception is None:
            await asyncio.sleep(0.1)

        if not uvicorn_server.started:
            # Normally, the program will not reach this point, as the server will throw the exception itself earlier.
            raise RuntimeError(
                f"Server did not start up within {time.time() - start_time:.2f} seconds."
            ) from server_start_exception

        logger.info(f"Server started up in {time.time() - start_time:.2f} seconds.")

        # Check for health endpoint status if provided
        if health_url is not None:
            logger.info(f"Probing health endpoint {health_url}...")
            async with aiohttp.ClientSession() as session:
                while time.time() < deadline:
                    try:
                        async with session.get(health_url) as resp:
                            if resp.status == 200:
                                logger.info(
                                    f"Server is healthy at {health_url} in {time.time() - start_time:.2f} seconds."
                                )
                                return
                            else:
                                logger.debug(
                                    f"Server is NOT healthy at {health_url} in {time.time() - start_time:.2f} seconds. Got status {resp.status}."
                                )
                    except Exception as e:
                        logger.debug(f"Error probing health endpoint {health_url}: {str(e)}")
                    await asyncio.sleep(0.1)

            # If the server is not healthy, kill it if requested.
            health_failed_seconds = time.time() - start_time
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
            logger.info("Server does not provide a health check endpoint. Skipping health check.")

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

    serve_task = asyncio.create_task(_serve_server())
    watch_task = asyncio.create_task(_watch_server())

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
    - Post `ChildEvent(kind="ready")` once the server is up.
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
    - On success: put `ChildEvent(kind="ready")`.
    - On failure/timeout: put `ChildEvent(kind="error")` and request a graceful shutdown.

    `serve_context` will be applied around the `arbiter.run()` call.
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
        start = time.time()
        deadline = start + timeout

        # First, wait for arbiter.workers to get populated
        while time.time() < deadline and not arbiter.WORKERS:  # type: ignore
            # If arbiter died early, abort quickly.
            if runtime_error is not None:
                logger.error("Gunicorn arbiter exited during startup. Watchdog exiting.")
                return
            time.sleep(0.1)

        if not arbiter.WORKERS:  # type: ignore
            elapsed_time = time.time() - start
            logger.error("Gunicorn workers did not start within %.2f seconds.", elapsed_time)
            if runtime_error is None:
                # Timeout case: arbiter throws no exception.
                event_queue.put(
                    ChildEvent(
                        kind="error",
                        exc_type="RuntimeError",
                        message=f"Gunicorn workers did not start within {elapsed_time:.2f} seconds.",
                        traceback=None,
                    )
                )
                logger.info("Halting Gunicorn arbiter.")
                # Ask arbiter to stop if it's still alive.
                # It will make the watchdog exit too.
                arbiter.signal(signal.SIGTERM, inspect.currentframe())  # type: ignore
            else:
                # Timeout case: arbiter has thrown an exception.
                logger.error("Gunicorn arbiter exited during startup. Watchdog exiting.")
            return

        # Second, check for health endpoint status if provided
        if health_url:
            while time.time() < deadline:
                # If arbiter died early, abort.
                if runtime_error is not None:
                    logger.error("Gunicorn arbiter exited during health check. Watchdog exiting.")
                    return

                # Check if the server is healthy.
                try:
                    resp = requests.get(health_url, timeout=2.0)
                    if resp.status_code == 200:
                        logger.debug(f"Server is healthy at {health_url} in {time.time() - start:.2f} seconds.")
                        # Check arbiter status again.
                        if runtime_error is None:
                            event_queue.put(ChildEvent(kind="ready"))
                        else:
                            logger.error(
                                "Response status is 200 but arbiter has thrown an exception. This should not happen."
                            )
                        return
                except Exception:
                    logger.debug(
                        f"Server is still not healthy at {health_url} in {time.time() - start:.2f} seconds.",
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
                # Arbiter throws no exception. This is a simple timeout case.
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
                logger.info("Halting Gunicorn arbiter.")
                # Ask arbiter to stop if it's still alive.
                arbiter.signal(signal.SIGTERM, inspect.currentframe())  # type: ignore
            else:
                # If arbiter has thrown an exception, report it.
                logger.error("Gunicorn arbiter exited during health check. Watchdog exiting.")

        else:
            # No health check; workers up => ready.
            if runtime_error is None:
                event_queue.put(ChildEvent(kind="ready"))
            else:
                # If arbiter has thrown an exception, report it.
                logger.error("Gunicorn arbiter exited unexpectedly before health check. Watchdog exiting.")

    def _watchdog_with_exception() -> None:
        try:
            _watchdog()
        except Exception as exc:
            logger.exception("Exception in watchdog thread.")
            event_queue.put(
                ChildEvent(
                    kind="error", exc_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()
                )
            )

    watchdog_thread = threading.Thread(target=_watchdog_with_exception, daemon=True)
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


def _get_default_ipv4_address() -> str:
    """Determine the default outbound IPv4 address for this machine.

    Implementation:
        Opens a UDP socket and "connects" to a public address to force route
        selection, then inspects the socket's local address. No packets are sent.

    Returns:
        str: Best-guess IPv4 like `192.168.x.y`. Falls back to `127.0.0.1`.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually contact 8.8.8.8; just forces the OS to pick a route.
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


class PythonServerLauncher:
    """Unified launcher for FastAPI, using uvicorn or gunicorn per mode/worker count.

    See [`PythonServerLauncherArgs`][agentlightning.utils.server_launcher.PythonServerLauncherArgs] for configuration options.

    Args:
        app: The FastAPI app to launch.
        args: The configuration for the server.
        serve_context: An optional context manager to apply around the server startup.
    """

    def __init__(
        self, app: FastAPI, args: PythonServerLauncherArgs, serve_context: Optional[AsyncContextManager[Any]] = None
    ):
        """Initialize the launcher with the FastAPI app, configuration, and optional serve context."""
        self.app = app
        self.args = args
        self.serve_context = serve_context
        self._host: Optional[str] = self.args.host
        self._port: Optional[int] = self.args.port
        self._access_host: Optional[str] = self.args.access_host
        self.initialize()

    def initialize(self):
        # ensure the host/port/access_host are set
        self._ensure_host()
        self._ensure_port()
        self._ensure_access_host()

        # uvicorn (in-proc asyncio)
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._uvicorn_task: Optional[asyncio.Task[None]] = None  # returned by run_uvicorn_asyncio()

        # uvicorn (thread)
        self._thread: Optional[threading.Thread] = None
        self._thread_event_queue: Optional[queue.Queue[ChildEvent]] = None
        self._thread_stop_event: Optional[threading.Event] = None

        # subprocess (uvicorn / gunicorn)
        self._proc: Optional[BaseProcess] = None
        self._mp_event_queue: Optional[multiprocessing.Queue[ChildEvent]] = None
        self._gunicorn_app: Optional[GunicornApp] = None  # programmatic gunicorn wrapper

        # is_running flag
        self._is_running: bool = False

    def __getstate__(self):
        """Control pickling to prevent server state from being sent to subprocesses."""
        return {
            "app": self.app,
            "args": self.args,
            "serve_context": self.serve_context,
            "_host": self._host,
            "_port": self._port,
            "_access_host": self._access_host,
        }

    def __setstate__(self, state: Dict[str, Any]):
        self.app = state["app"]
        self.args = cast(PythonServerLauncherArgs, state["args"])
        self.serve_context = state["serve_context"]
        self._host = state["_host"]
        self._port = state["_port"]
        self._access_host = state["_access_host"]
        self.initialize()

    @property
    def endpoint(self) -> str:
        """Return the externally advertised host:port pair regardless of accessibility."""
        return f"http://{self._ensure_host()}:{self._ensure_port()}"

    @property
    def access_endpoint(self) -> str:
        """Return a loopback-friendly URL so health checks succeed even when binding to 0.0.0.0."""
        return f"http://{self._ensure_access_host()}:{self._ensure_port()}"

    @property
    def health_url(self) -> Optional[str]:
        """Build the absolute health-check endpoint from args, if one is configured."""
        if not self.args.healthcheck_url:
            return None
        path = self.args.healthcheck_url
        if not path.startswith("/"):
            path = "/" + path
        return f"{self.access_endpoint}{path}"

    async def start(self):
        """Starts the server according to launch_mode and n_workers."""
        logger.info(f"Starting server {self._normalize_app_ref(self.app)}...")
        mode = self.args.launch_mode
        if mode == "mp":
            await self._start_serving_process()
        elif mode == "thread":
            await self._start_uvicorn_thread()
        elif mode == "asyncio":
            await self._start_uvicorn_asyncio()
        else:
            raise ValueError(f"Unsupported launch mode: {mode}")
        logger.info(f"Server {self._normalize_app_ref(self.app)} started at {self.endpoint}")

    async def stop(self):
        """Stop the server using the inverse of whatever launch mode was used to start it."""
        logger.info(f"Stopping server {self._normalize_app_ref(self.app)}...")
        mode = self.args.launch_mode
        if mode == "mp":
            await self._stop_serving_process()
        elif mode == "thread":
            await self._stop_uvicorn_thread()
        elif mode == "asyncio":
            await self._stop_uvicorn_asyncio()
        else:
            raise ValueError(f"Unsupported launch mode: {mode}")
        logger.info(f"Server {self._normalize_app_ref(self.app)} stopped")

    async def reload(self):
        """Restart the server by stopping it if necessary and invoking start again."""
        if self.is_running():
            await self.stop()
        await self.start()

    async def run_forever(self):
        """Start the server and block the caller until it exits, respecting the configured mode."""
        mode = self.args.launch_mode
        if mode == "asyncio":
            await self._start_uvicorn_asyncio()
            try:
                if self._uvicorn_task is not None:
                    # Wait for the server
                    # Won't allow outer cancel to directly cancel the inner task
                    await asyncio.shield(self._uvicorn_task)
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.warning("Server received cancellation signal. Shutting down gracefully.")
                await self._stop_uvicorn_asyncio()
                raise

        elif mode == "thread":
            await self._start_uvicorn_thread()
            try:
                # Wait for the thread to exit
                while self._thread and self._thread.is_alive():
                    await asyncio.sleep(0.5)
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.warning("Server thread received cancellation signal. Shutting down gracefully.")
                await self._stop_uvicorn_thread()
                raise

        elif mode == "mp":
            await self._start_serving_process()
            try:
                # Wait for the process to exit
                while self._proc and self._proc.is_alive():
                    await asyncio.sleep(0.5)
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.warning("Server process received cancellation signal. Shutting down gracefully.")
                await self._stop_serving_process()
                raise

        else:
            raise ValueError(f"Unsupported launch mode: {mode}")

    def is_running(self) -> bool:
        """Return True if the server has been started and not yet stopped."""
        return self._is_running

    @staticmethod
    def _normalize_app_ref(app: FastAPI) -> str:
        module = getattr(app, "__module__", None)
        if module and module != "__main__":
            return f"{module}:app"
        return "unknown:app"

    def _ensure_host(self) -> str:
        if self._host is None:
            logger.warning("No host provided, using 0.0.0.0.")
            self._host = "0.0.0.0"
        return self._host

    def _ensure_port(self) -> int:
        if self._port is None:
            logger.warning("No port provided, using pick_unused_port to pick a random unused port.")
            self._port = pick_unused_port()
        return self._port

    def _ensure_access_host(self) -> str:
        if self._access_host is None:
            if self.args.access_host is None:
                if self._ensure_host() in ("0.0.0.0", "::"):
                    # Probe host normalization for 0.0.0.0
                    logger.warning("No access host provided, using default outbound IPv4 address for this machine.")
                    self._access_host = _get_default_ipv4_address()
                else:
                    logger.warning("No access host provided, using the host provided.")
                    self._access_host = self._ensure_host()
            else:
                self._access_host = self.args.access_host
        return self._access_host  # type: ignore

    def _create_uvicorn_server(self) -> uvicorn.Server:
        config = uvicorn.Config(
            app=self.app,
            host=self._ensure_host(),
            port=self._ensure_port(),
            log_level=self.args.log_level,
            access_log=self.args.access_log,
            loop="asyncio",
            timeout_keep_alive=self.args.timeout_keep_alive,
        )
        return uvicorn.Server(config)

    def _ctx(self) -> AsyncContextManager[Any]:
        # Use the provided serve_context if any; otherwise a no-op async CM
        if self.serve_context is None:
            logger.info("No serve_context provided, using noop_context.")
            return noop_context()
        return self.serve_context

    # --- Mode 1: asyncio (in-proc) using run_uvicorn_asyncio ---

    async def _start_uvicorn_asyncio(self):
        if self.is_running():
            raise RuntimeError("Server is already running. Stopping it first.")

        logger.info("Starting uvicorn asyncio server...")
        self._uvicorn_server = self._create_uvicorn_server()
        # Start server; return after health passes; keep serving in background task
        self._uvicorn_task = await run_uvicorn_asyncio(
            uvicorn_server=self._uvicorn_server,
            serve_context=self._ctx(),
            timeout=self.args.startup_timeout,
            health_url=self.health_url,
            wait_for_serve=False,  # return once startup/health OK
            kill_unhealthy_server=self.args.kill_unhealthy_server,
        )
        self._is_running = True
        logger.info("Uvicorn asyncio server started")

    async def _stop_uvicorn_asyncio(self):
        # Gracefully shut down the in-proc uvicorn server task if running
        logger.info("Stopping uvicorn asyncio server...")
        if self._uvicorn_server and self._uvicorn_task:
            await shutdown_uvicorn_server(self._uvicorn_server, self._uvicorn_task)
        self._uvicorn_task = None
        self._uvicorn_server = None
        self._is_running = False
        logger.info("Uvicorn asyncio server stopped")

    # --- Mode 2: thread (in-proc) using run_uvicorn_thread ---

    async def _start_uvicorn_thread(self):
        if self.is_running():
            raise RuntimeError("Server is already running. Stopping it first.")

        logger.info("Starting uvicorn thread server...")
        self._uvicorn_server = self._create_uvicorn_server()
        self._thread_event_queue = queue.Queue()
        self._thread_stop_event = threading.Event()

        self._thread = threading.Thread(
            target=run_uvicorn_thread,
            kwargs={
                "uvicorn_server": self._uvicorn_server,
                "serve_context": self._ctx(),
                "event_queue": self._thread_event_queue,
                "stop_event": self._thread_stop_event,
                "timeout": self.args.startup_timeout,
                "health_url": self.health_url,
            },
            daemon=True,
        )
        self._thread.start()

        # Wait for ready or error event from the thread
        timeout = self.args.startup_timeout * 2  # Allows twice the timeout for the thread to get the event
        try:
            evt: ChildEvent = await asyncio.to_thread(self._thread_event_queue.get, True, timeout)
        except queue.Empty:
            if not self._thread.is_alive():
                raise RuntimeError("Threaded server failed to start and is not alive. No error event was received.")
            logger.error(
                "Threaded server failed to start and sends no event. This should not happen. Shutting down server."
            )
            await self._stop_uvicorn_thread()
            raise RuntimeError("Threaded server failed to start and sends no event. This should not happen.")

        if evt.kind == "error":
            logger.error("Threaded server failed to start (%s): %s\n%s", evt.exc_type, evt.message, evt.traceback)
            await asyncio.to_thread(self._thread.join, self.args.thread_join_timeout)
            if self._thread.is_alive():
                logger.error("Threaded server failed to start and refused to shut down.")
            raise RuntimeError(evt.message)
        else:
            logger.info("Threaded server started successfully.")
            self._is_running = True

    async def _stop_uvicorn_thread(self):
        logger.info("Stopping uvicorn thread server...")
        if self._thread_stop_event:
            self._thread_stop_event.set()
        if self._thread:
            await asyncio.to_thread(self._thread.join, self.args.thread_join_timeout)
            if self._thread.is_alive():
                raise RuntimeError("Threaded server refused to shut down.")
        else:
            logger.info("Uvicorn thread server was not running. Nothing to stop.")

        self._thread = None
        self._thread_event_queue = None
        self._thread_stop_event = None
        self._uvicorn_server = None
        self._is_running = False
        logger.info("Uvicorn thread server stopped")

    # --- Mode 3: subprocess (uvicorn / gunicorn) using run_uvicorn_subprocess or run_gunicorn ---

    async def _start_serving_process(self):
        if self.is_running():
            raise RuntimeError("Server process is already running. Stopping it first.")

        host = self._ensure_host()
        port = self._ensure_port()

        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError as e:
            raise RuntimeError(
                "Process launch requires 'fork' start method (Linux/macOS). "
                "On Windows, use 'thread' or 'asyncio' modes."
            ) from e
        self._mp_event_queue = ctx.Queue()

        # Gunicorn path when n_workers > 1
        if self.args.n_workers > 1:
            logger.info(f"Starting Gunicorn server...")
            options = {
                "bind": f"{host}:{port}",
                "workers": int(self.args.n_workers),
                "worker_class": "uvicorn_worker.UvicornWorker",
                "loglevel": logging.getLevelName(self.args.log_level).lower(),
                "accesslog": "-" if self.args.access_log else None,
                "errorlog": "-",
                "preload_app": True,
                "graceful_timeout": int(
                    self.args.process_join_timeout / 2
                ),  # Allow half the timeout for graceful shutdown
            }
            self._gunicorn_app = GunicornApp(self.app, options)

            self._proc = ctx.Process(
                target=run_gunicorn,
                kwargs={
                    "gunicorn_app": self._gunicorn_app,
                    "serve_context": self._ctx(),
                    "event_queue": self._mp_event_queue,
                    "timeout": self.args.startup_timeout,
                    "health_url": self.health_url,
                },
                daemon=False,
            )
            self._proc.start()

        else:
            # Single-worker subprocess uvicorn
            logger.info("Starting uvicorn subprocess server...")
            self._uvicorn_server = self._create_uvicorn_server()

            self._proc = ctx.Process(
                target=run_uvicorn_subprocess,
                kwargs={
                    "uvicorn_server": self._uvicorn_server,
                    "serve_context": self._ctx(),
                    "event_queue": self._mp_event_queue,
                    "timeout": self.args.startup_timeout,
                    "health_url": self.health_url,
                },
                daemon=True,
            )
            self._proc.start()

        # Wait for ready or error event from the thread
        timeout = self.args.startup_timeout * 2  # Allows twice the timeout for the thread to get the event
        try:
            evt: ChildEvent = await asyncio.to_thread(self._mp_event_queue.get, True, timeout)
        except queue.Empty:
            if not self._proc.is_alive():
                raise RuntimeError("Server process failed to start and is not alive. No error event was received.")
            logger.error(
                "Server process failed to start and sends no event. This should not happen. Shutting down server."
            )
            await self._stop_serving_process()
            raise RuntimeError("Server process failed to start and sends no event. This should not happen.")

        if evt.kind == "error":
            logger.error(
                "Server process (%s) failed to start (%s): %s\n%s",
                "gunicorn" if self.args.n_workers > 1 else "uvicorn",
                evt.exc_type,
                evt.message,
                evt.traceback,
            )
            await asyncio.to_thread(self._proc.join, self.args.process_join_timeout)
            if self._proc.is_alive():
                logger.error("Server process failed to start and refused to shut down.")
            raise RuntimeError(evt.message)
        else:
            logger.info("Subprocess server started successfully.")
            self._is_running = True

    async def _stop_serving_process(self):
        logger.info("Stopping subprocess server...")
        if self._proc is not None:
            if self._proc.is_alive():
                # Prefer graceful: SIGTERM, then wait
                try:
                    self._proc.terminate()
                except Exception:
                    logger.exception("Error sending SIGTERM to server process.")
                await asyncio.to_thread(self._proc.join, self.args.process_join_timeout)

            if self._proc.is_alive():
                # Still alive, send SIGKILL
                try:
                    self._proc.kill()
                except Exception:
                    logger.exception("Error sending SIGKILL to server process.")
                await asyncio.to_thread(self._proc.join, 5.0)  # Use a constant timeout for SIGKILL

            if self._proc.is_alive():
                raise RuntimeError("Server process failed to shut down after SIGTERM and SIGKILL.")
        else:
            logger.info("Subprocess server was not running. Nothing to stop.")

        if self._mp_event_queue is not None:
            self._mp_event_queue.close()
            try:
                self._mp_event_queue.join_thread()
            except Exception:
                logger.exception("Error joining event queue thread.")

        self._proc = None
        self._mp_event_queue = None
        self._gunicorn_app = None
        self._uvicorn_server = None
        self._is_running = False
        logger.info("Subprocess server stopped")
