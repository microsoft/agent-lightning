# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import socket
import threading
import time
import traceback
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncContextManager, Dict, Literal, Optional, Union

import aiohttp
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


class _GunicornApp(BaseApplication):
    """
    Programmatic Gunicorn application that:

    - Stores its Arbiter to allow `stop()`/`reload()` from caller.
    - Accepts a `FastAPI` app object and option dict.
    - Uses `uvicorn_worker.UvicornWorker`.
    """

    def __init__(self, app: FastAPI, options: Dict[str, Any]):
        super().__init__()  # type: ignore
        self.application = app
        self.options = options or {}
        self._arbiter: Optional[Arbiter] = None

    def load_config(self):
        cfg = self.cfg
        valid_keys = cfg.settings.keys()  # type: ignore
        for k, v in (self.options or {}).items():
            if k in valid_keys and v is not None:
                cfg.set(k, v)  # type: ignore

    def load(self):
        return self.application


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
        while time.time() < deadline and not uvicorn_server.started and server_start_exception is None:
            await asyncio.sleep(0.1)

        if not uvicorn_server.started:
            # Normally, the program will not reach this point, as the server will throw the exception itself earlier.
            raise RuntimeError(f"Server did not start up within {timeout:.2f} seconds.") from server_start_exception

        logger.debug(f"Server started up in {time.time() - start_time:.2f} seconds.")

        # Check for health endpoint status if provided
        if health_url is not None:
            async with aiohttp.ClientSession() as session:
                while time.time() < deadline:
                    with suppress(Exception):
                        async with session.get(health_url) as resp:
                            if resp.status == 200:
                                logger.debug(
                                    f"Server is healthy at {health_url} in {time.time() - start_time:.2f} seconds."
                                )
                                return
                    await asyncio.sleep(0.1)

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
                raise RuntimeError(
                    f"Server is not healthy at {health_url} after {health_failed_seconds:.2f} seconds. It has been left running."
                )

        else:
            logger.debug("Server does not provide a health check endpoint. Skipping health check.")

    async def _serve_server() -> None:
        nonlocal server_start_exception
        async with serve_context:
            try:
                await uvicorn_server.serve()
            except BaseException as exc:  # including KeyboardInterrupt
                server_start_exception = exc
                # This probably sends out earlier than watcher exception; but either one is fine.
                raise RuntimeError("Uvicorn server failed to serve") from exc

    watch_task = asyncio.create_task(_watch_server())
    serve_task = asyncio.create_task(_serve_server())

    if wait_for_serve:
        await asyncio.gather(watch_task, serve_task)
    else:
        # Wait for watch only, the serve task will run in the background.
        await watch_task
    return serve_task


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
        start_deadline = time.time() + 10
        while time.time() < start_deadline:
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

        deadline = time.time() + timeout_s
        async with aiohttp.ClientSession() as session:
            while time.time() < deadline:
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
