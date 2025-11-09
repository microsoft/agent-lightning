# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import multiprocessing
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import aiohttp
import uvicorn
from fastapi import FastAPI
from gunicorn.app.base import BaseApplication
from gunicorn.arbiter import Arbiter

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


class PythonServerLauncher:
    """Unified launcher for FastAPI, using uvicorn or gunicorn per mode/worker count."""

    def __init__(self, app: FastAPI, args: PythonServerLauncherArgs):
        self.app = app
        self.args = args

        self._endpoint = f"http://{self.args.host}:{self.args.port}"

        # uvicorn in-proc state (server inside this process)
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._serving_thread: Optional[threading.Thread] = None
        self._server_start_exception: Optional[BaseException] = None

        # gunicorn/uvicorn state: master inside an arbiter process; workers are forked (if gunicorn)
        self._serving_process: Optional[multiprocessing.Process] = None

    @property
    def endpoint(self) -> str:
        """The endpoint of the server."""
        return self._endpoint

    @property
    def health_url(self) -> Optional[str]:
        """The health check URL of the server, if available."""
        if self.args.healthcheck_url is not None:
            return f"{self.endpoint}{self.args.healthcheck_url}"
        return None

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
        elif self.args.launch_mode in ("asyncio", "thread") or (
            self.args.launch_mode == "mp" and self.args.n_workers == 1
        ):
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

    async def run_forever(self):
        mode = LaunchMode(self.args.launch_mode)

        if mode == LaunchMode.ASYNCIO:
            assert self._uvicorn_server is not None

            async def _wait_till_healthy():
                if not await self._server_health_check():
                    raise RuntimeError("Server did not become healthy within the 10 seconds.")
                logger.info("Server is online at %s", self.endpoint)

            async def _serve_capture():
                try:
                    await self._uvicorn_server.serve()
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
                startup_failed = not self._uvicorn_server.started or isinstance(
                    self._server_start_exception, (SystemExit, OSError)
                )
                if startup_failed:
                    self._handle_failed_start()
                    raise RuntimeError(self._format_start_failure_reason())
                raise

        elif mode == LaunchMode.MP and self.args.n_workers > 1:
            # Gunicorn: just wait for health then block while thread is alive.
            if not await self._server_health_check():
                raise RuntimeError("Server did not become healthy within the 10 seconds.")
            logger.info("Server is online at %s", self.endpoint)
            while True:
                if self._gunicorn_thread is None or not self._gunicorn_thread.is_alive():
                    return
                await asyncio.sleep(0.5)

        else:
            # uvicorn in a thread
            if not await self._server_health_check():
                raise RuntimeError("Server did not become healthy within the 10 seconds.")
            logger.info("Server is online at %s", self.endpoint)
            while True:
                if self._serving_thread is None or not self._serving_thread.is_alive():
                    if self._server_start_exception:
                        raise RuntimeError(self._format_start_failure_reason())
                    return
                await asyncio.sleep(0.5)

    def is_running(self) -> bool:
        mode = LaunchMode(self.args.launch_mode)

        if mode == LaunchMode.MP and self.args.n_workers > 1:
            return self._gunicorn_thread is not None and self._gunicorn_thread.is_alive()
        else:
            return self._uvicorn_server is not None and self._uvicorn_server.started

    # ---------------
    # Uvicorn (in-process)
    # ---------------
    async def _start_uvicorn_asyncio(self):
        if self.is_running():
            await self._stop_uvicorn_inproc()

        config = uvicorn.Config(
            app=self.args.app,
            host=self.args.host,
            port=self.args.port,
            log_level=self.args.log_level,
            access_log=self.args.uvicorn_access_log,
            timeout_keep_alive=self.args.uvicorn_timeout_keep_alive,
        )
        self._uvicorn_server = uvicorn.Server(config)

        logger.info(f"Starting server at {self.endpoint}")
        self._server_start_exception = None

        def run_server_forever():
            try:
                asyncio.run(self._uvicorn_server.serve())
            except (SystemExit, Exception) as exc:
                logger.debug("Server thread exiting due to %s", exc, exc_info=exc)
                self._server_start_exception = exc

        serving_thread = threading.Thread(target=run_server_forever, daemon=True)
        self._serving_thread = serving_thread
        serving_thread.start()

        # Wait for uvicorn started flag
        start_deadline = time.time() + 10
        while time.time() < start_deadline:
            if self._uvicorn_server.started:
                break
            if self._server_start_exception is not None or not serving_thread.is_alive():
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
            not self._uvicorn_server.started
            or not serving_thread.is_alive()
            or self._server_start_exception is not None
        ):
            self._handle_failed_start()
            raise RuntimeError(self._format_start_failure_reason())

    async def _start_uvicorn_thread(self):
        if self.is_running():
            await self._stop_uvicorn_inproc()

        config = uvicorn.Config(
            app=self.args.app,
            host=self.args.host,
            port=self.args.port,
            log_level=self.args.log_level,
            access_log=self.args.uvicorn_access_log,
            timeout_keep_alive=self.args.uvicorn_timeout_keep_alive,
        )
        self._uvicorn_server = uvicorn.Server(config)

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

        # Validate imports only when needed
        try:
            import gunicorn  # noqa: F401
            import uvicorn_worker  # noqa: F401
        except Exception as exc:
            raise RuntimeError("mp mode with n_workers>1 requires `gunicorn` and `uvicorn-worker` packages.") from exc

        options = {
            "bind": f"{self.args.host}:{self.args.port}",
            "workers": int(self.args.n_workers),
            # IMPORTANT: use uvicorn_worker.UvicornWorker (preferred over deprecated class)
            "worker_class": "uvicorn_worker.UvicornWorker",
            "preload_app": bool(self.args.gunicorn_preload_app),
            "timeout": int(self.args.gunicorn_timeout),
            "graceful_timeout": int(self.args.gunicorn_graceful_timeout),
            "backlog": int(self.args.gunicorn_backlog),
            "accesslog": self.args.gunicorn_accesslog,
            "errorlog": self.args.gunicorn_errorlog,
            "loglevel": self.args.log_level,
        }

        app = self._expect_fastapi_instance(self.args.app)
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
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self._health_url) as resp:
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

    @staticmethod
    def _expect_fastapi_instance(app: Union[FastAPI, str]) -> FastAPI:
        if isinstance(app, FastAPI):
            return app
        raise ValueError(
            "When using Gunicorn (mp with n_workers>1), pass a FastAPI instance to PythonServerLauncherArgs.app "
            "(Gunicorn wrapper runs the object directly)."
        )
