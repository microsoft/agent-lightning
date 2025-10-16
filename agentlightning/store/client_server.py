# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Sequence, Union

import aiohttp
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    Span,
    TaskInput,
)

from .base import UNSET, LightningStore, Unset

logger = logging.getLogger(__name__)


class PydanticUnset(BaseModel):
    _type: Literal["UNSET"] = "UNSET"


class RolloutRequest(BaseModel):
    input: TaskInput
    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None
    config: Optional[RolloutConfig] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryRolloutsRequest(BaseModel):
    status: Optional[List[RolloutStatus]] = None
    rollout_ids: Optional[List[str]] = None


class WaitForRolloutsRequest(BaseModel):
    rollout_ids: List[str]
    timeout: Optional[float] = None


class RolloutId(BaseModel):
    rollout_id: str


class AddResourcesRequest(BaseModel):
    resources: NamedResources


class UpdateRolloutRequest(BaseModel):
    rollout_id: str
    input: Union[TaskInput, PydanticUnset] = Field(default_factory=PydanticUnset)
    mode: Union[Optional[Literal["train", "val", "test"]], PydanticUnset] = Field(default_factory=PydanticUnset)
    resources_id: Union[Optional[str], PydanticUnset] = Field(default_factory=PydanticUnset)
    status: Union[RolloutStatus, PydanticUnset] = Field(default_factory=PydanticUnset)
    config: Union[RolloutConfig, PydanticUnset] = Field(default_factory=PydanticUnset)
    metadata: Union[Dict[str, Any], PydanticUnset] = Field(default_factory=PydanticUnset)


class UpdateAttemptRequest(BaseModel):
    rollout_id: str
    attempt_id: Union[str, Literal["latest"]]
    status: Union[AttemptStatus, PydanticUnset] = Field(default_factory=PydanticUnset)
    worker_id: Union[str, PydanticUnset] = Field(default_factory=PydanticUnset)
    last_heartbeat_time: Union[float, PydanticUnset] = Field(default_factory=PydanticUnset)
    metadata: Union[Dict[str, Any], PydanticUnset] = Field(default_factory=PydanticUnset)


class LightningStoreServer(LightningStore):
    """
    Server wrapper that exposes a LightningStore via HTTP API.
    Delegates all operations to an underlying store implementation.

    Healthcheck and watchdog relies on the underlying store.
    """

    def __init__(self, store: LightningStore, host: str, port: int):
        super().__init__()
        self.store = store
        self.host = host
        self.port = port
        self.app: FastAPI | None = FastAPI(title="LightningStore Server")
        self._setup_routes()
        self._uvicorn_config: uvicorn.Config | None = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="error"
        )
        self._uvicorn_server: uvicorn.Server | None = uvicorn.Server(self._uvicorn_config)

        self._serving_thread: Optional[threading.Thread] = None

        # Process-awareness:
        # LightningStoreServer holds a plain Python object (self.store) in one process
        # (the process that runs uvicorn/FastAPI).
        # When you multiprocessing.Process(...) and call methods on a different LightningStore instance
        # (or on a copy inherited via fork), you’re mutating another process’s memory, not the server’s memory.
        # So we need to track the owner process (whoever creates the server),
        # and only mutate the store in that process.
        self._owner_pid = os.getpid()
        self._client: Optional[LightningStoreClient] = None

    def __getstate__(self):
        """
        Control pickling to prevent server state from being sent to subprocesses.

        When LightningStoreServer is pickled (e.g., passed to a subprocess), we only
        serialize the underlying store and connection details. The FastAPI app and
        uvicorn server are excluded as they should not be transferred between processes.

        The subprocess should create its own server instance if needed.
        """
        return {
            "store": self.store,
            "host": self.host,
            "port": self.port,
            "_owner_pid": self._owner_pid,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore from pickle by reconstructing only the essential attributes.

        Note: This creates a new server instance without FastAPI/uvicorn initialized.
        Call __init__() pattern or create a new LightningStoreServer if you need
        a fully functional server in the subprocess.
        """
        self.store = state["store"]
        self.host = state["host"]
        self.port = state["port"]
        self._owner_pid = state["_owner_pid"]
        self._client = None
        # Do NOT reconstruct app, _uvicorn_config, _uvicorn_server
        # to avoid transferring server state to subprocess

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Starts the FastAPI server in the background.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None
        logger.info(f"Starting server at {self.endpoint}")

        uvicorn_server = self._uvicorn_server

        def run_server_forever():
            asyncio.run(uvicorn_server.serve())

        self._serving_thread = threading.Thread(target=run_server_forever, daemon=True)
        self._serving_thread.start()

        # Wait for /health to be available
        if not await self._server_health_check():
            raise RuntimeError("Server failed to start within the 10 seconds.")

    async def _server_health_check(self) -> bool:
        """Checks if the server is healthy."""
        current_time = time.time()
        while time.time() - current_time < 10:
            async with aiohttp.ClientSession() as session:
                with suppress(Exception):
                    async with session.get(f"{self.endpoint}/health") as response:
                        if response.status == 200:
                            return True
            await asyncio.sleep(0.1)
        return False

    async def run_forever(self):
        """Runs the FastAPI server indefinitely.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None

        async def _wait_till_healthy():
            health = await self._server_health_check()
            if not health:
                raise RuntimeError("Server did not become healthy within the 10 seconds.")
            logger.info("Store server is online at %s", self.endpoint)

        # We run _wait_till_healthy and self._uvicorn_server.serve in parallel
        # until one of them raises an exception.
        await asyncio.gather(_wait_till_healthy(), self._uvicorn_server.serve())

    async def stop(self):
        """Gracefully stops the running FastAPI server.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None
        if self._uvicorn_server.started:
            logger.info("Stopping server...")
            self._uvicorn_server.should_exit = True
            if self._serving_thread is not None:
                self._serving_thread.join(timeout=10)
            self._serving_thread = None
            logger.info("Server stopped.")

    def _backend(self) -> LightningStore:
        """Returns the object to delegate to in *this* process.

        - In the owner process: delegate to the in-process store.
        - In a different process: delegate to a HTTP client talking to the server.
        """
        if os.getpid() == self._owner_pid:
            return self.store
        if self._client is None:
            self._client = LightningStoreClient(self.endpoint)
        return self._client

    def _setup_routes(self):
        """Set up FastAPI routes for all store operations."""
        assert self.app is not None

        @self.app.exception_handler(Exception)
        async def _app_exception_handler(request: Request, exc: Exception):  # pyright: ignore[reportUnusedFunction]
            """
            Convert unhandled application exceptions into 400 responses.

            - Client needs a reliable signal to distinguish "app bug / bad request"
              from transport/session failures.
            - 400 here means "do not retry"; network issues will surface as aiohttp
              exceptions or 5xx and will be retried by the client shield.
            """
            logger.exception("Unhandled application error", exc_info=exc)
            return JSONResponse(
                status_code=400,
                content={
                    "detail": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )

        @self.app.middleware("http")
        async def _log_time(  # pyright: ignore[reportUnusedFunction]
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ):
            start = time.perf_counter()
            response = await call_next(request)
            duration = (time.perf_counter() - start) * 1000
            client = request.client
            if client is None:
                client_address = "unknown"
            else:
                client_address = f"{client.host}:{client.port}"
            logger.info(
                f"{client_address} - "
                f'"{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" '
                f"{response.status_code} in {duration:.2f} ms"
            )
            return response

        @self.app.get("/health")
        async def health():  # pyright: ignore[reportUnusedFunction]
            return {"status": "ok"}

        @self.app.post("/start_rollout", response_model=AttemptedRollout)
        async def start_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.start_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @self.app.post("/enqueue_rollout", response_model=Rollout)
        async def enqueue_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.enqueue_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @self.app.get("/dequeue_rollout", response_model=Optional[AttemptedRollout])
        async def dequeue_rollout():  # pyright: ignore[reportUnusedFunction]
            return await self.store.dequeue_rollout()

        @self.app.post("/start_attempt", response_model=AttemptedRollout)
        async def start_attempt(request: RolloutId):  # pyright: ignore[reportUnusedFunction]
            return await self.store.start_attempt(request.rollout_id)

        @self.app.post("/query_rollouts", response_model=List[Rollout])
        async def query_rollouts(request: QueryRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.query_rollouts(status=request.status)

        @self.app.get("/query_attempts/{rollout_id}", response_model=List[Attempt])
        async def query_attempts(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.query_attempts(rollout_id)

        @self.app.get("/get_latest_attempt/{rollout_id}", response_model=Optional[Attempt])
        async def get_latest_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_latest_attempt(rollout_id)

        @self.app.get("/get_rollout_by_id/{rollout_id}", response_model=Optional[Rollout])
        async def get_rollout_by_id(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_rollout_by_id(rollout_id)

        @self.app.post("/add_resources", response_model=ResourcesUpdate)
        async def add_resources(resources: AddResourcesRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.add_resources(resources.resources)

        @self.app.post("/update_resources", response_model=ResourcesUpdate)
        async def update_resources(update: ResourcesUpdate):  # pyright: ignore[reportUnusedFunction]
            return await self.store.update_resources(update.resources_id, update.resources)

        @self.app.get("/get_resources_by_id/{resources_id}", response_model=Optional[ResourcesUpdate])
        async def get_resources_by_id(resources_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_resources_by_id(resources_id)

        @self.app.get("/get_latest_resources", response_model=Optional[ResourcesUpdate])
        async def get_latest_resources():  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_latest_resources()

        @self.app.post("/add_span", response_model=Span)
        async def add_span(span: Span):  # pyright: ignore[reportUnusedFunction]
            return await self.store.add_span(span)

        @self.app.get("/get_next_span_sequence_id/{rollout_id}/{attempt_id}", response_model=int)
        async def get_next_span_sequence_id(rollout_id: str, attempt_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

        @self.app.post("/wait_for_rollouts", response_model=List[Rollout])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

        @self.app.get("/query_spans/{rollout_id}", response_model=List[Span])
        async def query_spans(  # pyright: ignore[reportUnusedFunction]
            rollout_id: str, attempt_id: Optional[str] = None
        ):
            return await self.store.query_spans(rollout_id, attempt_id)

        @self.app.post("/update_rollout", response_model=Rollout)
        async def update_rollout(request: UpdateRolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.update_rollout(
                rollout_id=request.rollout_id,
                input=request.input if not isinstance(request.input, PydanticUnset) else UNSET,
                mode=request.mode if not isinstance(request.mode, PydanticUnset) else UNSET,
                resources_id=request.resources_id if not isinstance(request.resources_id, PydanticUnset) else UNSET,
                status=request.status if not isinstance(request.status, PydanticUnset) else UNSET,
                config=request.config if not isinstance(request.config, PydanticUnset) else UNSET,
                metadata=request.metadata if not isinstance(request.metadata, PydanticUnset) else UNSET,
            )

        @self.app.post("/update_attempt", response_model=Attempt)
        async def update_attempt(request: UpdateAttemptRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.update_attempt(
                rollout_id=request.rollout_id,
                attempt_id=request.attempt_id,
                status=request.status if not isinstance(request.status, PydanticUnset) else UNSET,
                worker_id=request.worker_id if not isinstance(request.worker_id, PydanticUnset) else UNSET,
                last_heartbeat_time=(
                    request.last_heartbeat_time if not isinstance(request.last_heartbeat_time, PydanticUnset) else UNSET
                ),
                metadata=request.metadata if not isinstance(request.metadata, PydanticUnset) else UNSET,
            )

    # Delegate methods
    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        return await self._backend().start_rollout(input, mode, resources_id, config, metadata)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        return await self._backend().enqueue_rollout(input, mode, resources_id, config, metadata)

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await self._backend().dequeue_rollout()

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        return await self._backend().start_attempt(rollout_id)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        return await self._backend().query_rollouts(status=status, rollout_ids=rollout_ids)

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        return await self._backend().query_attempts(rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        return await self._backend().get_latest_attempt(rollout_id)

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        return await self._backend().get_rollout_by_id(rollout_id)

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        return await self._backend().add_resources(resources)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        return await self._backend().update_resources(resources_id, resources)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await self._backend().get_resources_by_id(resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        return await self._backend().get_latest_resources()

    async def add_span(self, span: Span) -> Span:
        return await self._backend().add_span(span)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return await self._backend().get_next_span_sequence_id(rollout_id, attempt_id)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        return await self._backend().add_otel_span(rollout_id, attempt_id, readable_span, sequence_id)

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        return await self._backend().wait_for_rollouts(rollout_ids=rollout_ids, timeout=timeout)

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        return await self._backend().query_spans(rollout_id, attempt_id)

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        return await self._backend().update_rollout(
            rollout_id=rollout_id,
            input=input,
            mode=mode,
            resources_id=resources_id,
            status=status,
            config=config,
            metadata=metadata,
        )

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        return await self._backend().update_attempt(
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            status=status,
            worker_id=worker_id,
            last_heartbeat_time=last_heartbeat_time,
            metadata=metadata,
        )


class LightningStoreClient(LightningStore):
    """HTTP client that talks to a remote LightningStoreServer.

    Args:
        server_address: The address of the LightningStoreServer to connect to.
        retry_delays:
            Backoff schedule (seconds) used when the initial request fails for a
            non-application reason. Each entry is a retry attempt.
        health_retry_delays:
            Delays between /health probes while waiting for the server to come back.
    """

    def __init__(
        self,
        server_address: str,
        *,
        retry_delays: Sequence[float] = (1.0, 2.0, 5.0),
        health_retry_delays: Sequence[float] = (0.1, 0.2, 0.5),
    ):
        self.server_address = server_address.rstrip("/")
        self._sessions: Dict[int, aiohttp.ClientSession] = {}  # id(loop) -> ClientSession
        self._lock = threading.RLock()

        # retry config
        self._retry_delays = tuple(float(d) for d in retry_delays)
        self._health_retry_delays = tuple(float(d) for d in health_retry_delays)

        # Store whether the dequeue was successful in history
        self._dequeue_was_successful: bool = False
        self._dequeue_first_unsuccessful: bool = True

    async def _get_session(self) -> aiohttp.ClientSession:
        # In the proxy process, FastAPI middleware calls
        # client_store.get_next_span_sequence_id(...). With
        # reuse_session=True, _get_session() creates and caches a
        # single ClientSession bound to the uvicorn event loop.
        #
        # Later, the OpenTelemetry exporter (LightningSpanExporter)
        # runs its flush on its own private event loop (in a different
        # thread) and calls client_store.add_otel_span(...) ->
        # client_store.add_span(...).
        #
        # If we reuse one session across all, the exporter tries to reuse the
        # same cached ClientSession that was created on the uvicorn
        # loop. aiohttp.ClientSession is not loop-agnostic or
        # thread-safe. Using it from another loop can hang on the
        # first request. That's why we need a map from loop to session.

        loop = asyncio.get_running_loop()
        key = id(loop)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None or sess.closed:
                sess = aiohttp.ClientSession()
                self._sessions[key] = sess
        return sess

    async def _wait_until_healthy(self, session: aiohttp.ClientSession) -> bool:
        """
        Probe the server's /health until it responds 200 or retries are exhausted.
        Returns True if healthy, False otherwise.
        """
        logger.info(f"Waiting for server to be healthy at {self.server_address}/health")
        for delay in [*self._health_retry_delays, 0.0]:
            try:
                async with session.get(f"{self.server_address}/health") as r:
                    if r.status == 200:
                        logger.info(f"Server is healthy at {self.server_address}/health")
                        return True
            except Exception:
                # swallow and retry
                if delay > 0.0:
                    logger.warning(f"Server is not healthy yet. Retrying in {delay} seconds.")
            if delay > 0.0:
                await asyncio.sleep(delay)
        logger.error(
            f"Server is not healthy at {self.server_address}/health after {len(self._health_retry_delays)} retry attempts"
        )
        return False

    async def _request_json(
        self,
        method: Literal["get", "post"],
        path: str,
        *,
        json: Any | None = None,
    ) -> Any:
        """
        Make an HTTP request with:

        1) First attempt.
        2) On network/session failures: probe /health until back, then retry
           according to self._retry_delays.
        3) On 4xx (e.g., 400 set by server exception handler): do not retry.

        Returns parsed JSON (or raw JSON scalar like int).
        Raises the last exception if all retries fail.
        """
        session = await self._get_session()
        url = f"{self.server_address}{path if path.startswith('/') else '/'+path}"

        # attempt 0 is immediate, then follow retry schedule
        attempts = (0.0,) + self._retry_delays
        last_exc: Exception | None = None

        for delay in attempts:
            if delay:
                logger.info(f"Waiting {delay} seconds before retrying {method}: {path}")
                await asyncio.sleep(delay)
            try:
                http_call = getattr(session, method)
                async with http_call(url, json=json) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError as cre:
                # Respect app-level 4xx as final (server marks app faults as 400)
                # 4xx => application issue; do not retry (except 408 which is transient)
                logger.exception(f"ClientResponseError: {cre.status} {cre.message}")
                if 400 <= cre.status < 500 and cre.status != 408:
                    raise
                # 5xx and others will be retried below if they raise
                last_exc = cre
                logger.info(f"5xx and other status codes will be retried. Retrying the request {method}: {path}")
                # before next retry, ensure server is healthy
                if not await self._wait_until_healthy(session):
                    break  # server is not healthy, do not retry
            except (
                aiohttp.ServerDisconnectedError,
                aiohttp.ClientConnectorError,
                aiohttp.ClientOSError,
                asyncio.TimeoutError,
            ) as net_exc:
                # Network/session issue: probe health before retrying
                logger.exception(f"Network/session issue: {net_exc}")
                last_exc = net_exc
                logger.info(f"Network/session issue will be retried. Retrying the request {method}: {path}")
                if not await self._wait_until_healthy(session):
                    break  # server is not healthy, do not retry

        # exhausted retries
        assert last_exc is not None
        raise last_exc

    async def close(self):
        """Close the HTTP session."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        # close them on their own loops to avoid warnings
        async def _close(sess: aiohttp.ClientSession):
            if not sess.closed:
                await sess.close()

        # If called from one loop, best-effort close here.
        for s in sessions:
            try:
                await _close(s)
            except RuntimeError:
                # If created on a different loop/thread, schedule a thread-safe close
                # Fallback: close without awaiting (library tolerates it in practice),
                # or keep a per-loop shutdown hook where they were created.
                pass

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        data = await self._request_json(
            "post",
            "/start_rollout",
            json=RolloutRequest(
                input=input,
                mode=mode,
                resources_id=resources_id,
                config=config,
                metadata=metadata,
            ).model_dump(exclude_none=False),
        )
        return AttemptedRollout.model_validate(data)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        data = await self._request_json(
            "post",
            "/enqueue_rollout",
            json=RolloutRequest(
                input=input,
                mode=mode,
                resources_id=resources_id,
                config=config,
                metadata=metadata,
            ).model_dump(exclude_none=False),
        )
        return Rollout.model_validate(data)

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """
        Dequeue a rollout from the server queue.

        Returns:
            AttemptedRollout if a rollout is available, None if queue is empty.

        Note:
            This method does NOT retry on failures. If any exception occurs (network error,
            server error, etc.), it logs the error and returns None immediately.
        """
        session = await self._get_session()
        url = f"{self.server_address}/dequeue_rollout"
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._dequeue_was_successful = True
                return AttemptedRollout.model_validate(data) if data else None
        except Exception as e:
            if self._dequeue_was_successful:
                if self._dequeue_first_unsuccessful:
                    logger.error(f"dequeue_rollout failed with exception: {e}", exc_info=True)
                    self._dequeue_first_unsuccessful = False
            # Else ignore the exception because the server is not ready yet
            return None

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        data = await self._request_json(
            "post",
            "/start_attempt",
            json=RolloutId(rollout_id=rollout_id).model_dump(),
        )
        return AttemptedRollout.model_validate(data)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        data = await self._request_json(
            "post",
            "/query_rollouts",
            json=QueryRolloutsRequest(
                status=list(status) if status else None,
                rollout_ids=list(rollout_ids) if rollout_ids else None,
            ).model_dump(),
        )
        return [Rollout.model_validate(item) for item in data]

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        data = await self._request_json("get", f"/query_attempts/{rollout_id}")
        return [Attempt.model_validate(item) for item in data]

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """
        Get the latest attempt for a rollout.

        Args:
            rollout_id: ID of the rollout to query.

        Returns:
            Attempt if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", f"/get_latest_attempt/{rollout_id}")
            return Attempt.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_latest_attempt failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True)
            return None

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        """
        Get a rollout by its ID.

        Args:
            rollout_id: ID of the rollout to retrieve.

        Returns:
            Rollout if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", f"/get_rollout_by_id/{rollout_id}")
            return Rollout.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_rollout_by_id failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True)
            return None

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        request = AddResourcesRequest(resources=resources)
        data = await self._request_json("post", "/add_resources", json=request.model_dump())
        return ResourcesUpdate.model_validate(data)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        data = await self._request_json(
            "post",
            "/update_resources",
            json=ResourcesUpdate(resources_id=resources_id, resources=resources).model_dump(),
        )
        return ResourcesUpdate.model_validate(data)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Get resources by their ID.

        Args:
            resources_id: ID of the resources to retrieve.

        Returns:
            ResourcesUpdate if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", f"/get_resources_by_id/{resources_id}")
            return ResourcesUpdate.model_validate(data) if data else None
        except Exception as e:
            logger.error(
                f"get_resources_by_id failed after all retries for resources_id={resources_id}: {e}", exc_info=True
            )
            return None

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """
        Get the latest resources.

        Returns:
            ResourcesUpdate if found, None if not found or if all retries are exhausted.

        Note:
            This method retries on transient failures (network errors, 5xx status codes).
            If all retries fail, it logs the error and returns None instead of raising an exception.
        """
        try:
            data = await self._request_json("get", "/get_latest_resources")
            return ResourcesUpdate.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_latest_resources failed after all retries: {e}", exc_info=True)
            return None

    async def add_span(self, span: Span) -> Span:
        data = await self._request_json("post", "/add_span", json=span.model_dump(mode="json"))
        return Span.model_validate(data)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        data = await self._request_json("get", f"/get_next_span_sequence_id/{rollout_id}/{attempt_id}")
        # endpoint returns a plain JSON number
        return int(data)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        # unchanged logic, now benefits from retries inside add_span/get_next_span_sequence_id
        if sequence_id is None:
            sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)
        span = Span.from_opentelemetry(
            readable_span,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
        )
        await self.add_span(span)
        return span

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        if timeout is not None and timeout > 0.1:
            raise ValueError(
                "Timeout must be less than 0.1 seconds in LightningStoreClient to avoid blocking the event loop"
            )
        data = await self._request_json(
            "post",
            "/wait_for_rollouts",
            json=WaitForRolloutsRequest(rollout_ids=rollout_ids, timeout=timeout).model_dump(),
        )
        return [Rollout.model_validate(item) for item in data]

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        path = f"/query_spans/{rollout_id}"
        if attempt_id is not None:
            path += f"?attempt_id={attempt_id}"
        data = await self._request_json("get", path)
        return [Span.model_validate(item) for item in data]

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        payload: Dict[str, Any] = {"rollout_id": rollout_id}
        if not isinstance(input, Unset):
            payload["input"] = input
        if not isinstance(mode, Unset):
            payload["mode"] = mode
        if not isinstance(resources_id, Unset):
            payload["resources_id"] = resources_id
        if not isinstance(status, Unset):
            payload["status"] = status
        if not isinstance(config, Unset):
            payload["config"] = config.model_dump()
        if not isinstance(metadata, Unset):
            payload["metadata"] = metadata

        data = await self._request_json("post", "/update_rollout", json=payload)
        return Rollout.model_validate(data)

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        payload: Dict[str, Any] = {
            "rollout_id": rollout_id,
            "attempt_id": attempt_id,
        }
        if not isinstance(status, Unset):
            payload["status"] = status
        if not isinstance(worker_id, Unset):
            payload["worker_id"] = worker_id
        if not isinstance(last_heartbeat_time, Unset):
            payload["last_heartbeat_time"] = last_heartbeat_time
        if not isinstance(metadata, Unset):
            payload["metadata"] = metadata

        data = await self._request_json("post", "/update_attempt", json=payload)
        return Attempt.model_validate(data)
