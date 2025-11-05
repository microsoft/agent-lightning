# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from contextlib import suppress
from typing import Any, Awaitable, Callable, Dict, Generic, List, Literal, Optional, Sequence, TypeVar

import aiohttp
import uvicorn
from fastapi import Body, Depends, FastAPI, HTTPException
from fastapi import Query as FastAPIQuery
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field, TypeAdapter

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

AGL_API_V1_PREFIX = "/v1/agl"

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    limit: int
    offset: int
    total: int


class RolloutRequest(BaseModel):
    input: TaskInput
    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None
    config: Optional[RolloutConfig] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryRolloutsRequest(BaseModel):
    status_in: Optional[List[RolloutStatus]] = Field(FastAPIQuery(default=None))
    rollout_id_in: Optional[List[str]] = Field(FastAPIQuery(default=None))
    rollout_id_contains: Optional[str] = None
    # Pagination
    limit: int = -1
    offset: int = 0
    # Sorting
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"
    # Filtering logic
    filter_logic: Literal["and", "or"] = "and"


class WaitForRolloutsRequest(BaseModel):
    rollout_ids: List[str]
    timeout: Optional[float] = None


class NextSequenceIdRequest(BaseModel):
    rollout_id: str
    attempt_id: str


class NextSequenceIdResponse(BaseModel):
    sequence_id: int


class UpdateRolloutRequest(BaseModel):
    input: Optional[TaskInput] = None
    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None
    status: Optional[RolloutStatus] = None
    config: Optional[RolloutConfig] = None
    metadata: Optional[Dict[str, Any]] = None


class UpdateAttemptRequest(BaseModel):
    status: Optional[AttemptStatus] = None
    worker_id: Optional[str] = None
    last_heartbeat_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryAttemptsRequest(BaseModel):
    # Pagination
    limit: int = -1
    offset: int = 0
    # Sorting
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"


class QueryResourcesRequest(BaseModel):
    # Filtering
    resources_id: Optional[str] = None
    resources_id_contains: Optional[str] = None
    # Pagination
    limit: int = -1
    offset: int = 0
    # Sorting
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"


class QuerySpansRequest(BaseModel):
    rollout_id: str
    attempt_id: Optional[str] = None
    # Filtering
    trace_id: Optional[str] = None
    trace_id_contains: Optional[str] = None
    span_id: Optional[str] = None
    span_id_contains: Optional[str] = None
    parent_id: Optional[str] = None
    parent_id_contains: Optional[str] = None
    name: Optional[str] = None
    name_contains: Optional[str] = None
    filter_logic: Literal["and", "or"] = "and"
    # Pagination
    limit: int = -1
    offset: int = 0
    # Sorting
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"


def _apply_filters_sort_paginate(
    items: List[T],
    filters: Dict[str, Any],
    filter_logic: Literal["and", "or"],
    sort_by: Optional[str],
    sort_order: Literal["asc", "desc"],
    limit: int,
    offset: int,
) -> PaginatedResponse[T]:
    """Apply filtering, sorting, and pagination to a list of items."""
    # Apply filters
    filtered_items: List[T] = []
    if not filters:
        filtered_items = items
    else:
        for item in items:
            matches: List[bool] = []
            for key, value in filters.items():
                if value is None:
                    continue

                # Handle _in suffix (list membership)
                if key.endswith("_in"):
                    field = key[:-3]
                    item_value = getattr(item, field, None)
                    matches.append(item_value in value if isinstance(value, list) else False)
                # Handle _contains suffix (substring match)
                elif key.endswith("_contains"):
                    field = key[:-9]
                    item_value = getattr(item, field, None)
                    if item_value is not None and isinstance(item_value, str) and isinstance(value, str):
                        matches.append(value in item_value)
                    else:
                        matches.append(False)
                # Exact match
                else:
                    item_value = getattr(item, key, None)
                    matches.append(item_value == value)

            if matches:
                if filter_logic == "and":
                    if all(matches):
                        filtered_items.append(item)
                else:  # "or"
                    if any(matches):
                        filtered_items.append(item)

    # Apply sorting

    def _get_sort_value(item: T, sort_by: str) -> Any:
        if sort_by.endswith("_time"):
            value = getattr(item, sort_by, None)
            if value is None:
                value = float("inf")
            return value
        else:
            # Other than _time, we assume the value must be a string
            value = getattr(item, sort_by, None)
            if value is None:
                if sort_by not in item.__class__.model_fields:  # type: ignore
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to sort items by {sort_by}: {sort_by} is not a field of {item.__class__.__name__}",
                    )
                field_type = str(item.__class__.model_fields[sort_by].annotation)  # type: ignore
                if "str" in field_type or "Literal" in field_type:
                    return ""
                if "int" in field_type:
                    return 0
                if "float" in field_type:
                    return 0.0
                raise HTTPException(
                    status_code=400, detail=f"Failed to sort items by {sort_by}: {value} is not a string or number"
                )
            return value

    if sort_by:
        reverse = sort_order == "desc"
        filtered_items.sort(key=lambda x: _get_sort_value(x, sort_by), reverse=reverse)

    # Get total count before pagination
    total = len(filtered_items)

    # Apply pagination
    if limit == -1:
        paginated_items = filtered_items[offset:]
    else:
        paginated_items = filtered_items[offset : offset + limit]

    return PaginatedResponse(items=paginated_items, limit=limit, offset=offset, total=total)


class LightningStoreServer(LightningStore):
    """
    Server wrapper that exposes a LightningStore via HTTP API.
    Delegates all operations to an underlying store implementation.

    Healthcheck and watchdog relies on the underlying store.

    `agl store` is a convenient CLI to start a store server.
    """

    def __init__(self, store: LightningStore, host: str, port: int):
        super().__init__()
        self.store = store
        self._lock = threading.Lock()
        self.host = host
        self.port = port
        self.app: FastAPI | None = FastAPI(title="LightningStore Server")
        self._setup_routes()
        self._uvicorn_config: uvicorn.Config | None = uvicorn.Config(
            self.app, host="0.0.0.0", port=self.port, log_level="error"
        )
        self._uvicorn_server: uvicorn.Server | None = uvicorn.Server(self._uvicorn_config)

        self._serving_thread: Optional[threading.Thread] = None
        self._server_start_exception: Optional[BaseException] = None

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
        self._lock = threading.Lock()
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
        self._server_start_exception = None

        def run_server_forever():
            try:
                asyncio.run(uvicorn_server.serve())
            except (SystemExit, Exception) as exc:
                logger.debug("LightningStore server thread exiting due to %s", exc, exc_info=exc)
                self._server_start_exception = exc

        serving_thread = threading.Thread(target=run_server_forever, daemon=True)
        self._serving_thread = serving_thread
        serving_thread.start()

        # Wait for uvicorn to report that it has started before pinging /health.
        start_deadline = time.time() + 10
        while time.time() < start_deadline:
            if uvicorn_server.started:
                break
            if self._server_start_exception is not None or not serving_thread.is_alive():
                self._handle_failed_start()
                raise RuntimeError(self._format_start_failure_reason())
            await asyncio.sleep(0.05)
        else:
            self._handle_failed_start()
            raise RuntimeError("Server failed to start within the 10 seconds.")

        # Wait for /health to be available once uvicorn reports started.
        if not await self._server_health_check():
            self._handle_failed_start()
            raise RuntimeError("Server failed to start within the 10 seconds.")

        # If startup failed (e.g. port already in use), uvicorn never flips `started`
        # and the worker thread stops immediately. Guard against latching on to a
        # different process that happened to satisfy the health check.
        if not uvicorn_server.started or not serving_thread.is_alive() or self._server_start_exception is not None:
            self._handle_failed_start()
            failure_reason = self._format_start_failure_reason()
            raise RuntimeError(failure_reason)

    async def _server_health_check(self) -> bool:
        """Checks if the server is healthy."""
        current_time = time.time()
        while time.time() - current_time < 10:
            async with aiohttp.ClientSession() as session:
                with suppress(Exception):
                    async with session.get(f"{self.endpoint}{AGL_API_V1_PREFIX}/health") as response:
                        if response.status == 200:
                            return True
            await asyncio.sleep(0.1)
        return False

    def _handle_failed_start(self) -> None:
        """Clean up thread state when startup fails."""
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._serving_thread is not None:
            # Thread already exited in most failure scenarios; join defensively.
            self._serving_thread.join(timeout=0.1)
            self._serving_thread = None

    def _format_start_failure_reason(self) -> str:
        base_message = f"LightningStore server failed to start on {self.endpoint}."
        if isinstance(self._server_start_exception, SystemExit):
            return f"{base_message} Another process may already be using this port."
        if isinstance(self._server_start_exception, OSError):
            return f"{base_message} {self._server_start_exception.strerror}."
        if self._server_start_exception is not None:
            return f"{base_message} Reason: {self._server_start_exception}."
        return f"{base_message} Another process may already be using this port."

    async def run_forever(self):
        """Runs the FastAPI server indefinitely.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None
        uvicorn_server = self._uvicorn_server

        async def _wait_till_healthy():
            health = await self._server_health_check()
            if not health:
                raise RuntimeError("Server did not become healthy within the 10 seconds.")
            logger.info("Store server is online at %s", self.endpoint)

        async def _serve_capture():
            try:
                await uvicorn_server.serve()
            except KeyboardInterrupt:
                raise
            except (SystemExit, Exception) as exc:
                logger.debug("LightningStore server serve() raised %s", exc, exc_info=exc)
                self._server_start_exception = exc
                raise RuntimeError("LightningStore server failed to serve") from exc

        # We run _wait_till_healthy and self._uvicorn_server.serve in parallel
        # until one of them raises an exception.
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

        @self.app.middleware("http")
        async def _app_exception_handler(  # pyright: ignore[reportUnusedFunction]
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ) -> Response:
            """
            Convert unhandled application exceptions into 500 responses.

            Only covers /v1/agl requests.

            - Client needs a reliable signal to distinguish "app bug / bad request"
              from transport/session failures.
            - 400 means "do not retry"; network issues will surface as aiohttp
              exceptions or 5xx and will be retried by the client shield.
            """
            try:
                return await call_next(request)
            except Exception as exc:
                # decide whether to convert this into your 400 JSONResponse
                if request.url.path.startswith(AGL_API_V1_PREFIX):
                    logger.exception("Unhandled application error", exc_info=exc)
                    payload = {
                        "detail": "Internal server error",
                        "error_type": type(exc).__name__,
                        "traceback": traceback.format_exc(),
                    }
                    # 500 so clients can decide to retry
                    return JSONResponse(status_code=500, content=payload)
                # otherwise re-raise and let FastAPI/Starlette handle it (500 or other handlers)
                raise

        @self.app.middleware("http")
        async def _log_time(  # pyright: ignore[reportUnusedFunction]
            request: Request, call_next: Callable[[Request], Awaitable[Response]]
        ):
            if not request.url.path.startswith("/v1/agl/"):
                return await call_next(request)

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

        @self.app.get(AGL_API_V1_PREFIX + "/health")
        async def health():  # pyright: ignore[reportUnusedFunction]
            return {"status": "ok"}

        @self.app.post(AGL_API_V1_PREFIX + "/queues/rollouts/enqueue", status_code=201, response_model=Rollout)
        async def enqueue_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.enqueue_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @self.app.post(AGL_API_V1_PREFIX + "/queues/rollouts/dequeue", response_model=Optional[AttemptedRollout])
        async def dequeue_rollout():  # pyright: ignore[reportUnusedFunction]
            return await self.dequeue_rollout()

        @self.app.post(AGL_API_V1_PREFIX + "/rollouts", status_code=201, response_model=AttemptedRollout)
        async def start_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.start_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @self.app.get(AGL_API_V1_PREFIX + "/rollouts", response_model=PaginatedResponse[Rollout])
        async def query_rollouts(params: QueryRolloutsRequest = Depends()):  # pyright: ignore[reportUnusedFunction]
            # Get all rollouts from the underlying store
            all_rollouts = await self.query_rollouts()

            # Build filter dict
            filters: Dict[str, Any] = {}
            if params.status_in:
                filters["status_in"] = params.status_in
            if params.rollout_id_in:
                filters["rollout_id_in"] = params.rollout_id_in
            if params.rollout_id_contains is not None:
                filters["rollout_id_contains"] = params.rollout_id_contains

            return _apply_filters_sort_paginate(
                all_rollouts,
                filters,
                params.filter_logic,
                params.sort_by,
                params.sort_order,
                params.limit,
                params.offset,
            )

        @self.app.get(AGL_API_V1_PREFIX + "/rollouts/{rollout_id}", response_model=Rollout)
        async def get_rollout_by_id(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_rollout_by_id(rollout_id)

        def _get_mandatory_field_or_unset(request: BaseModel, field: str) -> Any:
            # If some fields are mandatory by the underlying store, but optional in the FastAPI,
            # we make sure it's set to non-null value or UNSET via this function.
            if field in request.model_fields_set:
                value = getattr(request, field)
                if value is None:
                    raise HTTPException(status_code=400, detail=f"{field} is invalid; it cannot be a null value.")
                return value
            else:
                return UNSET

        @self.app.post(AGL_API_V1_PREFIX + "/rollouts/{rollout_id}", response_model=Rollout)
        async def update_rollout(  # pyright: ignore[reportUnusedFunction]
            rollout_id: str, request: UpdateRolloutRequest = Body(...)
        ):
            return await self.update_rollout(
                rollout_id=rollout_id,
                input=request.input if "input" in request.model_fields_set else UNSET,
                mode=request.mode if "mode" in request.model_fields_set else UNSET,
                resources_id=request.resources_id if "resources_id" in request.model_fields_set else UNSET,
                status=_get_mandatory_field_or_unset(request, "status"),
                config=_get_mandatory_field_or_unset(request, "config"),
                metadata=request.metadata if "metadata" in request.model_fields_set else UNSET,
            )

        @self.app.post(
            AGL_API_V1_PREFIX + "/rollouts/{rollout_id}/attempts", status_code=201, response_model=AttemptedRollout
        )
        async def start_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.start_attempt(rollout_id)

        @self.app.post(AGL_API_V1_PREFIX + "/rollouts/{rollout_id}/attempts/{attempt_id}", response_model=Attempt)
        async def update_attempt(  # pyright: ignore[reportUnusedFunction]
            rollout_id: str, attempt_id: str, request: UpdateAttemptRequest = Body(...)
        ):
            return await self.update_attempt(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                status=_get_mandatory_field_or_unset(request, "status"),
                worker_id=_get_mandatory_field_or_unset(request, "worker_id"),
                last_heartbeat_time=_get_mandatory_field_or_unset(request, "last_heartbeat_time"),
                metadata=_get_mandatory_field_or_unset(request, "metadata"),
            )

        @self.app.get(AGL_API_V1_PREFIX + "/rollouts/{rollout_id}/attempts", response_model=PaginatedResponse[Attempt])
        async def query_attempts(  # pyright: ignore[reportUnusedFunction]
            rollout_id: str, params: QueryAttemptsRequest = Depends()
        ):
            # Get all attempts for the rollout
            all_attempts = await self.query_attempts(rollout_id)

            return _apply_filters_sort_paginate(
                all_attempts,
                {},  # No filters for attempts
                "and",
                params.sort_by,
                params.sort_order,
                params.limit,
                params.offset,
            )

        @self.app.get(AGL_API_V1_PREFIX + "/rollouts/{rollout_id}/attempts/latest", response_model=Optional[Attempt])
        async def get_latest_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_latest_attempt(rollout_id)

        @self.app.get(AGL_API_V1_PREFIX + "/resources", response_model=PaginatedResponse[ResourcesUpdate])
        async def query_resources(params: QueryResourcesRequest = Depends()):  # pyright: ignore[reportUnusedFunction]
            # Get all resources
            all_resources = await self.query_resources()

            # Build filter dict
            filters: Dict[str, Any] = {}
            if params.resources_id is not None:
                filters["resources_id"] = params.resources_id
            if params.resources_id_contains is not None:
                filters["resources_id_contains"] = params.resources_id_contains

            return _apply_filters_sort_paginate(
                all_resources,
                filters,
                "and",
                params.sort_by,
                params.sort_order,
                params.limit,
                params.offset,
            )

        @self.app.post(AGL_API_V1_PREFIX + "/resources", status_code=201, response_model=ResourcesUpdate)
        async def add_resources(resources: NamedResources):  # pyright: ignore[reportUnusedFunction]
            return await self.add_resources(resources)

        @self.app.get(AGL_API_V1_PREFIX + "/resources/latest", response_model=Optional[ResourcesUpdate])
        async def get_latest_resources():  # pyright: ignore[reportUnusedFunction]
            return await self.get_latest_resources()

        @self.app.post(AGL_API_V1_PREFIX + "/resources/{resources_id}", response_model=ResourcesUpdate)
        async def update_resources(  # pyright: ignore[reportUnusedFunction]
            resources_id: str, resources: NamedResources
        ):
            return await self.update_resources(resources_id, resources)

        @self.app.get(AGL_API_V1_PREFIX + "/resources/{resources_id}", response_model=Optional[ResourcesUpdate])
        async def get_resources_by_id(resources_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_resources_by_id(resources_id)

        @self.app.post(AGL_API_V1_PREFIX + "/spans", status_code=201, response_model=Span)
        async def add_span(span: Span):  # pyright: ignore[reportUnusedFunction]
            return await self.add_span(span)

        @self.app.get(AGL_API_V1_PREFIX + "/spans", response_model=PaginatedResponse[Span])
        async def query_spans(params: QuerySpansRequest = Depends()):  # pyright: ignore[reportUnusedFunction]
            # Get all spans for the rollout/attempt
            all_spans = await self.query_spans(params.rollout_id, params.attempt_id)

            # Build filter dict
            filters: Dict[str, Any] = {}
            if params.trace_id is not None:
                filters["trace_id"] = params.trace_id
            if params.trace_id_contains is not None:
                filters["trace_id_contains"] = params.trace_id_contains
            if params.span_id is not None:
                filters["span_id"] = params.span_id
            if params.span_id_contains is not None:
                filters["span_id_contains"] = params.span_id_contains
            if params.parent_id is not None:
                filters["parent_id"] = params.parent_id
            if params.parent_id_contains is not None:
                filters["parent_id_contains"] = params.parent_id_contains
            if params.name is not None:
                filters["name"] = params.name
            if params.name_contains is not None:
                filters["name_contains"] = params.name_contains

            return _apply_filters_sort_paginate(
                all_spans, filters, params.filter_logic, params.sort_by, params.sort_order, params.limit, params.offset
            )

        @self.app.post(AGL_API_V1_PREFIX + "/spans/next", response_model=NextSequenceIdResponse)
        async def get_next_span_sequence_id(request: NextSequenceIdRequest):  # pyright: ignore[reportUnusedFunction]
            sequence_id = await self.get_next_span_sequence_id(request.rollout_id, request.attempt_id)
            return NextSequenceIdResponse(sequence_id=sequence_id)

        @self.app.post(AGL_API_V1_PREFIX + "/waits/rollouts", response_model=List[Rollout])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

    # Delegate methods
    async def _call_store_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        backend = self._backend()
        method = getattr(backend, method_name)
        if backend is self.store:
            if method_name == "wait_for_rollouts":
                # wait_for_rollouts can block for a long time; avoid holding the lock
                # so other requests can make progress while we wait.
                return await method(*args, **kwargs)
            with self._lock:
                return await method(*args, **kwargs)
        return await method(*args, **kwargs)

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        return await self._call_store_method(
            "start_rollout",
            input,
            mode,
            resources_id,
            config,
            metadata,
        )

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        return await self._call_store_method(
            "enqueue_rollout",
            input,
            mode,
            resources_id,
            config,
            metadata,
        )

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await self._call_store_method("dequeue_rollout")

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        return await self._call_store_method("start_attempt", rollout_id)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        return await self._call_store_method("query_rollouts", status=status, rollout_ids=rollout_ids)

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        return await self._call_store_method("query_attempts", rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        return await self._call_store_method("get_latest_attempt", rollout_id)

    async def query_resources(self) -> List[ResourcesUpdate]:
        return await self._call_store_method("query_resources")

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        return await self._call_store_method("get_rollout_by_id", rollout_id)

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        return await self._call_store_method("add_resources", resources)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        return await self._call_store_method("update_resources", resources_id, resources)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await self._call_store_method("get_resources_by_id", resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        return await self._call_store_method("get_latest_resources")

    async def add_span(self, span: Span) -> Span:
        return await self._call_store_method("add_span", span)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return await self._call_store_method("get_next_span_sequence_id", rollout_id, attempt_id)

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        return await self._call_store_method(
            "add_otel_span",
            rollout_id,
            attempt_id,
            readable_span,
            sequence_id,
        )

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        return await self._call_store_method("wait_for_rollouts", rollout_ids=rollout_ids, timeout=timeout)

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        return await self._call_store_method("query_spans", rollout_id, attempt_id)

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
        return await self._call_store_method(
            "update_rollout",
            rollout_id,
            input,
            mode,
            resources_id,
            status,
            config,
            metadata,
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
        return await self._call_store_method(
            "update_attempt",
            rollout_id,
            attempt_id,
            status,
            worker_id,
            last_heartbeat_time,
            metadata,
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
        self.server_address = server_address.rstrip("/") + AGL_API_V1_PREFIX
        self._sessions: Dict[int, aiohttp.ClientSession] = {}  # id(loop) -> ClientSession
        self._lock = threading.RLock()

        # retry config
        self._retry_delays = tuple(float(d) for d in retry_delays)
        self._health_retry_delays = tuple(float(d) for d in health_retry_delays)

        # Store whether the dequeue was successful in history
        self._dequeue_was_successful: bool = False
        self._dequeue_first_unsuccessful: bool = True

    def __getstate__(self):
        """
        When LightningStoreClient is pickled (e.g., passed to a subprocess), we only
        serialize the server address and retry configurations. The ClientSessions
        are excluded as they should not be transferred between processes.
        """
        return {
            "server_address": self.server_address,
            "_retry_delays": self._retry_delays,
            "_health_retry_delays": self._health_retry_delays,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore from pickle by reconstructing only the essential attributes.

        Replicating `__init__` logic to create another client instance in the subprocess.
        """
        self.server_address = state["server_address"]
        self._sessions = {}
        self._lock = threading.RLock()
        self._retry_delays = state["_retry_delays"]
        self._health_retry_delays = state["_health_retry_delays"]
        self._dequeue_was_successful = False
        self._dequeue_first_unsuccessful = True

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
                timeout = aiohttp.ClientTimeout(total=30.0, connect=5.0, sock_connect=5.0, sock_read=30.0)
                sess = aiohttp.ClientSession(timeout=timeout)
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
        params: Dict[str, Any] | None = None,
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
                async with http_call(url, json=json, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError as cre:
                # Respect app-level 4xx as final
                # 4xx => application issue; do not retry (except 408 which is transient)
                logger.debug(f"ClientResponseError: {cre.status} {cre.message}", exc_info=True)
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
                logger.debug(f"Network/session issue: {net_exc}", exc_info=True)
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
            "/rollouts",
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
            "/queues/rollouts/enqueue",
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
        url = f"{self.server_address}/queues/rollouts/dequeue"
        try:
            async with session.post(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._dequeue_was_successful = True
                return AttemptedRollout.model_validate(data) if data else None
        except Exception as e:
            if self._dequeue_was_successful:
                if self._dequeue_first_unsuccessful:
                    logger.warning(f"dequeue_rollout failed with exception: {e}")
                    self._dequeue_first_unsuccessful = False
            logger.debug("dequeue_rollout failed with exception. Details:", exc_info=True)
            # Else ignore the exception because the server is not ready yet
            return None

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        data = await self._request_json(
            "post",
            f"/rollouts/{rollout_id}/attempts",
        )
        return AttemptedRollout.model_validate(data)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Rollout]:
        params: Dict[str, Any] = {}
        if status is not None:
            params["status_in"] = status
        if rollout_ids is not None:
            params["rollout_id_in"] = list(rollout_ids)

        data = await self._request_json("get", "/rollouts", params=params if params else None)
        # Extract items from PaginatedResponse
        return [Rollout.model_validate(item) for item in data["items"]]

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        data = await self._request_json("get", f"/rollouts/{rollout_id}/attempts")
        # Extract items from PaginatedResponse
        return [Attempt.model_validate(item) for item in data["items"]]

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
            data = await self._request_json("get", f"/rollouts/{rollout_id}/attempts/latest")
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
            data = await self._request_json("get", f"/rollouts/{rollout_id}")
            return Rollout.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_rollout_by_id failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True)
            return None

    async def query_resources(self) -> List[ResourcesUpdate]:
        """
        List all resource snapshots stored on the server.
        """
        data = await self._request_json("get", "/resources")
        if not data:
            return []
        # Extract items from PaginatedResponse
        return [ResourcesUpdate.model_validate(item) for item in data["items"]]

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        data = await self._request_json("post", "/resources", json=TypeAdapter(NamedResources).dump_python(resources))
        return ResourcesUpdate.model_validate(data)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        data = await self._request_json(
            "post", f"/resources/{resources_id}", json=TypeAdapter(NamedResources).dump_python(resources)
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
            data = await self._request_json("get", f"/resources/{resources_id}")
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
            data = await self._request_json("get", "/resources/latest")
            return ResourcesUpdate.model_validate(data) if data else None
        except Exception as e:
            logger.error(f"get_latest_resources failed after all retries: {e}", exc_info=True)
            return None

    async def add_span(self, span: Span) -> Span:
        data = await self._request_json("post", "/spans", json=span.model_dump(mode="json"))
        return Span.model_validate(data)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        data = await self._request_json(
            "post",
            "/spans/next",
            json=NextSequenceIdRequest(rollout_id=rollout_id, attempt_id=attempt_id).model_dump(),
        )
        response = NextSequenceIdResponse.model_validate(data)
        return response.sequence_id

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
        """Wait for rollouts to complete.

        Args:
            rollout_ids: List of rollout IDs to wait for.
            timeout: Timeout in seconds. If not None, the method will raise a ValueError if the timeout is greater than 0.1 seconds.

        Returns:
            List of rollouts that are completed.
        """
        if timeout is not None and timeout > 0.1:
            raise ValueError(
                "Timeout must be less than 0.1 seconds in LightningStoreClient to avoid blocking the event loop"
            )
        data = await self._request_json(
            "post",
            "/waits/rollouts",
            json=WaitForRolloutsRequest(rollout_ids=rollout_ids, timeout=timeout).model_dump(),
        )
        return [Rollout.model_validate(item) for item in data]

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        params: Dict[str, str] = {"rollout_id": rollout_id}
        if attempt_id is not None:
            params["attempt_id"] = attempt_id
        data = await self._request_json("get", "/spans", params=params)
        # Extract items from PaginatedResponse
        return [Span.model_validate(item) for item in data["items"]]

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
        payload: Dict[str, Any] = {}
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

        data = await self._request_json("post", f"/rollouts/{rollout_id}", json=payload)
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
        payload: Dict[str, Any] = {}
        if not isinstance(status, Unset):
            payload["status"] = status
        if not isinstance(worker_id, Unset):
            payload["worker_id"] = worker_id
        if not isinstance(last_heartbeat_time, Unset):
            payload["last_heartbeat_time"] = last_heartbeat_time
        if not isinstance(metadata, Unset):
            payload["metadata"] = metadata

        data = await self._request_json(
            "post",
            f"/rollouts/{rollout_id}/attempts/{attempt_id}",
            json=payload,
        )
        return Attempt.model_validate(data)
