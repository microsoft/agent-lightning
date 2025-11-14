# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Generic, List, Literal, Optional, Sequence, TypeVar, Union

import aiohttp
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException
from fastapi import Query as FastAPIQuery
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest as PbExportTraceServiceRequest,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceResponse as PbExportTraceServiceResponse,
)
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
    Worker,
    WorkerStatus,
)
from agentlightning.utils.otlp import handle_otlp_export, spans_from_text_proto
from agentlightning.utils.server_launcher import LaunchMode, PythonServerLauncher, PythonServerLauncherArgs

from .base import UNSET, LightningStore, LightningStoreCapabilities, Unset

server_logger = logging.getLogger("agentlightning.store.server")
client_logger = logging.getLogger("agentlightning.store.client")

API_V1_PREFIX = "/v1"
API_AGL_PREFIX = "/agl"
API_V1_AGL_PREFIX = API_V1_PREFIX + API_AGL_PREFIX

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


class DequeueRolloutRequest(BaseModel):
    worker_id: Optional[str] = None


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


class UpdateWorkerRequest(BaseModel):
    heartbeat_stats: Optional[Dict[str, Any]] = None


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


class QueryWorkersRequest(BaseModel):
    status_in: Optional[List[WorkerStatus]] = Field(FastAPIQuery(default=None))
    worker_id_contains: Optional[str] = None
    # Pagination
    limit: int = -1
    offset: int = 0
    # Sorting
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "asc"
    # Filtering logic
    filter_logic: Literal["and", "or"] = "and"


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


class CachedStaticFiles(StaticFiles):
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        resp = super().file_response(*args, **kwargs)
        # hashed filenames are safe to cache "forever"
        resp.headers.setdefault("Cache-Control", "public, max-age=31536000, immutable")
        return resp


class LightningStoreServer(LightningStore):
    """
    Server wrapper that exposes a LightningStore via HTTP API.
    Delegates all operations to an underlying store implementation.

    Healthcheck and watchdog relies on the underlying store.

    `agl store` is a convenient CLI to start a store server.

    When the server is executed in a subprocess, the store will discover itself having a different PID
    and automatically delegate to an HTTP client instead of using the local store.
    This ensures one single copy of the store will be shared across all processes.

    Args:
        store: The underlying store to delegate operations to.
        host: The hostname or IP address to bind the server to.
        port: The TCP port to listen on.
        cors_allow_origins: A list of CORS origins to allow. Use '*' to allow all origins.
        launch_mode: The launch mode to use for the server. Defaults to "thread",
            which runs the server in a separate thread.
        launcher_args: The arguments to use for the server launcher.
            It's not allowed to set `host`, `port`, `launch_mode` together with `launcher_args`.
    """

    def __init__(
        self,
        store: LightningStore,
        host: str | None = None,
        port: int | None = None,
        cors_allow_origins: Sequence[str] | str | None = None,
        launch_mode: LaunchMode = "thread",
        launcher_args: PythonServerLauncherArgs | None = None,
    ):
        super().__init__()
        self.store = store

        if launcher_args is not None:
            if host is not None or port is not None or launch_mode != "thread":
                raise ValueError("host, port, and launch_mode cannot be set when launcher_args is provided.")
            self.launcher_args = launcher_args
        else:
            if port is None:
                server_logger.warning("No port provided, using default port 4747.")
                port = 4747
            self.launcher_args = PythonServerLauncherArgs(
                host=host,
                port=port,
                launch_mode=launch_mode,
                healthcheck_url=API_V1_AGL_PREFIX + "/health",
            )

        store_capabilities = self.store.capabilities
        if not store_capabilities.get("async_safe", False):
            raise ValueError("The store is not async-safe. Please use another store for the server.")
        if self.launcher_args.launch_mode == "mp" and not store_capabilities.get("zero_copy", False):
            raise ValueError(
                "The store does not support zero-copy. Please use another store, or use asyncio or thread mode to launch the server."
            )
        if self.launcher_args.launch_mode == "thread" and not store_capabilities.get("thread_safe", False):
            server_logger.warning(
                "The store is not thread-safe. Please be careful when using the store server and the underlying store in different threads."
            )

        self.app: FastAPI | None = FastAPI(title="LightningStore Server")
        self.server_launcher = PythonServerLauncher(
            app=self.app,
            args=self.launcher_args,
        )

        self._lock: threading.Lock = threading.Lock()
        self._cors_allow_origins = self._normalize_cors_origins(cors_allow_origins)
        self._apply_cors()
        self._setup_routes()

        # Process-awareness:
        # LightningStoreServer holds a plain Python object (self.store) in one process
        # (the process that runs uvicorn/FastAPI).
        # When you multiprocessing.Process(...) and call methods on a different LightningStore instance
        # (or on a copy inherited via fork), you’re mutating another process’s memory, not the server’s memory.
        # So we need to track the owner process (whoever creates the server),
        # and only mutate the store in that process.
        self._owner_pid = os.getpid()
        self._client: Optional[LightningStoreClient] = None

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            async_safe=True,
            thread_safe=True,
            zero_copy=True,
            otlp_traces=True,
        )

    def otlp_traces_endpoint(self) -> str:
        """Return the OTLP/HTTP traces endpoint of the store."""
        return f"{self.endpoint}/v1/traces"

    def __getstate__(self):
        """
        Control pickling to prevent server state from being sent to subprocesses.

        When LightningStoreServer is pickled (e.g., passed to a subprocess), we only
        serialize the underlying store and connection details. The client instance
        and process-awareness state are excluded as they should not be transferred between processes.

        The subprocess should create its own server instance if needed.
        """
        # server-launcher is needed for the host/port address are propagated to the subprocess
        return {
            "launcher_args": self.launcher_args,
            "server_launcher": self.server_launcher,
            "_owner_pid": self._owner_pid,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore from pickle by reconstructing only the essential attributes.

        Note: This creates a new server instance without FastAPI/uvicorn initialized.
        Call __init__() pattern or create a new LightningStoreServer if you need
        a fully functional server in the subprocess.
        The unpickled server will also have no app and store attributes,
        this is to make sure there is only one copy of the server in the whole system.
        """
        self.app = None
        self.store = None
        self.launcher_args = state["launcher_args"]
        self.server_launcher = state["server_launcher"]
        self._owner_pid = state["_owner_pid"]
        self._cors_allow_origins = state.get("_cors_allow_origins")
        self._client = None
        self._lock = threading.Lock()
        # Do NOT reconstruct app, _uvicorn_config, _uvicorn_server
        # to avoid transferring server state to subprocess

    @staticmethod
    def _normalize_cors_origins(
        origins: Sequence[str] | str | None,
    ) -> list[str] | None:
        if origins is None:
            return None

        if isinstance(origins, str):
            candidates = [origins]
        else:
            candidates = list(origins)

        cleaned: list[str] = []
        for origin in candidates:
            if not origin or not origin.strip():
                continue
            value = origin.strip()
            if value == "*":
                return ["*"]
            cleaned.append(value)

        return cleaned or None

    def _apply_cors(self) -> None:
        if self.app is None or not self._cors_allow_origins:
            return

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self._cors_allow_origins.copy(),
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=["*"],
        )

    @property
    def endpoint(self) -> str:
        """Endpoint is the address that the client will use to connect to the server."""
        return self.server_launcher.access_endpoint

    async def start(self):
        """Starts the FastAPI server in the background.

        You need to call this method in the same process as the server was created in.
        """
        server_logger.info(
            f"Serving the lightning store at {self.server_launcher.endpoint}, accessible at {self.server_launcher.access_endpoint}"
        )

        start_time = time.time()
        await self.server_launcher.start()
        end_time = time.time()
        server_logger.info(f"Lightning store server started in {end_time - start_time:.2f} seconds")

    async def run_forever(self):
        """Runs the FastAPI server indefinitely."""
        server_logger.info(
            f"Running the lightning store server at {self.server_launcher.endpoint}, accessible at {self.server_launcher.access_endpoint}"
        )
        await self.server_launcher.run_forever()

    async def stop(self):
        """Gracefully stops the running FastAPI server.

        You need to call this method in the same process as the server was created in.
        """
        server_logger.info("Stopping the lightning store server...")
        await self.server_launcher.stop()
        server_logger.info("Lightning store server stopped.")

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
                if request.url.path.startswith(API_V1_AGL_PREFIX):
                    server_logger.exception("Unhandled application error", exc_info=exc)
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
            # If not API request, just pass through
            if not request.url.path.startswith(API_V1_AGL_PREFIX):
                return await call_next(request)

            start = time.perf_counter()
            response = await call_next(request)
            duration = (time.perf_counter() - start) * 1000
            client = request.client
            if client is None:
                client_address = "unknown"
            else:
                client_address = f"{client.host}:{client.port}"
            server_logger.debug(
                f"{client_address} - "
                f'"{request.method} {request.url.path} HTTP/{request.scope["http_version"]}" '
                f"{response.status_code} in {duration:.2f} ms"
            )
            return response

        api = APIRouter(prefix=API_V1_PREFIX)

        @api.get(API_AGL_PREFIX + "/health")
        async def health():  # pyright: ignore[reportUnusedFunction]
            return {"status": "ok"}

        @api.post(API_AGL_PREFIX + "/queues/rollouts/enqueue", status_code=201, response_model=Rollout)
        async def enqueue_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.enqueue_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @api.post(API_AGL_PREFIX + "/queues/rollouts/dequeue", response_model=Optional[AttemptedRollout])
        async def dequeue_rollout(  # pyright: ignore[reportUnusedFunction]
            request: DequeueRolloutRequest | None = Body(None),
        ):
            worker_id = request.worker_id if request else None
            return await self.dequeue_rollout(worker_id=worker_id)

        @api.post(API_AGL_PREFIX + "/rollouts", status_code=201, response_model=AttemptedRollout)
        async def start_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.start_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                config=request.config,
                metadata=request.metadata,
            )

        @api.get(API_AGL_PREFIX + "/rollouts", response_model=PaginatedResponse[Union[AttemptedRollout, Rollout]])
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

        @api.get(API_AGL_PREFIX + "/rollouts/{rollout_id}", response_model=Union[AttemptedRollout, Rollout])
        async def get_rollout_by_id(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_rollout_by_id(rollout_id)

        def _get_mandatory_field_or_unset(request: BaseModel | None, field: str) -> Any:
            # If some fields are mandatory by the underlying store, but optional in the FastAPI,
            # we make sure it's set to non-null value or UNSET via this function.
            if request is None:
                return UNSET
            if field in request.model_fields_set:
                value = getattr(request, field)
                if value is None:
                    raise HTTPException(status_code=400, detail=f"{field} is invalid; it cannot be a null value.")
                return value
            else:
                return UNSET

        @api.post(API_AGL_PREFIX + "/rollouts/{rollout_id}", response_model=Rollout)
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

        @api.post(API_AGL_PREFIX + "/rollouts/{rollout_id}/attempts", status_code=201, response_model=AttemptedRollout)
        async def start_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.start_attempt(rollout_id)

        @api.post(API_AGL_PREFIX + "/rollouts/{rollout_id}/attempts/{attempt_id}", response_model=Attempt)
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

        @api.get(API_AGL_PREFIX + "/workers", response_model=PaginatedResponse[Worker])
        async def query_workers(params: QueryWorkersRequest = Depends()):  # pyright: ignore[reportUnusedFunction]
            all_workers = await self.query_workers()

            filters: Dict[str, Any] = {}
            if params.status_in:
                filters["status_in"] = params.status_in
            if params.worker_id_contains is not None:
                filters["worker_id_contains"] = params.worker_id_contains

            return _apply_filters_sort_paginate(
                all_workers,
                filters,
                params.filter_logic,
                params.sort_by,
                params.sort_order,
                params.limit,
                params.offset,
            )

        @api.get(API_AGL_PREFIX + "/workers/{worker_id}", response_model=Optional[Worker])
        async def get_worker(worker_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_worker_by_id(worker_id)

        @api.post(API_AGL_PREFIX + "/workers/{worker_id}", response_model=Worker)
        async def update_worker(  # pyright: ignore[reportUnusedFunction]
            worker_id: str, request: UpdateWorkerRequest | None = Body(None)
        ):
            return await self.update_worker(
                worker_id=worker_id,
                heartbeat_stats=_get_mandatory_field_or_unset(request, "heartbeat_stats"),
            )

        @api.get(API_AGL_PREFIX + "/rollouts/{rollout_id}/attempts", response_model=PaginatedResponse[Attempt])
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

        @api.get(API_AGL_PREFIX + "/rollouts/{rollout_id}/attempts/latest", response_model=Optional[Attempt])
        async def get_latest_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_latest_attempt(rollout_id)

        @api.get(API_AGL_PREFIX + "/resources", response_model=PaginatedResponse[ResourcesUpdate])
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

        @api.post(API_AGL_PREFIX + "/resources", status_code=201, response_model=ResourcesUpdate)
        async def add_resources(resources: NamedResources):  # pyright: ignore[reportUnusedFunction]
            return await self.add_resources(resources)

        @api.get(API_AGL_PREFIX + "/resources/latest", response_model=Optional[ResourcesUpdate])
        async def get_latest_resources():  # pyright: ignore[reportUnusedFunction]
            return await self.get_latest_resources()

        @api.post(API_AGL_PREFIX + "/resources/{resources_id}", response_model=ResourcesUpdate)
        async def update_resources(  # pyright: ignore[reportUnusedFunction]
            resources_id: str, resources: NamedResources
        ):
            return await self.update_resources(resources_id, resources)

        @api.get(API_AGL_PREFIX + "/resources/{resources_id}", response_model=Optional[ResourcesUpdate])
        async def get_resources_by_id(resources_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.get_resources_by_id(resources_id)

        @api.post(API_AGL_PREFIX + "/spans", status_code=201, response_model=Span)
        async def add_span(span: Span):  # pyright: ignore[reportUnusedFunction]
            return await self.add_span(span)

        @api.get(API_AGL_PREFIX + "/spans", response_model=PaginatedResponse[Span])
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

        @api.post(API_AGL_PREFIX + "/spans/next", response_model=NextSequenceIdResponse)
        async def get_next_span_sequence_id(request: NextSequenceIdRequest):  # pyright: ignore[reportUnusedFunction]
            sequence_id = await self.get_next_span_sequence_id(request.rollout_id, request.attempt_id)
            return NextSequenceIdResponse(sequence_id=sequence_id)

        @api.post(API_AGL_PREFIX + "/waits/rollouts", response_model=List[Rollout])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

        # Setup OTLP endpoints
        self._setup_otlp(api)

        # Mount the API router of /v1/...
        self.app.include_router(api)

        # Finally, mount the dashboard assets
        self._setup_dashboard()

    def _setup_otlp(self, api: APIRouter):
        """Setup OTLP endpoints."""

        async def _trace_handler(request: PbExportTraceServiceRequest) -> None:
            spans = spans_from_text_proto(request)
            for span in spans:
                print(span)
                # await self.add_span(span)

        # Reserved methods for OTEL traces
        # https://opentelemetry.io/docs/specs/otlp/#otlphttp-request
        @api.post("/traces")
        async def otlp_traces(request: Request):  # pyright: ignore[reportUnusedFunction]
            return await handle_otlp_export(
                request, PbExportTraceServiceRequest, PbExportTraceServiceResponse, _trace_handler, "traces"
            )

        @api.post("/metrics")
        async def otlp_metrics():  # pyright: ignore[reportUnusedFunction]
            return Response(status_code=501)

        @api.post("/logs")
        async def otlp_logs():  # pyright: ignore[reportUnusedFunction]
            return Response(status_code=501)

        @api.post("/development/profiles")
        async def otlp_development_profiles():  # pyright: ignore[reportUnusedFunction]
            return Response(status_code=501)

    def _setup_dashboard(self):
        """Setup the dashboard static files and SPA."""
        assert self.app is not None

        dashboard_dir = (Path(__file__).parent.parent / "dashboard").resolve()
        if not dashboard_dir.exists():
            server_logger.error("Dashboard directory not found at %s. Please build the dashboard first.", dashboard_dir)
            return

        dashboard_assets_dir = dashboard_dir / "assets"
        if not dashboard_assets_dir.exists():
            server_logger.error(
                "Dashboard assets directory not found at %s. Please build the dashboard first.", dashboard_assets_dir
            )
            return

        index_file = dashboard_dir / "index.html"
        if not index_file.exists():
            server_logger.error("Dashboard index file not found at %s. Please build the dashboard first.", index_file)
            return

        # Mount the static files in dashboard/assets
        self.app.mount("/assets", CachedStaticFiles(directory=dashboard_assets_dir), name="assets")

        # SPA fallback (client-side routing)
        # Anything that's not /v1/* or a real file in /assets will serve index.html
        @self.app.get("/", include_in_schema=False)
        def root():  # pyright: ignore[reportUnusedFunction]
            return FileResponse(index_file)

        @self.app.get("/{full_path:path}", include_in_schema=False)
        def spa_fallback(full_path: str):  # pyright: ignore[reportUnusedFunction]
            # Let the frontend router handle it
            return FileResponse(index_file)

        server_logger.info("Agent-lightning dashboard will be available at %s", self.endpoint)

    # Delegate methods
    async def _call_store_method(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """First decide what store to delegate to in *this* process, and then call the method on it.

        - In the owner process: delegate to the in-process store.
        - In a different process: delegate to a HTTP client talking to the server.
        """
        if os.getpid() == self._owner_pid:
            if method_name == "wait_for_rollouts":
                # wait_for_rollouts can block for a long time; avoid holding the lock
                # so other requests can make progress while we wait.
                return await getattr(self.store, method_name)(*args, **kwargs)
            with self._lock:
                return await getattr(self.store, method_name)(*args, **kwargs)
        if self._client is None:
            self._client = LightningStoreClient(self.endpoint)
        return await getattr(self._client, method_name)(*args, **kwargs)

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

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        return await self._call_store_method("dequeue_rollout", worker_id)

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

    async def query_workers(self) -> List[Worker]:
        return await self._call_store_method("query_workers")

    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        return await self._call_store_method("get_worker_by_id", worker_id)

    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        return await self._call_store_method(
            "update_worker",
            worker_id,
            heartbeat_stats,
        )


class LightningStoreClient(LightningStore):
    """HTTP client that talks to a remote LightningStoreServer.

    Args:
        server_address: The address of the LightningStoreServer to connect to.
        retry_delays:
            Backoff schedule (seconds) used when the initial request fails for a
            non-application reason. Each entry is a retry attempt.
            Setting to an empty sequence to disable retries.
        health_retry_delays:
            Delays between /health probes while waiting for the server to come back.
            Setting to an empty sequence to disable health checks.
        request_timeout: Timeout (seconds) for each request.
        connection_timeout: Timeout (seconds) for establishing connection.
    """

    def __init__(
        self,
        server_address: str,
        *,
        retry_delays: Sequence[float] = (1.0, 2.0, 5.0),
        health_retry_delays: Sequence[float] = (0.1, 0.2, 0.5),
        request_timeout: float = 30.0,
        connection_timeout: float = 5.0,
    ):
        self.server_address_root = server_address.rstrip("/")
        self.server_address = self.server_address_root + API_V1_AGL_PREFIX
        self._sessions: Dict[int, aiohttp.ClientSession] = {}  # id(loop) -> ClientSession
        self._lock = threading.Lock()

        # retry config
        self._retry_delays = tuple(float(d) for d in retry_delays)
        self._health_retry_delays = tuple(float(d) for d in health_retry_delays)

        # Timeouts
        self._request_timeout = request_timeout
        self._connection_timeout = connection_timeout

        # Store whether the dequeue was successful in history
        self._dequeue_was_successful: bool = False
        self._dequeue_first_unsuccessful: bool = True

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            thread_safe=True,
            async_safe=True,
            zero_copy=True,
            otlp_traces=True,
        )

    def otlp_traces_endpoint(self) -> str:
        """Return the OTLP/HTTP traces endpoint of the store."""
        return f"{self.server_address_root}/v1/traces"

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
            "_request_timeout": self._request_timeout,
            "_connection_timeout": self._connection_timeout,
        }

    def __setstate__(self, state: Dict[str, Any]):
        """
        Restore from pickle by reconstructing only the essential attributes.

        Replicating `__init__` logic to create another client instance in the subprocess.
        """
        self.server_address = state["server_address"]
        self._sessions = {}
        self._lock = threading.Lock()
        self._retry_delays = state["_retry_delays"]
        self._health_retry_delays = state["_health_retry_delays"]
        self._request_timeout = state["_request_timeout"]
        self._connection_timeout = state["_connection_timeout"]
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
                timeout = aiohttp.ClientTimeout(
                    total=self._request_timeout,
                    connect=self._connection_timeout,
                    sock_connect=self._connection_timeout,
                    sock_read=self._request_timeout,
                )
                sess = aiohttp.ClientSession(timeout=timeout)
                self._sessions[key] = sess
        return sess

    async def _wait_until_healthy(self, session: aiohttp.ClientSession) -> bool:
        """
        Probe the server's /health until it responds 200 or retries are exhausted.
        Returns True if healthy, False otherwise.
        """
        if not self._health_retry_delays:
            client_logger.info("No health retry delays configured; skipping health checks.")
            return True

        client_logger.info(f"Waiting for server to be healthy at {self.server_address}/health")
        for delay in [*self._health_retry_delays, 0.0]:
            try:
                async with session.get(f"{self.server_address}/health") as r:
                    if r.status == 200:
                        client_logger.info(f"Server is healthy at {self.server_address}/health")
                        return True
            except Exception:
                # swallow and retry
                if delay > 0.0:
                    client_logger.warning(f"Server is not healthy yet. Retrying in {delay} seconds.")
            if delay > 0.0:
                await asyncio.sleep(delay)
        client_logger.error(
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
                client_logger.info(f"Waiting {delay} seconds before retrying {method}: {path}")
                await asyncio.sleep(delay)
            try:
                http_call = getattr(session, method)
                async with http_call(url, json=json, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except aiohttp.ClientResponseError as cre:
                # Respect app-level 4xx as final
                # 4xx => application issue; do not retry (except 408 which is transient)
                client_logger.debug(f"ClientResponseError: {cre.status} {cre.message}", exc_info=True)
                if 400 <= cre.status < 500 and cre.status != 408:
                    raise
                # 5xx and others will be retried below if they raise
                last_exc = cre
                client_logger.info(f"5xx and other status codes will be retried. Retrying the request {method}: {path}")
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
                client_logger.debug(f"Network/session issue: {net_exc}", exc_info=True)
                last_exc = net_exc
                client_logger.info(f"Network/session issue will be retried. Retrying the request {method}: {path}")
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

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
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
        request_kwargs: Dict[str, Any] = {}
        if worker_id is not None:
            request_kwargs["json"] = {"worker_id": worker_id}
        try:
            async with session.post(url, **request_kwargs) as resp:
                resp.raise_for_status()
                data = await resp.json()
                self._dequeue_was_successful = True
                return AttemptedRollout.model_validate(data) if data else None
        except Exception as e:
            if self._dequeue_was_successful:
                if self._dequeue_first_unsuccessful:
                    client_logger.warning(f"dequeue_rollout failed with exception: {e}")
                    self._dequeue_first_unsuccessful = False
            client_logger.debug("dequeue_rollout failed with exception. Details:", exc_info=True)
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
        return [
            (
                AttemptedRollout.model_validate(item)
                if isinstance(item, dict) and "attempt" in item
                else Rollout.model_validate(item)
            )
            for item in data["items"]
        ]

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
            client_logger.error(
                f"get_latest_attempt failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True
            )
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
            if isinstance(data, dict) and "attempt" in data:
                return AttemptedRollout.model_validate(data)
            else:
                return Rollout.model_validate(data)
        except Exception as e:
            client_logger.error(
                f"get_rollout_by_id failed after all retries for rollout_id={rollout_id}: {e}", exc_info=True
            )
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
            client_logger.error(
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
            client_logger.error(f"get_latest_resources failed after all retries: {e}", exc_info=True)
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

    async def query_workers(self) -> List[Worker]:
        data = await self._request_json("get", "/workers")
        items = data.get("items", [])
        return [Worker.model_validate(item) for item in items]

    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        data = await self._request_json("get", f"/workers/{worker_id}")
        if data is None:
            return None
        return Worker.model_validate(data)

    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        payload: Dict[str, Any] = {}
        if not isinstance(heartbeat_stats, Unset):
            payload["heartbeat_stats"] = heartbeat_stats
        json_payload = payload if payload else None

        data = await self._request_json("post", f"/workers/{worker_id}", json=json_payload)
        return Worker.model_validate(data)
