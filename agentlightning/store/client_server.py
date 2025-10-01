# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import aiohttp
import uvicorn
from fastapi import FastAPI
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel, Field

from agentlightning.tracer import Span
from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    RolloutConfig,
    RolloutStatus,
    RolloutV2,
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
    metadata: Optional[Dict[str, Any]] = None


class QueryRolloutsRequest(BaseModel):
    status: Optional[List[RolloutStatus]] = None
    rollout_ids: Optional[List[str]] = None


class WaitForRolloutsRequest(BaseModel):
    rollout_ids: List[str]
    timeout: Optional[float] = None


class RolloutId(BaseModel):
    rollout_id: str


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
            self.app, host=self.host, port=self.port, log_level="info"
        )
        self._uvicorn_server: uvicorn.Server | None = uvicorn.Server(self._uvicorn_config)

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
        asyncio.create_task(self._uvicorn_server.serve())
        await asyncio.sleep(1)  # Allow time for server to start up.

    async def stop(self):
        """Gracefully stops the running FastAPI server.

        You need to call this method in the same process as the server was created in.
        """
        assert self._uvicorn_server is not None
        if self._uvicorn_server.started:
            logger.info("Stopping server...")
            self._uvicorn_server.should_exit = True
            await asyncio.sleep(1)  # Allow time for graceful shutdown.
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

        @self.app.post("/start_rollout", response_model=AttemptedRollout)
        async def start_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.start_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                metadata=request.metadata,
            )

        @self.app.post("/enqueue_rollout", response_model=RolloutV2)
        async def enqueue_rollout(request: RolloutRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.enqueue_rollout(
                input=request.input,
                mode=request.mode,
                resources_id=request.resources_id,
                metadata=request.metadata,
            )

        @self.app.get("/dequeue_rollout", response_model=Optional[AttemptedRollout])
        async def dequeue_rollout():  # pyright: ignore[reportUnusedFunction]
            return await self.store.dequeue_rollout()

        @self.app.post("/start_attempt", response_model=AttemptedRollout)
        async def start_attempt(request: RolloutId):  # pyright: ignore[reportUnusedFunction]
            return await self.store.start_attempt(request.rollout_id)

        @self.app.post("/query_rollouts", response_model=List[RolloutV2])
        async def query_rollouts(request: QueryRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.query_rollouts(status=request.status)

        @self.app.get("/query_attempts/{rollout_id}", response_model=List[Attempt])
        async def query_attempts(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.query_attempts(rollout_id)

        @self.app.get("/get_latest_attempt/{rollout_id}", response_model=Optional[Attempt])
        async def get_latest_attempt(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_latest_attempt(rollout_id)

        @self.app.get("/get_rollout_by_id/{rollout_id}", response_model=Optional[RolloutV2])
        async def get_rollout_by_id(rollout_id: str):  # pyright: ignore[reportUnusedFunction]
            return await self.store.get_rollout_by_id(rollout_id)

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

        @self.app.post("/wait_for_rollouts", response_model=List[RolloutV2])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):  # pyright: ignore[reportUnusedFunction]
            return await self.store.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

        @self.app.get("/query_spans/{rollout_id}", response_model=List[Span])
        async def query_spans(  # pyright: ignore[reportUnusedFunction]
            rollout_id: str, attempt_id: Optional[str] = None
        ):
            return await self.store.query_spans(rollout_id, attempt_id)

        @self.app.post("/update_rollout", response_model=RolloutV2)
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
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        return await self._backend().start_rollout(input, mode, resources_id, metadata)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        return await self._backend().enqueue_rollout(input, mode, resources_id, metadata)

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        return await self._backend().dequeue_rollout()

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        return await self._backend().start_attempt(rollout_id)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[RolloutV2]:
        return await self._backend().query_rollouts(status=status, rollout_ids=rollout_ids)

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        return await self._backend().query_attempts(rollout_id)

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        return await self._backend().get_latest_attempt(rollout_id)

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[RolloutV2]:
        return await self._backend().get_rollout_by_id(rollout_id)

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

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
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
    ) -> RolloutV2:
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
    """HTTP client that talks to a remote LightningStoreServer."""

    def __init__(self, server_address: str):
        self.server_address = server_address.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close the HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        session = await self._get_session()
        request_data = RolloutRequest(input=input, mode=mode, resources_id=resources_id, metadata=metadata)

        async with session.post(f"{self.server_address}/start_rollout", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return AttemptedRollout.model_validate(data)

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        session = await self._get_session()
        request_data = RolloutRequest(input=input, mode=mode, resources_id=resources_id, metadata=metadata)

        async with session.post(f"{self.server_address}/enqueue_rollout", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data)

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/dequeue_rollout") as response:
            response.raise_for_status()
            data = await response.json()
            return AttemptedRollout.model_validate(data) if data else None

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        session = await self._get_session()
        request_data = RolloutId(rollout_id=rollout_id)

        async with session.post(f"{self.server_address}/start_attempt", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return AttemptedRollout.model_validate(data)

    async def query_rollouts(
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[RolloutV2]:
        session = await self._get_session()
        request_data = QueryRolloutsRequest(
            status=list(status) if status else None, rollout_ids=list(rollout_ids) if rollout_ids else None
        )

        async with session.post(f"{self.server_address}/query_rollouts", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return [RolloutV2.model_validate(item) for item in data]

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/query_attempts/{rollout_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return [Attempt.model_validate(item) for item in data]

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/get_latest_attempt/{rollout_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return Attempt.model_validate(data) if data else None

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[RolloutV2]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/get_rollout_by_id/{rollout_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data) if data else None

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        session = await self._get_session()
        update = ResourcesUpdate(resources_id=resources_id, resources=resources)

        async with session.post(f"{self.server_address}/update_resources", json=update.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return ResourcesUpdate.model_validate(data)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/get_resources_by_id/{resources_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return ResourcesUpdate.model_validate(data) if data else None

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/get_latest_resources") as response:
            response.raise_for_status()
            data = await response.json()
            return ResourcesUpdate.model_validate(data) if data else None

    async def add_span(self, span: Span) -> Span:
        session = await self._get_session()

        async with session.post(f"{self.server_address}/add_span", json=span.model_dump(mode="json")) as response:
            response.raise_for_status()
            data = await response.json()
            return Span.model_validate(data)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        session = await self._get_session()

        async with session.get(
            f"{self.server_address}/get_next_span_sequence_id/{rollout_id}/{attempt_id}"
        ) as response:
            response.raise_for_status()
            data = await response.json()
            return data

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
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

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        session = await self._get_session()
        if timeout is not None and timeout > 0.1:
            raise ValueError(
                "Timeout must be less than 0.1 seconds in LightningStoreClient to avoid blocking the event loop"
            )
        request_data = WaitForRolloutsRequest(rollout_ids=rollout_ids, timeout=timeout)

        async with session.post(f"{self.server_address}/wait_for_rollouts", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return [RolloutV2.model_validate(item) for item in data]

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
    ) -> List[Span]:
        session = await self._get_session()

        url = f"{self.server_address}/query_spans/{rollout_id}"
        if attempt_id is not None:
            url += f"?attempt_id={attempt_id}"

        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
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
    ) -> RolloutV2:
        session = await self._get_session()

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

        async with session.post(f"{self.server_address}/update_rollout", json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data)

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        session = await self._get_session()

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

        async with session.post(f"{self.server_address}/update_attempt", json=payload) as response:
            response.raise_for_status()
            data = await response.json()
            return Attempt.model_validate(data)
