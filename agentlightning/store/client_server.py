# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Literal, Optional, Sequence

import aiohttp
import uvicorn
from fastapi import FastAPI
from opentelemetry.sdk.trace import ReadableSpan
from pydantic import BaseModel

from agentlightning.tracer import Span
from agentlightning.types import NamedResources, ResourcesUpdate, RolloutStatus, RolloutV2

from .base import LightningStore, LightningStoreWatchDog

logger = logging.getLogger(__name__)


# Request/Response models for API
class AddTaskRequest(BaseModel):
    sample: Any
    mode: Optional[Literal["train", "val", "test"]] = None
    resources_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class QueryRolloutsRequest(BaseModel):
    status: Optional[List[RolloutStatus]] = None


class WaitForRolloutsRequest(BaseModel):
    rollout_ids: List[str]
    timeout: Optional[float] = None


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
        self.app = FastAPI(title="LightningStore Server")
        self._setup_routes()
        self._uvicorn_config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._uvicorn_server = uvicorn.Server(self._uvicorn_config)

    @property
    def endpoint(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def start(self):
        """Starts the FastAPI server in the background."""
        logger.info(f"Starting server at {self.endpoint}")
        asyncio.create_task(self._uvicorn_server.serve())
        await asyncio.sleep(1)  # Allow time for server to start up.

    async def stop(self):
        """Gracefully stops the running FastAPI server."""
        if self._uvicorn_server.started:
            logger.info("Stopping server...")
            self._uvicorn_server.should_exit = True
            await asyncio.sleep(1)  # Allow time for graceful shutdown.
            logger.info("Server stopped.")

    def _setup_routes(self):
        """Set up FastAPI routes for all store operations."""

        @self.app.post("/add_task", response_model=RolloutV2)
        async def add_task(request: AddTaskRequest):  # type: ignore[unused]
            return await self.store.add_task(
                sample=request.sample,
                mode=request.mode,
                resources_id=request.resources_id,
                metadata=request.metadata,
            )

        @self.app.get("/pop_rollout", response_model=Optional[RolloutV2])
        async def pop_rollout():  # type: ignore[unused]
            return await self.store.pop_rollout()

        @self.app.post("/query_rollouts", response_model=List[RolloutV2])
        async def query_rollouts(request: QueryRolloutsRequest):  # type: ignore[unused]
            return await self.store.query_rollouts(status=request.status)

        @self.app.post("/update_resources", response_model=ResourcesUpdate)
        async def update_resources(update: ResourcesUpdate):  # type: ignore[unused]
            return await self.store.update_resources(update.resources_id, update.resources)

        @self.app.post("/add_rollout", response_model=RolloutV2)
        async def add_rollout(rollout: RolloutV2):  # type: ignore[unused]
            return await self.store.add_rollout(rollout)

        @self.app.get("/get_resources_by_id/{resources_id}", response_model=Optional[ResourcesUpdate])
        async def get_resources_by_id(resources_id: str):  # type: ignore[unused]
            return await self.store.get_resources_by_id(resources_id)

        @self.app.get("/get_latest_resources", response_model=Optional[ResourcesUpdate])
        async def get_latest_resources():  # type: ignore[unused]
            return await self.store.get_latest_resources()

        @self.app.post("/add_span", response_model=Span)
        async def add_span(span: Span):  # type: ignore[unused]
            return await self.store.add_span(span)

        @self.app.post("/wait_for_rollouts", response_model=List[RolloutV2])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):  # type: ignore[unused]
            return await self.store.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

        @self.app.get("/query_spans/{rollout_id}", response_model=List[Span])
        async def query_spans(rollout_id: str, attempt_id: Optional[str] = None):  # type: ignore[unused]
            return await self.store.query_spans(rollout_id, attempt_id)

        @self.app.get("/get_next_span_sequence_id/{rollout_id}/{attempt_id}", response_model=int)
        async def get_next_span_sequence_id(rollout_id: str, attempt_id: str):  # type: ignore[unused]
            return await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

        @self.app.post("/update_rollout", response_model=RolloutV2)
        async def update_rollout(  # type: ignore[unused]
            rollout_id: str,
            status: Optional[RolloutStatus] = None,
            worker_id: Optional[str] = None,
            attempt_sequence_id: Optional[int] = None,
            attempt_id: Optional[str] = None,
            attempt_start_time: Optional[float] = None,
            last_attempt_status: Optional[RolloutStatus] = None,
            **kwargs: Any,
        ):
            return await self.store.update_rollout(
                rollout_id=rollout_id,
                status=status,
                worker_id=worker_id,
                attempt_sequence_id=attempt_sequence_id,
                attempt_id=attempt_id,
                attempt_start_time=attempt_start_time,
                last_attempt_status=last_attempt_status,
                **kwargs,
            )

    # Delegate all LightningStore methods to the underlying store
    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        return await self.store.add_task(sample, mode, resources_id, metadata)

    async def add_rollout(self, rollout: RolloutV2) -> RolloutV2:
        return await self.store.add_rollout(rollout)

    async def pop_rollout(self) -> Optional[RolloutV2]:
        return await self.store.pop_rollout()

    async def query_rollouts(self, status: Optional[Sequence[RolloutStatus]] = None) -> List[RolloutV2]:
        return await self.store.query_rollouts(status)

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        return await self.store.update_resources(resources_id, resources)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await self.store.get_resources_by_id(resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        return await self.store.get_latest_resources()

    async def add_span(self, span: Span) -> Span:
        return await self.store.add_span(span)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        return await self.store.get_next_span_sequence_id(rollout_id, attempt_id)

    async def add_otel_span(
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        return await self.store.add_otel_span(rollout_id, attempt_id, readable_span, sequence_id)

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        return await self.store.wait_for_rollouts(rollout_ids, timeout)

    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        return await self.store.query_spans(rollout_id, attempt_id)

    async def update_rollout(
        self,
        rollout_id: str,
        status: Optional[RolloutStatus] = None,
        worker_id: Optional[str] = None,
        attempt_sequence_id: Optional[int] = None,
        attempt_id: Optional[str] = None,
        attempt_start_time: Optional[float] = None,
        last_attempt_status: Optional[RolloutStatus] = None,
        **kwargs: Any,
    ) -> RolloutV2:
        return await self.store.update_rollout(
            rollout_id=rollout_id,
            status=status,
            worker_id=worker_id,
            attempt_sequence_id=attempt_sequence_id,
            attempt_id=attempt_id,
            attempt_start_time=attempt_start_time,
            last_attempt_status=last_attempt_status,
            **kwargs,
        )


class LightningStoreClient(LightningStore):
    """
    Client implementation that communicates with a remote LightningStoreServer via HTTP.
    """

    def __init__(self, server_address: str, watchdog: LightningStoreWatchDog | None = None):
        super().__init__(watchdog)
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

    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        session = await self._get_session()
        request_data = AddTaskRequest(sample=sample, mode=mode, resources_id=resources_id, metadata=metadata)

        async with session.post(f"{self.server_address}/add_task", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data)

    async def pop_rollout(self) -> Optional[RolloutV2]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/pop_rollout") as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data) if data else None

    async def query_rollouts(self, status: Optional[Sequence[RolloutStatus]] = None) -> List[RolloutV2]:
        session = await self._get_session()
        request_data = QueryRolloutsRequest(status=list(status) if status else None)

        async with session.post(f"{self.server_address}/query_rollouts", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return [RolloutV2.model_validate(item) for item in data]

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

    async def add_rollout(self, rollout: RolloutV2) -> RolloutV2:
        session = await self._get_session()

        async with session.post(f"{self.server_address}/add_rollout", json=rollout.model_dump(mode="json")) as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data)

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
        self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan, sequence_id: int | None = None
    ) -> Span:
        if sequence_id is None:
            sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)
        span = Span.from_opentelemetry(
            readable_span, rollout_id=rollout_id, attempt_id=attempt_id, sequence_id=sequence_id
        )
        await self.add_span(span)
        return span

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        session = await self._get_session()
        request_data = WaitForRolloutsRequest(rollout_ids=rollout_ids, timeout=timeout)

        async with session.post(f"{self.server_address}/wait_for_rollouts", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return [RolloutV2.model_validate(item) for item in data]

    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        session = await self._get_session()

        url = f"{self.server_address}/query_spans/{rollout_id}"
        if attempt_id:
            url += f"?attempt_id={attempt_id}"

        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.json()
            return [Span.model_validate(item) for item in data]

    async def update_rollout(
        self,
        rollout_id: str,
        status: Optional[RolloutStatus] = None,
        worker_id: Optional[str] = None,
        attempt_sequence_id: Optional[int] = None,
        attempt_id: Optional[str] = None,
        attempt_start_time: Optional[float] = None,
        last_attempt_status: Optional[RolloutStatus] = None,
        **kwargs: Any,
    ) -> RolloutV2:
        session = await self._get_session()

        request_data = {
            "rollout_id": rollout_id,
            "status": status,
            "worker_id": worker_id,
            "attempt_sequence_id": attempt_sequence_id,
            "attempt_id": attempt_id,
            "attempt_start_time": attempt_start_time,
            "last_attempt_status": last_attempt_status,
            **kwargs,
        }
        # Remove None values
        request_data = {k: v for k, v in request_data.items() if v is not None}

        async with session.post(f"{self.server_address}/update_rollout", json=request_data) as response:
            response.raise_for_status()
            data = await response.json()
            return RolloutV2.model_validate(data)
