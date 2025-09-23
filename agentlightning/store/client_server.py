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
from agentlightning.types import ResourcesUpdate, RolloutStatus, RolloutV2

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


class AddSpanRequest(BaseModel):
    rollout_id: str
    attempt_id: str
    span_data: str  # JSON serialized ReadableSpan


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
        async def add_task(request: AddTaskRequest):
            return await self.store.add_task(
                sample=request.sample,
                mode=request.mode,
                resources_id=request.resources_id,
                metadata=request.metadata,
            )

        @self.app.get("/pop_rollout", response_model=Optional[RolloutV2])
        async def pop_rollout():
            return await self.store.pop_rollout()

        @self.app.post("/query_rollouts", response_model=List[RolloutV2])
        async def query_rollouts(request: QueryRolloutsRequest):
            return await self.store.query_rollouts(status=request.status)

        @self.app.post("/update_resources")
        async def update_resources(update: ResourcesUpdate):
            await self.store.update_resources(update)
            return {"status": "success"}

        @self.app.get("/get_resources_by_id/{resources_id}", response_model=Optional[ResourcesUpdate])
        async def get_resources_by_id(resources_id: str):
            return await self.store.get_resources_by_id(resources_id)

        @self.app.get("/get_latest_resources", response_model=Optional[ResourcesUpdate])
        async def get_latest_resources():
            return await self.store.get_latest_resources()

        @self.app.post("/add_span", response_model=Span)
        async def add_span(request: AddSpanRequest):
            # This would need to reconstruct ReadableSpan from JSON
            # For now, we'll use a placeholder approach
            raise NotImplementedError("Span serialization not yet implemented")

        @self.app.post("/wait_for_rollouts", response_model=List[RolloutV2])
        async def wait_for_rollouts(request: WaitForRolloutsRequest):
            return await self.store.wait_for_rollouts(rollout_ids=request.rollout_ids, timeout=request.timeout)

        @self.app.get("/query_spans/{rollout_id}", response_model=List[Span])
        async def query_spans(rollout_id: str):
            return await self.store.query_spans(rollout_id)

    # Delegate all LightningStore methods to the underlying store
    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        return await self.store.add_task(sample, mode, resources_id, metadata)

    async def pop_rollout(self) -> Optional[RolloutV2]:
        return await self.store.pop_rollout()

    async def query_rollouts(self, status: Optional[Sequence[RolloutStatus]] = None) -> List[RolloutV2]:
        return await self.store.query_rollouts(status)

    async def update_resources(self, update: ResourcesUpdate) -> None:
        return await self.store.update_resources(update)

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        return await self.store.get_resources_by_id(resources_id)

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        return await self.store.get_latest_resources()

    async def add_span(self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan) -> Span:
        return await self.store.add_span(rollout_id, attempt_id, readable_span)

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        return await self.store.wait_for_rollouts(rollout_ids, timeout)

    async def query_spans(self, rollout_id: str) -> List[Span]:
        return await self.store.query_spans(rollout_id)


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

    async def update_resources(self, update: ResourcesUpdate):
        session = await self._get_session()

        async with session.post(f"{self.server_address}/update_resources", json=update.model_dump()) as response:
            response.raise_for_status()

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

    async def add_span(self, rollout_id: str, attempt_id: str, readable_span: ReadableSpan) -> Span:
        # This would need to serialize ReadableSpan to JSON
        # For now, we'll raise NotImplementedError
        raise NotImplementedError("Span serialization over HTTP not yet implemented")

    async def wait_for_rollouts(self, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        session = await self._get_session()
        request_data = WaitForRolloutsRequest(rollout_ids=rollout_ids, timeout=timeout)

        async with session.post(f"{self.server_address}/wait_for_rollouts", json=request_data.model_dump()) as response:
            response.raise_for_status()
            data = await response.json()
            return [RolloutV2.model_validate(item) for item in data]

    async def query_spans(self, rollout_id: str) -> List[Span]:
        session = await self._get_session()

        async with session.get(f"{self.server_address}/query_spans/{rollout_id}") as response:
            response.raise_for_status()
            data = await response.json()
            return [Span.model_validate(item) for item in data]
