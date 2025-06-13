import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Path
from pydantic import Field

from .types import (
    Rollout,
    Task,
    TaskIfAny,
    NamedResources,
    GenericResponse,
    ResourcesUpdate,
    TaskMetadata,
)

logger = logging.getLogger(__name__)


class ServerDataStore:
    """
    A centralized, thread-safe, async, in-memory data store for the server's state.
    This holds the task queue, versioned resources, and completed rollouts.
    """

    def __init__(self):
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self._completed_rollouts: Dict[str, Rollout] = {}

        # Store for versioned resources
        self._resource_versions: Dict[str, NamedResources] = {}
        self._latest_resources_id: Optional[str] = None

        # Locks for thread-safe access
        self._results_lock = asyncio.Lock()
        self._resources_lock = asyncio.Lock()

    async def add_task(self, sample: Any, metadata: TaskMetadata) -> str:
        """
        Adds a new task to the queue with specific metadata and returns its unique ID.
        """
        rollout_id = f"rollout-{uuid.uuid4()}"
        task = Task(rollout_id=rollout_id, input=sample, metadata=metadata)
        await self._task_queue.put(task)
        logger.info(f"Task queued: {rollout_id} (Metadata: {metadata})")
        return rollout_id

    async def get_next_task(self) -> Optional[Task]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.
        """
        # TODO: send the task back to queue if timeout.
        try:
            return self._task_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def update_resources(self, update: ResourcesUpdate):
        """
        Safely stores a new version of named resources and sets it as the latest.
        """
        # TODO: evict old resources if necessary.
        async with self._resources_lock:
            self._resource_versions[update.resources_id] = update.resources
            self._latest_resources_id = update.resources_id
            logger.info(f"Resources updated. New version '{update.resources_id}' is now latest.")

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves a specific version of named resources by its ID.
        """
        async with self._resources_lock:
            resources = self._resource_versions.get(resources_id)
            if resources:
                return ResourcesUpdate(resources_id=resources_id, resources=resources)
            return None

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """
        Safely retrieves the latest version of named resources.
        """
        async with self._resources_lock:
            if self._latest_resources_id:
                return await self.get_resources_by_id(self._latest_resources_id)
            return None

    async def store_rollout(self, rollout: Rollout):
        """
        Safely stores a completed rollout from a client.
        """
        async with self._results_lock:
            self._completed_rollouts[rollout.rollout_id] = rollout
            logger.info(f"Rollout received and stored: {rollout.rollout_id}")

    async def retrieve_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """
        Safely retrieves a single rollout by its ID, removing it from the store.
        """
        async with self._results_lock:
            return self._completed_rollouts.pop(rollout_id, None)

    async def retrieve_completed_rollouts(self) -> List[Rollout]:
        """
        Retrieves all completed rollouts and clears the store.
        """
        async with self._results_lock:
            rollouts = list(self._completed_rollouts.values())
            self._completed_rollouts.clear()
            return rollouts


server_store: Optional[ServerDataStore] = None


def get_server_store() -> ServerDataStore:
    """Returns the global server data store instance."""
    global server_store
    if not server_store:
        server_store = ServerDataStore()
    return server_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager to handle server startup and shutdown logic."""
    logger.info("Agent Lightning Server is starting up...")
    yield
    logger.info("Agent Lightning Server is shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/task", response_model=TaskIfAny)
async def next_task() -> TaskIfAny:
    """Endpoint for clients to poll for the next available task."""
    task = await get_server_store().get_next_task()
    if task:
        logger.debug(f"Serving task {task.rollout_id} to a client.")
        return TaskIfAny(is_available=True, task=task)
    else:
        logger.debug("No task available for client.")
        return TaskIfAny(is_available=False)


@app.get("/resources/latest", response_model=ResourcesUpdate)
async def fetch_latest_resources() -> ResourcesUpdate:
    """Endpoint for clients to poll for the latest available resources."""
    store = get_server_store()
    resources_update = await store.get_latest_resources()
    if not resources_update:
        raise HTTPException(status_code=404, detail="No resources have been set on the server.")
    logger.debug(f"Serving latest resources '{resources_update.resources_id}' to a client.")
    return resources_update


@app.get("/resources/{resource_id}", response_model=ResourcesUpdate)
async def fetch_resources_by_id(
    resource_id: str = Path(..., description="The unique identifier for the resource version.")
) -> ResourcesUpdate:
    """Endpoint for clients to fetch a specific version of resources."""
    store = get_server_store()
    resources_update = await store.get_resources_by_id(resource_id)
    if not resources_update:
        raise HTTPException(status_code=404, detail=f"Resource ID '{resource_id}' not found.")
    logger.debug(f"Serving resources for ID '{resource_id}' to a client.")
    return resources_update


@app.post("/rollout", response_model=GenericResponse)
async def post_rollout(payload: Rollout) -> GenericResponse:
    """Endpoint for clients to report a completed rollout."""
    await get_server_store().store_rollout(payload)
    return GenericResponse(
        status="ok",
        message=f"Rollout {payload.rollout_id} received and stored.",
    )


class AgentLightningServer:
    """
    The main SDK class for developers to control the Agent Lightning Server.

    This class manages the server lifecycle, task queueing, resources updates,
    and retrieval of results, providing a simple interface for the optimization logic.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """Initializes the server controller."""
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}"
        self._store = get_server_store()
        self._uvicorn_config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        self._uvicorn_server = uvicorn.Server(self._uvicorn_config)

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

    async def queue_task(self, sample: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Adds a task to the queue for a client to process.

        The task will be automatically associated with the latest resource version.

        Args:
            sample: The data sample for the task (e.g., a dictionary with a prompt).
            metadata: A dictionary with additional context for the task (e.g., mode: 'train').

        Returns:
            A unique `rollout_id` for tracking the task.
        """
        task_metadata = TaskMetadata(**(metadata or {}))
        return await self._store.add_task(sample, task_metadata)

    async def update_resources(self, resources: NamedResources) -> str:
        """
        Updates the resources, creating a new version and setting it as the latest.

        Args:
            resources: A `NamedResources` object containing the full set of
                       resources (e.g., LLMs, prompts) for the next tasks.

        Returns:
            The unique `resources_id` for the newly created resource version.
        """
        resources_id = f"res-{uuid.uuid4()}"
        update = ResourcesUpdate(resources_id=resources_id, resources=resources)
        await self._store.update_resources(update)
        return resources_id

    async def get_completed_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """
        Retrieves a specific completed rollout by its ID.
        The rollout is removed from the store once retrieved.

        Args:
            rollout_id: The ID of the task to retrieve.

        Returns:
            The rollout payload if found, otherwise None.
        """
        return await self._store.retrieve_rollout(rollout_id)

    async def poll_completed_rollout(self, rollout_id: str, timeout: Optional[float] = None) -> Optional[Rollout]:
        """
        Polls for a completed rollout by its ID, waiting up to `timeout` seconds.

        Args:
            rollout_id: The ID of the task to retrieve.
            timeout: Optional timeout in seconds to wait for the rollout.

        Returns:
            The rollout payload if found, otherwise None.
        """
        start_time = time.time()
        while True:
            rollout = await self.get_completed_rollout(rollout_id)
            if rollout:
                return rollout
            if timeout and (time.time() - start_time) >= timeout:
                return None
            await asyncio.sleep(1)

    async def retrieve_completed_rollouts(self) -> List[Rollout]:
        """
        Retrieves all available completed trajectories and clears the internal store.
        This is useful for batch processing results in an optimization loop.

        Returns:
            A list of all completed rollouts.
        """
        return await self._store.retrieve_completed_rollouts()
