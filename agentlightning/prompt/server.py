import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, TypedDict, cast

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from agentlightning.client import TaskData, SamplingParameters, Transition

from .resource import NamedResources, PromptTemplate


logger = logging.getLogger(__name__)


class TrajectoryPayload(TypedDict):
    """
    Represents a completed trajectory from a client.

    This includes the rollout ID and trace of transitions.

    # TBD: is it necessary to include the full trace here?
    """

    rollout_id: str
    transitions: List[Transition]


class ServerDataStore:
    """
    A centralized, thread-safe, async, in-memory data store for the server's state.
    This holds the task queue, resources, and completed traces.
    """

    def __init__(self):
        self._task_queue: asyncio.Queue[TaskData] = asyncio.Queue()
        self._named_resources: NamedResources = NamedResources()
        self._completed_rollouts: Dict[str, TrajectoryPayload] = {}

        # Locks for thread-safe access to non-queue resources
        self._trajectories_lock = asyncio.Lock()
        self._resources_lock = asyncio.Lock()

    async def add_task(self, sample: Any, is_train: bool) -> str:
        """
        Adds a new task to the queue and returns its unique rollout_id.
        """
        rollout_id = f"rollout-{uuid.uuid4()}"
        task = cast(TaskData, {"rollout_id": rollout_id, "is_train": is_train, **sample})
        await self._task_queue.put(task)
        logger.info(f"Task queued: {rollout_id} (is_train: {is_train})")
        return rollout_id

    async def get_next_task(self) -> Optional[TaskData]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.
        """
        # TODO: send the task back to queue if timeout.
        try:
            return self._task_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def update_named_resources(self, resources: NamedResources):
        """
        Safely updates the named resources available for tasks.
        """
        async with self._resources_lock:
            self._named_resources = resources
            logger.info(f"Named resources updated: {resources.dump()}")

    async def get_named_resources(self) -> NamedResources:
        """
        Safely retrieves the current named resources.
        """
        async with self._resources_lock:
            return NamedResources(**self._named_resources)

    async def store_trajectory(self, trajectory: TrajectoryPayload):
        """
        Safely stores a completed trajectory from a client.
        """
        async with self._trajectories_lock:
            self._completed_rollouts[trajectory["rollout_id"]] = trajectory
            logger.info(f"Trajectory received and stored for rollout: {trajectory['rollout_id']}")

    async def retrieve_trajectory(self, rollout_id: str) -> Optional[TrajectoryPayload]:
        """
        Safely retrieves a single trajectory by its ID.
        """
        async with self._trajectories_lock:
            return self._completed_rollouts.pop(rollout_id, None)

    async def retrieve_all_trajectories(self) -> List[TrajectoryPayload]:
        """
        Retrieves all completed trajectories and clears the store.
        """
        async with self._trajectories_lock:
            trajectories = list(self._completed_rollouts.values())
            self._completed_rollouts.clear()
            return trajectories


server_store: Optional[ServerDataStore] = None


def get_server_store() -> ServerDataStore:
    """
    Returns the global server data store instance.
    This is used to access the shared state across the FastAPI app.
    """
    global server_store
    if not server_store:
        logger.debug("Creating a new ServerDataStore instance.")
        server_store = ServerDataStore()
    return server_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to handle server startup and shutdown logic.
    """
    logger.info("Agent Lightning Server is starting up...")
    yield
    logger.info("Agent Lightning Server is shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/task")
async def next_task():
    """
    Endpoint for clients to poll for the next available task.
    """
    task = await get_server_store().get_next_task()
    if task:
        logger.debug(f"Serving task {task['rollout_id']} to a client.")
        return {"is_available": True, "data": task}
    else:
        logger.debug("No task available for client.")
        return {"is_available": False, "data": None}


@app.get("/resources")
async def fetch_resources():
    """
    Endpoint for clients to poll for the latest sampling parameters.
    """
    resources = await get_server_store().get_named_resources()
    logger.debug(f"Serving resources to a client: {resources}")
    return resources.dump()


@app.post("/trajectory")
async def post_trajectory(payload: dict):  # FIXME: type annotation
    """
    Endpoint for clients to report a completed trajectory.
    """
    await get_server_store().store_trajectory(payload)
    return {"status": "ok", "message": f"Trajectory for {payload['rollout_id']} received."}


class AgentLightningServer:
    """
    The main SDK class for developers to control the Agent Lightning Server.

    This class manages the server lifecycle, task queueing, resources updates,
    and retrieval of results, providing a simple interface for the optimization logic.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000):
        """
        Initializes the server controller.

        Args:
            host: The host address to run the server on.
            port: The port to run the server on.
        """
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}"
        self._server_process = None

        # The server uses the global `server_store` instance.
        self._store = get_server_store()

        # Uvicorn server configuration
        self._uvicorn_config = uvicorn.Config(app, host=self.host, port=self.port, log_level="info")
        self._uvicorn_server = uvicorn.Server(self._uvicorn_config)

    async def start(self):
        """
        Starts the FastAPI server in the background.

        This method needs to be run within an existing asyncio event loop.
        """
        logger.info(f"Starting server at {self.endpoint}")
        asyncio.create_task(self._uvicorn_server.serve())
        # A small delay to ensure the server is up and running before proceeding.
        await asyncio.sleep(1)

    async def stop(self):
        """
        Gracefully stops the running FastAPI server.
        """
        if self._uvicorn_server.started:
            logger.info("Stopping server...")
            self._uvicorn_server.should_exit = True
            # Give it a moment to shut down connections
            await asyncio.sleep(1)
            logger.info("Server stopped.")

    async def queue_task(self, sample: Dict[str, Any], is_train: bool = True) -> str:
        """
        Adds a task to the queue for a client to process.

        Args:
            sample: The data sample for the task (e.g., a dictionary with a prompt).
            is_train: A boolean indicating if this is a training or validation task.

        Returns:
            A unique `rollout_id` for tracking the task.
        """
        return await self._store.add_task(sample, is_train)

    async def update_resources(self, resources: NamedResources):
        """
        Updates the named resources that all clients will use for subsequent tasks.

        This replaces any existing resources with the new set provided.

        Args:
            resources: A `NamedResources` object containing the full set of
                       resources (e.g., LLMs, prompts) for the next tasks.
        """
        await self._store.update_named_resources(resources)

    async def get_completed_rollout(self, rollout_id: str) -> Optional[TrajectoryPayload]:
        """
        Retrieves a specific completed rollout by its ID.
        The rollout is removed from the store once retrieved.

        Args:
            rollout_id: The ID of the task to retrieve.

        Returns:
            The rollout payload if found, otherwise None.
        """
        return await self._store.retrieve_trajectory(rollout_id)

    async def poll_completed_rollout(
        self, rollout_id: str, timeout: Optional[float] = None
    ) -> Optional[TrajectoryPayload]:
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
            trajectory = await self.get_completed_rollout(rollout_id)
            if trajectory:
                return trajectory
            if timeout and (time.time() - start_time) >= timeout:
                return None
            await asyncio.sleep(1)

    async def get_all_completed_trajectories(self) -> List[TrajectoryPayload]:
        """
        Retrieves all available completed trajectories and clears the internal store.
        This is useful for batch processing results in an optimization loop.

        Returns:
            A list of all completed trajectory payloads.
        """
        return await self._store.retrieve_all_trajectories()
