import asyncio
import logging
import time
import uuid
import threading
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Literal

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
)

logger = logging.getLogger(__name__)


class ServerDataStore:
    """
    A centralized, thread-safe, async, in-memory data store for the server's state.
    This holds the task queue, versioned resources, and completed rollouts.
    """

    def __init__(self):
        self._task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self._processing_tasks: Dict[str, Task] = {}  # Currently processing tasks
        self._completed_rollouts: Dict[str, Rollout] = {}

        # Store for versioned resources
        self._resource_versions: Dict[str, NamedResources] = {}
        self._latest_resources_id: Optional[str] = None

        # Locks for thread-safe access
        self._results_lock = asyncio.Lock()
        self._resources_lock = asyncio.Lock()

    async def add_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """
        Adds a new task to the queue with specific metadata and returns its unique ID.
        """
        rollout_id = f"rollout-{uuid.uuid4()}"
        task = Task(
            rollout_id=rollout_id,
            input=sample,
            mode=mode,
            resources_id=resources_id,
            create_time=time.time(),
            num_claims=0,
            metadata=metadata or {},
        )
        await self._task_queue.put(task)
        logger.info(f"Task queued: {rollout_id} (mode: {mode}, resources_id: {resources_id})")
        return rollout_id

    async def get_next_task(self) -> Optional[Task]:
        """
        Retrieves the next task from the queue without blocking.
        Returns None if the queue is empty.
        """
        try:
            async with self._results_lock:
                task = self._task_queue.get_nowait()
                task = task.model_copy(
                    update={
                        "last_claim_time": time.time(),
                        "num_claims": (task.num_claims or 0) + 1,
                    }
                )
                self._processing_tasks[task.rollout_id] = task
                if task.num_claims == 1:
                    logger.debug(f"Next task retrieved: {task.rollout_id}")
                else:
                    logger.info(f"Task {task.rollout_id} re-claimed (attempt {task.num_claims})")
                return task
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
        if self._latest_resources_id:
            return await self.get_resources_by_id(self._latest_resources_id)
        return None

    async def store_rollout(self, rollout: Rollout):
        """
        Safely stores a completed rollout from a client.
        """
        async with self._results_lock:
            self._processing_tasks.pop(rollout.rollout_id, None)
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

    def get_processing_tasks(self) -> Dict[str, Task]:
        """Returns a copy of currently processing tasks for timeout checking."""
        return self._processing_tasks.copy()

    async def requeue_task(self, task: Task):
        """Requeues a task that has timed out and removes it from processing."""
        logger.warning(f"Requeuing task {task.rollout_id} after timeout (attempt {task.num_claims})")
        async with self._results_lock:
            # Remove from processing tasks
            self._processing_tasks.pop(task.rollout_id, None)
            self._task_queue.put_nowait(task)


class AgentLightningServer:
    """
    The main SDK class for developers to control the Agent Lightning Server.

    This class manages the server lifecycle, task queueing, resources updates,
    and retrieval of results, providing a simple interface for the optimization logic.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, task_timeout_seconds: float = 300.0):
        """
        Initializes the server controller.

        Args:
            host: The host to bind the server to.
            port: The port to bind the server to.
            task_timeout_seconds: Time in seconds after which a claimed task is considered stale and requeued.
        """
        self.host = host
        self.port = port
        self.endpoint = f"http://{host}:{port}"
        self._task_timeout_seconds = task_timeout_seconds

        # Defer initialization and use event for cross-thread communication
        self._store: Optional[ServerDataStore] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.startup_event = threading.Event()

        # Create FastAPI app instance with a lifespan manager
        self._app = FastAPI(lifespan=self._lifespan)
        self._setup_routes()

        self._uvicorn_config = uvicorn.Config(self._app, host=self.host, port=self.port, log_level="info")
        self._uvicorn_server = uvicorn.Server(self._uvicorn_config)

    # --- ADDED: Lifespan context manager ---
    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """
        Manages server startup and shutdown. This runs inside the server's event loop.
        """
        logger.info("Server is starting up...")
        self.loop = asyncio.get_running_loop()
        self._store = ServerDataStore()  # Initialize data store here
        self.startup_event.set()  # Signal that the server is ready

        yield

        logger.info("Server is shutting down.")
        self._store = None
        self.startup_event.clear()  # Clear the startup event
        self.loop = None

    async def _check_and_requeue_stale_tasks(self):
        """
        Check for stale tasks and requeue them. Called reactively during get_next_task.
        """
        current_time = time.time()
        # Ensure store is initialized before checking
        if not self._store:
            return
        processing_tasks = self._store.get_processing_tasks()

        for rollout_id, task in processing_tasks.items():
            if task.last_claim_time and current_time - task.last_claim_time > self._task_timeout_seconds:
                await self._store.requeue_task(task)
                logger.warning(
                    f"Task {task.rollout_id} timed out after {self._task_timeout_seconds}s, requeued (attempt {task.num_claims})"
                )

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self._app.get("/task", response_model=TaskIfAny)
        async def next_task() -> TaskIfAny:
            """Endpoint for clients to poll for the next available task."""
            await self._check_and_requeue_stale_tasks()

            if not self._store:
                return TaskIfAny(is_available=False)

            task = await self._store.get_next_task()
            if task:
                logger.debug(f"Serving task {task.rollout_id} to a client.")
                return TaskIfAny(is_available=True, task=task)
            else:
                logger.debug("No task available for client.")
                return TaskIfAny(is_available=False)

        @self._app.get("/resources/latest", response_model=ResourcesUpdate)
        async def fetch_latest_resources() -> ResourcesUpdate:
            """Endpoint for clients to poll for the latest available resources."""
            if not self._store:
                raise HTTPException(status_code=503, detail="Server not fully initialized.")
            resources_update = await self._store.get_latest_resources()
            if not resources_update:
                raise HTTPException(status_code=404, detail="No resources have been set on the server.")
            logger.debug(f"Serving latest resources '{resources_update.resources_id}' to a client.")
            return resources_update

        @self._app.get("/resources/{resource_id}", response_model=ResourcesUpdate)
        async def fetch_resources_by_id(
            resource_id: str = Path(..., description="The unique identifier for the resource version.")
        ) -> ResourcesUpdate:
            """Endpoint for clients to fetch a specific version of resources."""
            if not self._store:
                raise HTTPException(status_code=503, detail="Server not fully initialized.")
            resources_update = await self._store.get_resources_by_id(resource_id)
            if not resources_update:
                raise HTTPException(status_code=404, detail=f"Resource ID '{resource_id}' not found.")
            logger.debug(f"Serving resources for ID '{resource_id}' to a client.")
            return resources_update

        @self._app.post("/rollout", response_model=GenericResponse)
        async def post_rollout(payload: Rollout) -> GenericResponse:
            """Endpoint for clients to report a completed rollout."""
            if not self._store:
                raise HTTPException(status_code=503, detail="Server not fully initialized.")
            await self._store.store_rollout(payload)
            return GenericResponse(
                status="ok",
                message=f"Rollout {payload.rollout_id} received and stored.",
            )

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

    async def run_forever(self):
        """
        Runs the server indefinitely until stopped.
        This is useful when async start and stop methods do not work.
        """
        await self._uvicorn_server.serve()

    async def queue_task(
        self,
        sample: Any,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        """
        Adds a task to the queue for a client to process.
        """
        if not self._store:
            raise RuntimeError("Store not initialized. The server may not be running.")
        return await self._store.add_task(sample, mode=mode, resources_id=resources_id, metadata=metadata)

    async def update_resources(self, resources: NamedResources) -> str:
        """
        Updates the resources, creating a new version and setting it as the latest.
        """
        if not self._store:
            raise RuntimeError("Store not initialized. The server may not be running.")
        resources_id = f"res-{uuid.uuid4()}"
        update = ResourcesUpdate(resources_id=resources_id, resources=resources)
        await self._store.update_resources(update)
        return resources_id

    async def get_completed_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """
        Retrieves a specific completed rollout by its ID.
        """
        if not self._store:
            raise RuntimeError("Store not initialized. The server may not be running.")
        return await self._store.retrieve_rollout(rollout_id)

    async def poll_completed_rollout(self, rollout_id: str, timeout: Optional[float] = None) -> Optional[Rollout]:
        """
        Polls for a completed rollout by its ID, waiting up to `timeout` seconds.
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
        """
        if not self._store:
            raise RuntimeError("Store not initialized. The server may not be running.")
        return await self._store.retrieve_completed_rollouts()
