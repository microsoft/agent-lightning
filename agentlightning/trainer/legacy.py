# Copyright (c) Microsoft. All rights reserved.

import logging
import multiprocessing
import time
from typing import Any, List, Optional, TypeVar, Union

from agentlightning.adapter import TraceAdapter
from agentlightning.algorithm import Algorithm
from agentlightning.client import AgentLightningClient
from agentlightning.litagent import LitAgent
from agentlightning.tracer.base import Tracer
from agentlightning.types import Dataset, ParallelWorkerBase

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class TrainerLegacy(ParallelWorkerBase):
    """Trainer for legacy mode for v0.1 compatibility."""

    def __init__(self, *args: Any, **kwargs: Any):
        """Initialize the TrainerLegacy.

        This method is mainly to make type checker happy.
        It won't be used in practice.
        """
        self._dev = kwargs.pop("dev", False)
        self.algorithm: Optional[Algorithm] = kwargs.pop("algorithm", None)
        self.tracer: Tracer = kwargs.pop("tracer", None)
        self.n_workers: int = kwargs.pop("n_workers", None)
        self.max_tasks: Optional[int] = kwargs.pop("max_tasks", None)
        self.daemon: bool = kwargs.pop("daemon", True)
        self.triplet_exporter: TraceAdapter[Any] = kwargs.pop("triplet_exporter", None)

    def _extract_client_from_data(
        self, data: Union[str, AgentLightningClient, Dataset[Any]]
    ) -> Optional[AgentLightningClient]:
        """Extract client from data if it's a string URL or AgentLightningClient."""
        if isinstance(data, str):
            if not data.startswith("http://") and not data.startswith("https://"):
                raise ValueError("String data must be a valid URL starting with http:// or https://")
            return AgentLightningClient(endpoint=data)
        elif isinstance(data, AgentLightningClient):
            return data
        return None

    def _extract_dataset_from_data(
        self, data: Union[str, AgentLightningClient, Dataset[Any]]
    ) -> Optional[Dataset[Any]]:
        """Extract dataset from data if it's a Dataset."""
        if isinstance(data, str) or isinstance(data, AgentLightningClient):
            return None
        return data

    def _determine_backend(
        self,
        train_data: Union[str, AgentLightningClient, Dataset[Any]],
        dev_data: Union[str, AgentLightningClient, Dataset[Any], None] = None,
    ) -> Union[str, AgentLightningClient]:
        """Determine which backend to use for initialization."""
        if self._dev:
            if dev_data is None:
                raise ValueError("dev_data must be provided when dev=True.")
            client = self._extract_client_from_data(dev_data)
            if client is None:
                raise ValueError("dev_data must be a string URL or AgentLightningClient when dev=True.")
            return client
        else:
            client = self._extract_client_from_data(train_data)
            if client is None and self.algorithm is None:
                raise ValueError(
                    "train_data must be a string URL or AgentLightningClient when no algorithm is provided."
                )
            elif client is None and self.algorithm is not None:
                # algorithm no longer owns legacy client creation; caller must pass an endpoint.
                raise ValueError(
                    "Algorithm.get_client() is no longer supported. "
                    "Pass a server URL or AgentLightningClient in train_data when using legacy Trainer flow."
                )
            if client is None:
                raise ValueError(
                    "train_data must be a string URL or AgentLightningClient when no algorithm is provided."
                )
            return client

    def init(self, backend: Union[str, AgentLightningClient]) -> None:
        logger.info(f"Initializing Trainer...")

        self._init_client(backend)

        self.tracer.init()

        logger.info(f"Trainer main initialization complete.")

    def teardown(self) -> None:
        logger.info(f"Cleaning up Trainer...")
        self.tracer.teardown()

        self._client = None
        logger.info(f"Trainer main cleanup complete.")

    def client(self) -> AgentLightningClient:
        """Returns the AgentLightningClient instance."""
        if self._client is None:
            raise RuntimeError("AgentLightningClient has not been initialized. Call `init` first.")
        return self._client

    def _init_client(self, backend: Union[str, AgentLightningClient]) -> AgentLightningClient:
        if self._client is None:
            if isinstance(backend, AgentLightningClient):
                logger.info("Using provided AgentLightningClient instance.")
                self._client = backend
            else:
                logger.info(f"Initializing AgentLightningClient with endpoint: {backend}")
                if not isinstance(backend, str):  # type: ignore
                    raise ValueError("backend must be a string URL or an AgentLightningClient instance.")
                if not backend.startswith("http://") and not backend.startswith("https://"):
                    raise ValueError("backend must be a valid URL starting with http:// or https://")
                # Initialize the client with the provided backend URL
                self._client = AgentLightningClient(endpoint=backend)
        else:
            logger.warning("AgentLightningClient already initialized. Returning existing instance.")
        return self._client

    def _worker_main_loop(self, agent: LitAgent[Any], worker_id: int, is_async: bool):
        """The main function for each worker process.

        This function initializes the client and the loop, then starts the
        execution. It also configures process-specific settings like the
        process title and signal handling.

        Args:
            agent: The `LitAgent` instance to run.
            worker_id: The unique ID for this worker.
            is_async: A boolean indicating if the async loop should be run.
        """
        raise RuntimeError("Legacy worker loop is removed in this major version.")

    def _initialize_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Setting up trainer environment...")  # worker_id included in process name
        self.tracer.init_worker(worker_id)

    def _teardown_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Cleaning up trainer environment...")
        self.tracer.teardown_worker(worker_id)
        logger.info(f"[Worker {worker_id}] Environment cleanup complete.")

    @staticmethod
    def kill_orphaned_processes() -> None:
        """
        Kill any orphaned processes that may have been left behind by previous runs.
        This is useful for cleaning up after crashes or unexpected exits.
        """
        import psutil

        for proc in psutil.process_iter():  # type: ignore
            # check whether the process name matches
            if proc.name().startswith("AgentLightning-"):
                proc.kill()

    def _terminate_processes(self, processes: List[multiprocessing.Process]) -> None:
        if self.n_workers > 1 and len(processes) > 0:
            for i, p in enumerate(processes):
                if p.is_alive():
                    logger.info(f"Terminating worker {i} (name: {p.name}, PID: {p.pid})...")
                    p.terminate()
                else:
                    logger.info(f"Worker {i} (name: {p.name}, PID: {p.pid}) is not alive or has already terminated.")
            for i, p in enumerate(processes):
                if p.is_alive():
                    p.join(timeout=10)  # Give some time to terminate
                if p.is_alive():  # If still alive, kill
                    logger.warning(
                        f"Worker {i} (name: {p.name}, PID: {p.pid}) did not terminate gracefully, killing..."
                    )
                    p.kill()
                    p.join(timeout=10)  # Ensure it's reaped

    def fit_v0(
        self,
        agent: LitAgent[T_co],
        train_data: Union[str, AgentLightningClient, Dataset[T_co]],
        *,
        val_data: Union[str, AgentLightningClient, Dataset[T_co], None] = None,
        dev_data: Union[str, AgentLightningClient, Dataset[T_co], None] = None,
        dev_backend: Union[str, AgentLightningClient, None] = None,
    ):
        """Train the agent using the provided data.

        Each data argument can be a string URL connecting to a agent-lightning server,
        or an AgentLightningClient instance connecting to a server (or mock server), or a dataset.
        If no algorithm is provided when instantiating the trainer, the data must be
        provided to connecting a server. Otherwise, dataset is also allowed and will be
        passed to the algorithm.

        If the algorithm is instantiated and there is no URL/client provided,
        this legacy path requires callers to provide a server client URL; algorithm-owned client
        creation is no longer supported.
        """
        raise RuntimeError("fit_v0 is removed in this major version. Use Trainer.fit().")
