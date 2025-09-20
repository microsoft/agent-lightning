# Copyright (c) Microsoft. All rights reserved.

import asyncio
import importlib
import logging
import multiprocessing
import signal
import time
import warnings
from typing import Any, Dict, List, Optional, TypeVar, Union

from .adapter import TraceTripletAdapter
from .algorithm.base import BaseAlgorithm
from .client import AgentLightningClient
from .litagent import LitAgent
from .runner import AgentRunner
from .tracer.agentops import AgentOpsTracer
from .tracer.base import BaseTracer
from .types import Dataset, ParallelWorkerBase

logger = logging.getLogger(__name__)

T_co = TypeVar("T_co", covariant=True)


class Trainer(ParallelWorkerBase):
    """Orchestrates the distributed execution of agent rollouts.

    The Trainer is responsible for launching one or more worker processes
    that run the agent's execution loop. It manages multiprocessing,
    handles graceful shutdown, and serves as the main entry point for
    running a client-side agent fleet.

    Attributes:
        dev: If True, rollouts are run against the dev endpoint provided in `fit`.
        n_workers: Number of agent workers (processes) to run in parallel.
        max_tasks: Maximum number of tasks to process per worker. If None,
                   workers run until no more tasks are available.
        daemon: Whether worker processes should be daemons. Daemon processes
                are terminated automatically when the main process exits.
        tracer: A tracer instance, or a string pointing to the class full name or a dictionary with a 'type' key
                that specifies the class full name and other initialization parameters.
                If None, a default `AgentOpsTracer` will be created with the current settings.
        triplet_exporter: An instance of `TraceTripletAdapter` to export triplets from traces,
                          or a dictionary with the initialization parameters for the exporter.
        algorithm: An instance of `BaseAlgorithm` to use for training.
    """

    def __init__(
        self,
        *,
        dev: bool = False,
        n_workers: int = 1,
        max_tasks: Optional[int] = None,
        daemon: bool = True,
        tracer: Union[BaseTracer, str, Dict[str, Any], None] = None,
        triplet_exporter: Union[TraceTripletAdapter, Dict[str, Any], None] = None,
        algorithm: Union[BaseAlgorithm, str, Dict[str, Any], None] = None,
    ):
        super().__init__()
        self.n_workers = n_workers
        self.max_tasks = max_tasks
        self.daemon = daemon
        self.dev = dev
        self._client: AgentLightningClient | None = None  # Will be initialized in fit method

        self.tracer = self._make_tracer(tracer)
        if isinstance(triplet_exporter, TraceTripletAdapter):
            self.triplet_exporter = triplet_exporter
        elif isinstance(triplet_exporter, dict):
            self.triplet_exporter = TraceTripletAdapter(**triplet_exporter)
        elif triplet_exporter is None:
            self.triplet_exporter = TraceTripletAdapter()
        else:
            raise ValueError(
                f"Invalid triplet_exporter type: {type(triplet_exporter)}. Expected TraceTripletAdapter, dict, or None."
            )

        self.algorithm = self._make_algorithm(algorithm)

        if not self.daemon:
            logger.warning(
                "daemon=False. Worker processes are non-daemonic. "
                "The worker processes will NOT be terminated when the main process exits. "
                "The cleanup must be handled manually."
            )

    def _make_tracer(self, tracer: Union[BaseTracer, str, Dict[str, Any], None]) -> BaseTracer:
        """Creates a tracer instance based on the provided configuration."""
        if isinstance(tracer, BaseTracer):
            return tracer
        if isinstance(tracer, str):
            module_name, class_name = tracer.rsplit(".", 1)
            module = importlib.import_module(module_name)
            tracer_cls = getattr(module, class_name)
            return tracer_cls()
        if isinstance(tracer, dict):
            tracer_type = tracer.get("type")
            if tracer_type is None:
                raise ValueError("tracer dict must have a 'type' key with the class full name")
            module_name, class_name = tracer_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            tracer_cls = getattr(module, class_name)
            # Remove 'type' key and pass remaining keys as kwargs
            tracer_kwargs = {k: v for k, v in tracer.items() if k != "type"}
            return tracer_cls(**tracer_kwargs)
        if tracer is None:
            return AgentOpsTracer(agentops_managed=True, instrument_managed=True, daemon=self.daemon)
        raise ValueError(f"Invalid tracer type: {type(tracer)}. Expected BaseTracer, str, dict, or None.")

    def _make_algorithm(self, algorithm: Union[BaseAlgorithm, str, Dict[str, Any], None]) -> Optional[BaseAlgorithm]:
        """Creates an algorithm instance based on the provided configuration."""
        if isinstance(algorithm, BaseAlgorithm):
            return algorithm
        if isinstance(algorithm, str):
            module_name, class_name = algorithm.rsplit(".", 1)
            module = importlib.import_module(module_name)
            algorithm_cls = getattr(module, class_name)
            return algorithm_cls()
        if isinstance(algorithm, dict):
            algorithm_type = algorithm.get("type")
            if algorithm_type is None:
                raise ValueError("algorithm dict must have a 'type' key with the class full name")
            module_name, class_name = algorithm_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            algorithm_cls = getattr(module, class_name)
            # Remove 'type' key and pass remaining keys as kwargs
            algorithm_kwargs = {k: v for k, v in algorithm.items() if k != "type"}
            return algorithm_cls(**algorithm_kwargs)
        if algorithm is None:
            return None
        raise ValueError(f"Invalid algorithm type: {type(algorithm)}. Expected BaseAlgorithm, str, dict, or None.")

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
        if self.dev:
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
                # Algorithm will be responsible for creating the client
                client = self.algorithm.get_client()
                logger.info(f"Algorithm created client: {client}")
                return client
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
        if self.n_workers > 1:
            import setproctitle

            # Ignore Ctrl+C in worker processes; the main process handles it
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            setproctitle.setproctitle(multiprocessing.current_process().name)

        # Now we are in child processes, so we can safely set up the environment.
        agent.set_trainer(self)
        # TODO: this should be set elsewhere
        if agent.trained_agents:
            self.triplet_exporter.agent_match = agent.trained_agents
        self._initialize_worker_env(worker_id)

        mode = "Async" if is_async else "Sync"
        logger.info(f"[Worker {worker_id}] {mode} worker process started.")

        num_processed = 0

        try:
            client = self.client()
            loop = AgentRunner(
                agent=agent,
                client=client,
                tracer=self.tracer,
                triplet_exporter=self.triplet_exporter,
                max_tasks=self.max_tasks,
                worker_id=worker_id,
            )
            loop.init_worker(worker_id)
            if is_async:
                num_processed = asyncio.run(loop.iter_async())
            else:
                num_processed = loop.iter()
        except Exception:
            logger.exception(f"[Worker {worker_id}] Unhandled exception in worker loop.")
        finally:
            self._teardown_worker_env(worker_id)

        return num_processed

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

    def fit(
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
        the algorithm will be responsible for creating a client that will connect to itself.
        It can also create a mock client if the algorithm does not require a server.
        """

        if dev_backend is not None:
            warnings.warn("dev_backend is deprecated. Use dev_data instead.")
            if dev_data is not None:
                raise ValueError("dev_data and dev_backend cannot be provided at the same time.")
            dev_data = dev_backend

        # Extract datasets for algorithm if available
        train_dataset = self._extract_dataset_from_data(train_data)
        val_dataset = self._extract_dataset_from_data(val_data) if val_data else None
        dev_dataset = self._extract_dataset_from_data(dev_data) if dev_data else None

        # Initialize the algorithm with trainer if provided
        if self.algorithm is not None:
            self.algorithm.set_trainer(self)
            # DO NOT RUN TRAINING HERE. Need to spawn the worker first.

        # Determine the backend to use for client-server mode
        backend = self._determine_backend(train_data, dev_data)

        if self.dev:
            logger.warning(f"Running in dev mode. Using dev backend: {backend}")
        else:
            logger.debug(f"Running in non-dev mode. Using backend: {backend}")

        self.init(backend)

        processes: List[multiprocessing.Process] = []

        # Determine if the agent is asynchronous

        mode = "asynchronous" if agent.is_async else "synchronous"

        try:
            if self.n_workers == 1:
                logger.info(f"Running with n_workers=1 ({mode} in main process).")

                # Warn if algorithm is set with single worker mode
                if self.algorithm is not None:
                    logger.warning(
                        "Algorithm is set but using single worker mode. Algorithm will never get the chance to run."
                    )
                    # Ideally the single worker should be run in a separate thread or process.

                num_tasks = self._worker_main_loop(agent, 0, agent.is_async)
                logger.info(f"Single worker mode finished. Tasks processed: {num_tasks}")

                # If algorithm is provided and we have datasets, run algorithm after worker completes
                if self.algorithm is not None and train_dataset is not None:
                    logger.info("Running algorithm training after worker completion.")
                    self.algorithm.run(
                        train_dataset=train_dataset,
                        validation_dataset=val_dataset,
                        dev_dataset=dev_dataset,
                    )
            else:
                logger.info(f"Running with n_workers={self.n_workers} ({mode} multiprocessing).")
                for i in range(self.n_workers):
                    process_name = f"AgentLightning-Worker-{i}"
                    p = multiprocessing.Process(
                        target=self._worker_main_loop,
                        args=(agent, i, agent.is_async),
                        daemon=self.daemon,
                        name=process_name,
                    )
                    processes.append(p)
                    logger.info(f"Starting worker process {i} (name: {process_name})...")
                    p.start()

                if self.daemon:
                    # If algorithm is provided and we have datasets, pass them to the algorithm
                    if self.algorithm is not None:
                        logger.info("All workers have been spawned. Running algorithm training with provided datasets.")
                        self.algorithm.run(
                            train_dataset=train_dataset,
                            validation_dataset=val_dataset,
                            dev_dataset=dev_dataset,
                        )
                        logger.info("Algorithm exits. Killing the workers.")
                        self._terminate_processes(processes)

                    for i, p in enumerate(processes):
                        p.join()  # Wait for the process to complete
                        logger.info(
                            f"Worker process {i} (name: {p.name}, PID: {p.pid}) joined with exit code {p.exitcode}."
                        )
                        if p.exitcode != 0:
                            logger.warning(
                                f"Worker process {i} (name: {p.name}, PID: {p.pid}) exited with non-zero code: {p.exitcode}."
                            )

                    logger.info(f"All {self.n_workers} worker processes have completed.")
                else:
                    logger.info("All worker processes started. Main process will not wait.")

                    # A hack to stop the main process from waiting for child processes to finish.
                    time.sleep(1)  # Give workers time to start
                    import multiprocessing.process as multiprocessing_process

                    multiprocessing_process._children.clear()  # type: ignore

                    if self.algorithm is not None:
                        logger.info("Main process continues to run algorithm.")
                        self.algorithm.run(
                            train_dataset=train_dataset,
                            validation_dataset=val_dataset,
                            dev_dataset=dev_dataset,
                        )
                        logger.info("Algorithm exits. Killing the workers.")
                        self._terminate_processes(processes)

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Killing the workers.")
            self._terminate_processes(processes)
            logger.info(f"Workers terminated or single worker interrupted.")
            raise
        except Exception:
            logger.exception(f"Unhandled exception in fit method.")
            self._terminate_processes(processes)
            logger.info(f"Workers terminated or single worker interrupted.")
            raise
        finally:
            if self.daemon:
                self.teardown()
            else:
                logger.info("Main process exiting. Please use Trainer.kill_orphaned_processes() for cleanup.")
