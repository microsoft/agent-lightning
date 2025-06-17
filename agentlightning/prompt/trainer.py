import asyncio
import logging
import multiprocessing
import os
import signal
import time
from typing import List, Optional

import agentops

from agentlightning.instrumentation.agentops import AgentOpsServerManager
from agentlightning.instrumentation import instrument_all
from .client import AgentLightningClient
from .litagent import LitAgent
from .runner import AgentRunner


logger = logging.getLogger(__name__)


class Trainer:
    """Orchestrates the distributed execution of agent rollouts.

    The Trainer is responsible for launching one or more worker processes
    that run the agent's execution loop. It manages multiprocessing,
    handles graceful shutdown, and serves as the main entry point for
    running a client-side agent fleet.

    Attributes:
        n_workers: Number of agent workers (processes) to run in parallel.
        max_tasks: Maximum number of tasks to process per worker. If None,
                   workers run until no more tasks are available.
        daemon: Whether worker processes should be daemons. Daemon processes
                are terminated automatically when the main process exits.
        agentops_managed: Whether to automatically manage `agentops`.
                          When set to true, trainer calls `agentops.init()` automatically and launching an agentops endpoint locally,
                          in which case the agentops will not behave normally.
                          If not, you are responsible for calling and using it before using the trainer.
        instrument_managed: Whether to automatically manage instrumentation.
                            When set to false, you will manage the instrumentation yourself and the trainer might not work as expected.
    """

    def __init__(
        self,
        *,
        n_workers: int = 1,
        max_tasks: Optional[int] = None,
        daemon: bool = True,
        agentops_managed: bool = True,
        instrument_managed: bool = True,
    ):
        self.n_workers = n_workers
        self.max_tasks = max_tasks
        self.daemon = daemon
        self.agentops_managed = agentops_managed
        self.instrument_managed = instrument_managed

        self._agentops_server_manager: AgentOpsServerManager | None = None
        self._agentops_server_port_val: int | None = None  # Stores the picklable port number

        self._client: AgentLightningClient | None = None  # Will be initialized in fit method

        if self.agentops_managed:
            self._agentops_server_manager = AgentOpsServerManager(self.daemon)

        if not self.agentops_managed and self.n_workers > 1:
            logger.warning(
                "Using n_workers > 1 with agentops_managed=False. Ensure manual AgentOps setup is process-safe."
            )
        if not self.instrument_managed:
            logger.warning("instrument_managed=False. You are responsible for all instrumentation.")
        if not self.daemon:
            logger.warning(
                "daemon=False. Worker processes are non-daemonic. "
                "The worker processes will NOT be terminated when the main process exits. "
                "The cleanup must be handled manually."
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_agentops_server_manager"] = None  # Exclude the unpicklable server manager
        # _agentops_server_port_val (int) is inherently picklable and will be included.
        logger.debug(f"Getting state for pickling Trainer (PID {os.getpid()}). _agentops_server_manager excluded.")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # In child process, self._agentops_server_manager will be None.
        logger.debug(f"Setting state for unpickled Trainer (PID {os.getpid()}). _agentops_server_manager is None.")

    def init(self, endpoint: str) -> None:
        logger.info(f"Initializing Trainer...")

        self._init_client(endpoint)

        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.start()
            self._agentops_server_port_val = self._agentops_server_manager.get_port()
            if self._agentops_server_port_val is None:
                if (
                    self._agentops_server_manager.server_process is not None
                    and self._agentops_server_manager.server_process.is_alive()
                ):
                    raise RuntimeError("AgentOps server started but port is None. Check server manager logic.")
                elif (
                    self._agentops_server_port_val is None and self._agentops_server_manager.server_process is None
                ):  # Server failed to start
                    raise RuntimeError("AgentOps server manager indicates server is not running and port is None.")
        logger.info(f"Trainer main initialization complete.")

    def cleanup(self) -> None:
        logger.info(f"Cleaning up Trainer...")
        if self.agentops_managed and self._agentops_server_manager:
            self._agentops_server_manager.stop()
        self._verl_client = None
        logger.info(f"Trainer main cleanup complete.")

    def client(self) -> AgentLightningClient:
        """Returns the AgentLightningClient instance."""
        if self._client is None:
            raise RuntimeError("AgentLightningClient has not been initialized. Call `init` first.")
        return self._client

    def _init_client(self, endpoint: str) -> AgentLightningClient:
        if self._client is None:
            self._client = AgentLightningClient(endpoint=endpoint)
        else:
            logger.warning("AgentLightningClient already initialized. Returning existing instance.")
        return self._client

    def _worker_main_loop(self, agent: LitAgent, worker_id: int, is_async: bool):
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

        mode = "Async" if is_async else "Sync"
        logger.info(f"[Worker {worker_id}] {mode} worker process started.")

        num_processed = 0
        self._initialize_worker_env(worker_id)

        try:
            client = self.client()
            loop = AgentRunner(
                agent=agent,
                client=client,
                max_tasks=self.max_tasks,
                worker_id=worker_id,
                agentops_managed=self.agentops_managed,
            )
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
        logger.info(f"[Worker {worker_id}] Setting up environment...")  # worker_id included in process name
        if self.agentops_managed:
            if self._agentops_server_port_val:  # Use the stored, picklable port value
                base_url = f"http://localhost:{self._agentops_server_port_val}"
                env_vars_to_set = {
                    "AGENTOPS_API_KEY": "dummy",
                    "AGENTOPS_API_ENDPOINT": base_url,
                    "AGENTOPS_APP_URL": f"{base_url}/notavailable",
                    "AGENTOPS_EXPORTER_ENDPOINT": f"{base_url}/traces",
                }
                for key, value in env_vars_to_set.items():
                    os.environ[key] = value
                    logger.info(f"[Worker {worker_id}] Env var set: {key}={value}")
            else:
                logger.warning(
                    f"[Worker {worker_id}] AgentOps managed, but local server port is not available. Client may not connect as expected."
                )

            if not agentops.get_client().initialized:
                agentops.init()
                logger.info(f"[Worker {worker_id}] AgentOps client initialized.")
            else:
                logger.warning(f"[Worker {worker_id}] AgentOps client was already initialized.")

        if self.instrument_managed:
            instrument_all()
            logger.info(f"[Worker {worker_id}] Instrumentation applied.")

    def _teardown_worker_env(self, worker_id: int):
        logger.info(f"[Worker {worker_id}] Cleaning up environment...")
        # Do nothing for now.
        logger.info(f"[Worker {worker_id}] Environment cleanup complete.")

    @staticmethod
    def kill_orphaned_processes() -> None:
        """
        Kill any orphaned processes that may have been left behind by previous runs.
        This is useful for cleaning up after crashes or unexpected exits.
        """
        import psutil

        for proc in psutil.process_iter():
            # check whether the process name matches
            if proc.name().startswith("AgentLightning-"):
                proc.kill()

    def fit(self, agent: LitAgent, endpoint: str) -> None:
        self.init(endpoint)
        processes: List[multiprocessing.Process] = []

        # Determine if the agent is asynchronous.
        is_async = (
            hasattr(agent, "training_rollout_async")
            and agent.__class__.training_rollout_async is not LitAgent.training_rollout_async
        )

        mode = "asynchronous" if is_async else "synchronous"

        try:
            if self.n_workers == 1:
                logger.info(f"Running with n_workers=1 ({mode} in main process).")
                num_tasks = self._worker_main_loop(agent, 0, is_async)
                logger.info(f"Single worker mode finished. Tasks processed: {num_tasks}")
            else:
                logger.info(f"Running with n_workers={self.n_workers} ({mode} multiprocessing).")
                for i in range(self.n_workers):
                    process_name = f"AgentLightning-Worker-{i}"
                    p = multiprocessing.Process(
                        target=self._worker_main_loop,
                        args=(agent, i, is_async),
                        daemon=self.daemon,
                        name=process_name,
                    )
                    processes.append(p)
                    logger.info(f"Starting worker process {i} (name: {process_name})...")
                    p.start()

                if self.daemon:
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

        except KeyboardInterrupt:
            if self.n_workers > 1 and len(processes) > 0:
                logger.info(f"KeyboardInterrupt received. Terminating workers...")
                for i, p in enumerate(processes):
                    if p.is_alive():
                        logger.info(f"Terminating worker {i} (name: {p.name}, PID: {p.pid})...")
                        p.terminate()
                    else:
                        logger.info(
                            f"Worker {i} (name: {p.name}, PID: {p.pid}) is not alive or has already terminated."
                        )
                for i, p in enumerate(processes):
                    if p.is_alive():
                        p.join(timeout=10)  # Give some time to terminate
                    if p.is_alive():  # If still alive, kill
                        logger.warning(
                            f"Worker {i} (name: {p.name}, PID: {p.pid}) did not terminate gracefully, killing..."
                        )
                        p.kill()
                        p.join(timeout=10)  # Ensure it's reaped
            logger.info(f"Workers terminated or single worker interrupted.")
        except Exception as e:
            logger.exception(f"Unhandled exception in fit method.")
        finally:
            if self.daemon:
                self.cleanup()
            else:
                logger.info("Main process exiting. Please use Trainer.kill_orphaned_processes() for cleanup.")
