import asyncio
import logging
import multiprocessing
import signal
import time
from typing import List

import psutil

from .client import AgentLightningClient
from .litagent import LitAgent
from .runner import Loop

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
    """

    def __init__(self, n_workers: int = 1, max_tasks: int | None = None, daemon: bool = True):
        """Initializes the Trainer.

        Args:
            n_workers: The number of parallel agent processes to launch.
            max_tasks: An optional limit on tasks per worker.
            daemon: Whether to run worker processes as daemons.
        """
        self.n_workers = n_workers
        self.max_tasks = max_tasks
        self.daemon = daemon

    def _worker_main_loop(self, agent: LitAgent, endpoint: str, worker_id: int, is_async: bool):
        """The main function for each worker process.

        This function initializes the client and the loop, then starts the
        execution. It also configures process-specific settings like the
        process title and signal handling.

        Args:
            agent: The `LitAgent` instance to run.
            endpoint: The URL of the Agent Lightning server.
            worker_id: The unique ID for this worker.
            is_async: A boolean indicating if the async loop should be run.
        """
        if self.n_workers > 1:
            import setproctitle

            # Ignore Ctrl+C in worker processes; the main process handles it
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            setproctitle.setproctitle(multiprocessing.current_process().name)

        mode = "Async" if is_async else "Sync"
        logger.info(f"[Worker {worker_id}] {mode} worker process started.")

        try:
            client = AgentLightningClient(endpoint=endpoint)
            loop = Loop(agent=agent, client=client, max_tasks=self.max_tasks)
            if is_async:
                asyncio.run(loop.run_async(worker_id))
            else:
                loop.run(worker_id)
        except Exception:
            logger.exception(f"[Worker {worker_id}] Unhandled exception in worker loop.")

    def fit(self, agent: LitAgent, endpoint: str):
        """Starts the training and execution process.

        This method launches the specified number of worker processes to
        connect to the server and execute tasks using the provided agent's
        logic.

        Args:
            agent: An instance of a `LitAgent` subclass.
            endpoint: The root URL of the Agent Lightning server.
        """
        # Determine if the agent is asynchronous by checking for an override
        is_async = agent.__class__.training_rollout_async is not LitAgent.training_rollout_async

        mode = "asynchronous" if is_async else "synchronous"
        processes: List[multiprocessing.Process] = []

        try:
            if self.n_workers <= 1:
                logger.info(f"Running with n_workers=1 ({mode} in main process).")
                self._worker_main_loop(agent, endpoint, 0, is_async)
            else:
                logger.info(f"Running with n_workers={self.n_workers} ({mode} multiprocessing).")
                for i in range(self.n_workers):
                    process_name = f"AgentLightning-Worker-{i}"
                    p = multiprocessing.Process(
                        target=self._worker_main_loop,
                        args=(agent, endpoint, i, is_async),
                        daemon=self.daemon,
                        name=process_name,
                    )
                    processes.append(p)
                    p.start()
                    logger.info(f"Started worker process {i} (name: {process_name}, PID: {p.pid}).")

                # Wait for all daemon processes to finish
                for i, p in enumerate(processes):
                    p.join()
                    logger.info(f"Worker process {i} (PID: {p.pid}) joined with exit code {p.exitcode}.")

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Terminating worker processes...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    logger.info(f"Terminating worker {i} (PID: {p.pid})...")
                    p.terminate()
            # Wait for processes to terminate
            for p in processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()  # Force kill if termination fails
                    p.join()
            logger.info("All worker processes terminated.")
        except Exception:
            logger.exception("An unhandled exception occurred in the main trainer process.")
        finally:
            logger.info("Trainer `fit` method finished.")

    @staticmethod
    def kill_orphaned_processes():
        """Finds and terminates any orphaned AgentLightning worker processes.

        This is a utility method for cleaning up stray processes that might
        be left running after an abnormal termination.
        """
        killed_count = 0
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                if proc.name().startswith("AgentLightning-Worker-"):
                    logger.warning(f"Found orphaned process: {proc.name()} (PID: {proc.pid}). Terminating.")
                    proc.kill()
                    killed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if killed_count > 0:
            logger.info(f"Terminated {killed_count} orphaned processes.")
        else:
            logger.info("No orphaned processes found.")
