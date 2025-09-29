# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import threading
from contextlib import suppress
from typing import Any, Awaitable, Literal, Optional, Protocol

from .store.base import LightningStore
from .store.client_server import LightningStoreClient, LightningStoreServer
from .store.threading import LightningStoreThreaded

logger = logging.getLogger(__name__)


class Event(Protocol):
    """
    A minimal protocol similar to threading.Event.

    Methods:
        set(): Signal event like a cancellation (idempotent).
        clear(): Reset to the non-set state.
        is_set() -> bool: True if event has been signaled.
        wait(timeout: Optional[float] = None) -> bool:
            Block until event is set or timeout. Returns True if event has signaled.
    """

    def set(self) -> None: ...
    def clear(self) -> None: ...
    def is_set(self) -> bool: ...
    def wait(self, timeout: Optional[float] = None) -> bool: ...


class AlgorithmBundle(Protocol):
    async def __call__(self, store: LightningStore, event: Event) -> None:
        """Initalization and execution logic."""


class RunnerBundle(Protocol):
    async def __call__(self, store: LightningStore, worker_id: int, event: Event) -> None:
        """Initalization and execution logic."""


class ExecutionStrategy:
    """When trainer has created the executable of algorithm and runner in two bundles,
    the execution strategy defines how to run them together, and how many parallel runners to run.

    The store is the centric place for the two bundles to communicate.

    The algorithm and runner's behavior (whether runner should perform one step or run forever,
    whether the algo would send out the tasks or not) are defined inside the bundle,
    and does not belong to the execution strategy.

    The execute should support Ctrl+C to exit gracefully.
    """

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        raise NotImplementedError()


class SharedMemoryExecutionStrategy(ExecutionStrategy):
    """Run algorithm and runners in a single process with threads sharing memory.

    This strategy wraps the provided `LightningStore` with a thread-safe adapter
    and runs the algorithm bundle and one or more runner bundles as threads.

    **Notes on termination:**
    Python threads cannot be forcefully killed. If a bundle does not cooperate
    (e.g., block forever in I/O without cancellation), the interpreter may not
    exit promptly. We therefore mark threads as daemons and join with a timeout,
    logging an error if any are still alive.
    """

    alias: str = "shm"

    def __init__(self, n_runners: int = 1, main_thread: Literal["algorithm", "runner"] = "runner") -> None:
        self.n_runners = n_runners
        self.main_thread = main_thread

    async def _run_until_completed_or_canceled(
        self, coro: asyncio._CoroutineLike[Any], stop_evt: threading.Event
    ) -> Any:
        task = asyncio.create_task(coro)

        # Bridge: when the threading.Event is set, cancel the task on this loop.
        async def watch_stop():
            await asyncio.to_thread(stop_evt.wait)  # doesn’t block the loop
            task.cancel()

        watcher = asyncio.create_task(watch_stop())

        result: Any = None

        try:
            await asyncio.wait({task, watcher}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            # Ensure the algorithm task is finished/cancelled and awaited
            if not task.done():
                logger.warning(f"Coroutine {coro} did not complete before stop event, cancelling...")
                task.cancel()
            else:
                logger.debug(f"Coroutine {coro} completed before stop event.")
            with suppress(asyncio.CancelledError):
                result = await task
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher

        return result

    def _algorithm_thread(self, algorithm: AlgorithmBundle, store: LightningStore) -> None:
        asyncio.run(algorithm(store))

    def _runner_thread(self, runner: RunnerBundle, store: LightningStore, worker_id: int) -> None:
        asyncio.run(runner(store, worker_id))

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info(f"Starting shm execution with {self.n_runners} runners, main thread runs '{self.main_thread}'")
        thread_safe_store = LightningStoreThreaded(store)
        if self.main_thread == "algo":
            try:
                threads = [
                    threading.Thread(target=self._runner_thread, args=(runner, thread_safe_store, i))
                    for i in range(self.n_runners)
                ]
                for t in threads:
                    t.start()

                self._algorithm_thread(algorithm, thread_safe_store)

                for t in threads:
                    t.join()
            except KeyboardInterrupt:
                logger.warning("Received KeyboardInterrupt, shutting down...")
                # FIXME: killing the threads is problematic right now

        else:
            if self.n_runners > 1:
                raise ValueError("When main_thread is 'runner', n_runners must be 1")
            try:
                algo_thread = threading.Thread(target=self._algorithm_thread, args=(algorithm, thread_safe_store))
                algo_thread.start()

                self._runner_thread(runner, thread_safe_store, 0)

                algo_thread.join()
            except KeyboardInterrupt:
                logger.warning("Received KeyboardInterrupt, shutting down...")
                # FIXME: killing the algo thread is problematic right now
                # algo_thread.kill() ?


class ClientServerExecutionStrategy(ExecutionStrategy):
    """Run algorithm (server) and runners (clients) as separate processes over HTTP.

    - Role "algorithm": start the HTTP server (`LightningStoreServer`) in-process
      and run the algorithm against it.
    - Role "runner": connect to a remote server via `LightningStoreClient` and run
      one or more runner processes.
    - Role "both": spawn runner processes, then start the algorithm + server in the
      main process.

    **Notes on termination**
    Child processes are tracked; on Ctrl+C or normal completion we terminate any
    still-alive runner processes and then `join()` them to avoid zombies.

    For termination, we try the following sequence in order:
    1. Send the special signal to the subprocesses to trigger graceful cleanup.
    2. Wait up to 5 seconds for them to exit on their own.
    3. If still alive, send SIGTERM to request termination.
    4. Wait another 5 seconds for them to exit on their own.
    5. If still alive, send SIGKILL to force immediate termination.
    """

    alias: str = "cs"

    def __init__(
        self,
        role: Literal["algorithm", "runner", "both"],
        server_host: str = "localhost",
        server_port: int = 4747,
        n_runners: int = 1,
    ) -> None:
        self.role = role
        self.n_runners = n_runners
        self.server_host = server_host
        self.server_port = server_port

    async def _execute_algorithm(self, algorithm: AlgorithmBundle, store: LightningStore) -> None:
        logger.info(f"Server will be running on {self.server_host}:{self.server_port}")
        server_store = LightningStoreServer(store, host=self.server_host, port=self.server_port)
        try:
            await server_store.start()

            await algorithm(server_store)
        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt, shutting down...")
        finally:
            await server_store.stop()

    async def _execute_runner(self, runner: RunnerBundle, store: LightningStore, worker_id: int) -> None:
        client_store = LightningStoreClient(f"http://{self.server_host}:{self.server_port}")
        try:
            await runner(client_store, worker_id)
        except KeyboardInterrupt:
            logger.warning("Received KeyboardInterrupt, shutting down...")

    def _spawn_runners(self, runner: RunnerBundle, store: LightningStore) -> list[multiprocessing.Process]:
        processes: list[multiprocessing.Process] = []

        def _runner_sync(runner: RunnerBundle, store: LightningStore, worker_id: int) -> None:
            asyncio.run(self._execute_runner(runner, store, worker_id))

        for i in range(self.n_runners):
            p = multiprocessing.Process(target=_runner_sync, args=(runner, store, i))
            processes.append(p)
            p.start()

        return processes

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info(f"Starting client-server execution with {self.n_runners} runners")

        if self.role == "algorithm":
            asyncio.run(self._execute_algorithm(algorithm, store))
        elif self.role == "runner":
            if self.n_runners == 1:
                asyncio.run(self._execute_runner(runner, store, 0))
            else:
                processes = self._spawn_runners(runner, store)
                for p in processes:
                    p.join()
        elif self.role == "both":
            # Auto start all the runners in subprocesses
            processes = self._spawn_runners(runner, store)
            # Run the algorithm in the main process
            asyncio.run(self._execute_algorithm(algorithm, store))
            # Wait for all runners to finish
            for p in processes:
                p.join()
        else:
            raise ValueError(f"Unknown role: {self.role}")


class InterProcessExecutionStrategy(ExecutionStrategy):

    alias: str = "ipc"

    # TODO: to be implemented
