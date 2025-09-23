# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import threading
from typing import Literal, Protocol

from .store.base import LightningStore
from .store.client_server import LightningStoreClient, LightningStoreServer
from .store.threading import LightningStoreThreaded

logger = logging.getLogger(__name__)


class AlgorithmBundle(Protocol):
    async def __call__(self, store: LightningStore) -> None: ...


class RunnerBundle(Protocol):
    async def __call__(self, store: LightningStore, worker_id: int) -> None: ...


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

    alias: str = "shm"

    def __init__(self, n_runners: int = 1, main_thread: Literal["algorithm", "runner"] = "runner") -> None:
        self.n_runners = n_runners
        self.main_thread = main_thread

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


class InterProcessExecutionStrategy(ExecutionStrategy):

    alias: str = "ipc"

    # TODO: to be implemented


class ClientServerExecutionStrategy(ExecutionStrategy):

    alias: str = "cs"

    def __init__(
        self,
        role: Literal["algorithm", "runner"],
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

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info(f"Starting client-server execution with {self.n_runners} runners")

        if self.role == "algorithm":
            asyncio.run(self._execute_algorithm(algorithm, store))
        else:
            if self.n_runners == 1:
                asyncio.run(self._execute_runner(runner, store, 0))
            else:
                processes: list[multiprocessing.Process] = []

                def _runner_sync(runner: RunnerBundle, store: LightningStore, worker_id: int) -> None:
                    asyncio.run(self._execute_runner(runner, store, worker_id))

                for i in range(self.n_runners):
                    p = multiprocessing.Process(target=_runner_sync, args=(runner, store, i))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()
