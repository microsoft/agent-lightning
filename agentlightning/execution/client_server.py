# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import threading
from contextlib import suppress
from typing import Any, Awaitable, Literal, Optional, Protocol

from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer

from .base import AlgorithmBundle, ExecutionStrategy, RunnerBundle

logger = logging.getLogger(__name__)


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
