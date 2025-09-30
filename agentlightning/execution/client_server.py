# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import signal
import time
from contextlib import suppress
from multiprocessing.context import BaseContext
from typing import Callable, Iterable, Literal, cast

from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer

from .base import AlgorithmBundle, ExecutionStrategy, RunnerBundle
from .events import Event, MultiprocessingEvent

logger = logging.getLogger(__name__)


class ClientServerExecutionStrategy(ExecutionStrategy):
    """Run algorithm (server) and runners (clients) as separate processes over HTTP.

    Execution roles:

    - "algorithm": start the HTTP server (`LightningStoreServer`) in-process and
      run the algorithm bundle against it.
    - "runner": connect to an already running server via `LightningStoreClient`
      and execute runner bundles (optionally in multiple processes).
    - "both": spawn the runner processes first, then launch the algorithm/server
      bundle on the main process. This mode orchestrates the full loop locally.

    Abort / stop model:

    - A shared :class:`~agentlightning.execution.events.MultiprocessingEvent`
      (``stop_evt``) is passed to *all* bundles. Bundles should check it to exit.
    - Any crash (algorithm or runner) sets ``stop_evt`` so the other side can
      stop cooperatively.
    - Ctrl+C is caught on the main process; we flip ``stop_evt`` and continue the
      staged shutdown below.
    - Runner processes are given a grace period to exit on their own after they receive
      the stop event. If they ignore it we escalate to ``terminate()`` and then
      ``kill()``.

    This mirrors the semantics implemented in :mod:`shared_memory`, but adapted to
    multiple processes and the HTTP client/server boundary.
    """

    alias: str = "cs"

    def __init__(
        self,
        role: Literal["algorithm", "runner", "both"],
        server_host: str = "localhost",
        server_port: int = 4747,
        n_runners: int = 1,
        graceful_timeout: float = 5.0,
        terminate_timeout: float = 5.0,
    ) -> None:
        """Configure the strategy.

        Args:
            role: Which side(s) to run in this process.
            server_host: Interface the HTTP server binds to when running the
                algorithm bundle locally.
            server_port: Port for the HTTP server in "algorithm"/"both" modes.
            n_runners: Number of runner processes to spawn in "runner"/"both".
            graceful_timeout: How long to wait (seconds) after setting the stop
                event before escalating to ``terminate()``.
            terminate_timeout: How long to wait after ``terminate()`` before
                escalating to ``kill()`` (seconds).
        """
        self.role = role
        self.n_runners = n_runners
        self.server_host = server_host
        self.server_port = server_port
        self.graceful_timeout = graceful_timeout
        self.terminate_timeout = terminate_timeout

    async def _execute_algorithm(self, algorithm: AlgorithmBundle, store: LightningStore, stop_evt: Event) -> None:
        logger.info("Starting LightningStore server on %s:%s", self.server_host, self.server_port)
        server_store = LightningStoreServer(store, host=self.server_host, port=self.server_port)
        server_started = False

        try:
            await server_store.start()
            server_started = True
            await algorithm(server_store, stop_evt)
        except KeyboardInterrupt:
            logger.warning("Algorithm received KeyboardInterrupt; signaling stop event")
            stop_evt.set()
            raise
        except BaseException:
            logger.exception("Algorithm bundle crashed; signaling stop event")
            stop_evt.set()
            raise
        finally:
            if server_started:
                try:
                    await server_store.stop()
                except Exception:
                    logger.exception("Error stopping LightningStore server")

    async def _execute_runner(self, runner: RunnerBundle, worker_id: int, stop_evt: Event) -> None:
        client_store = LightningStoreClient(f"http://{self.server_host}:{self.server_port}")
        try:
            await runner(client_store, worker_id, stop_evt)
        except KeyboardInterrupt:
            logger.warning("Runner %s received KeyboardInterrupt; signaling stop event", worker_id)
            stop_evt.set()
            raise
        except BaseException:
            logger.exception("Runner %s crashed; signaling stop event", worker_id)
            stop_evt.set()
            raise
        finally:
            try:
                await client_store.close()
            except Exception:
                logger.exception("Error closing LightningStore client")

    def _spawn_runners(
        self,
        runner: RunnerBundle,
        stop_evt: Event,
        *,
        ctx: BaseContext,
    ) -> list[multiprocessing.Process]:
        processes: list[multiprocessing.Process] = []

        def _runner_sync(runner: RunnerBundle, worker_id: int, stop_evt: Event) -> None:
            # Ignore Ctrl+C in worker processes; the main process handles it
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # Runners are executed in child processes; each process owns its own
            # event loop to keep the asyncio scheduler isolated.
            asyncio.run(self._execute_runner(runner, worker_id, stop_evt))

        for i in range(self.n_runners):
            process = cast(
                multiprocessing.Process,
                ctx.Process(target=_runner_sync, args=(runner, i, stop_evt), name=f"runner-{i}"),  # type: ignore
            )
            process.start()
            processes.append(process)

        return processes

    def _join_until_deadline(
        self,
        processes: Iterable[multiprocessing.Process],
        timeout: float,
    ) -> list[multiprocessing.Process]:
        """Join ``processes`` until ``timeout`` elapses, returning those still alive."""
        deadline = time.monotonic() + timeout
        still_alive: list[multiprocessing.Process] = []
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining > 0:
                process.join(remaining)
            else:
                process.join(0)
            if process.is_alive():
                still_alive.append(process)
        return still_alive

    def _signal_processes(
        self,
        processes: Iterable[multiprocessing.Process],
        action: Callable[[multiprocessing.Process], None],
    ) -> None:
        """Invoke ``action`` on each process while suppressing individual failures."""
        for process in processes:
            with suppress(Exception):
                action(process)

    def _shutdown_processes(
        self,
        processes: list[multiprocessing.Process],
        stop_evt: Event,
    ) -> None:
        if not processes:
            logger.info("No runner processes to shutdown")
            return

        if not stop_evt.is_set():
            logger.info("Sending cooperative stop signal to runner processes")
            stop_evt.set()
        else:
            logger.info("Stop event already set; waiting for runner processes to exit")

        alive = self._join_until_deadline(processes, self.graceful_timeout)
        if not alive:
            return

        logger.warning(
            "Runner processes still alive after %.1fs; sending terminate() to %s",
            self.graceful_timeout,
            ", ".join(p.name or str(p.pid) for p in alive),
        )
        self._signal_processes(alive, lambda p: p.terminate())

        alive = self._join_until_deadline(alive, self.terminate_timeout)
        if not alive:
            return

        logger.error(
            "Runner processes still alive after terminate(); sending kill() to %s",
            ", ".join(p.name or str(p.pid) for p in alive),
        )
        self._signal_processes(alive, lambda p: p.kill())
        alive = self._join_until_deadline(alive, self.terminate_timeout)

        if alive:
            logger.error(
                "Runner processes failed to exit even after kill(): %s", ", ".join(p.name or str(p.pid) for p in alive)
            )

    def _check_runner_exitcodes(self, processes: Iterable[multiprocessing.Process]) -> None:
        """Raise an error if any runner exited with a non-zero status."""
        failed = [p for p in processes if p.exitcode not in (0, None)]
        if failed:
            formatted = ", ".join(f"{p.name or p.pid} (exitcode={p.exitcode})" for p in failed)
            raise RuntimeError(f"Runner processes failed: {formatted}")

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info("Starting client-server execution with %d runner(s)", self.n_runners)

        # Re-use the active multiprocessing context so the event and processes
        # agree on the start method (fork/spawn/forkserver).
        ctx = multiprocessing.get_context()
        stop_evt = MultiprocessingEvent(ctx=ctx)
        # Track spawned processes so we can enforce termination ordering and
        # surface non-zero exit codes back to the caller.
        processes: list[multiprocessing.Process] = []

        exception: BaseException | None = None
        keyboard_interrupt = False

        try:
            if self.role == "algorithm":
                asyncio.run(self._execute_algorithm(algorithm, store, stop_evt))
            elif self.role == "runner":
                if self.n_runners == 1:
                    asyncio.run(self._execute_runner(runner, 0, stop_evt))
                else:
                    processes = self._spawn_runners(runner, stop_evt, ctx=ctx)
                    # Wait for the processes to finish naturally.
                    for process in processes:
                        process.join()
                    self._check_runner_exitcodes(processes)
            elif self.role == "both":
                processes = self._spawn_runners(runner, stop_evt, ctx=ctx)
                try:
                    asyncio.run(self._execute_algorithm(algorithm, store, stop_evt))
                finally:
                    # Always request the runner side to unwind once the
                    # algorithm/server portion finishes (successfully or not).
                    stop_evt.set()
            else:
                raise ValueError(f"Unknown role: {self.role}")
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received; initiating shutdown")
            stop_evt.set()
            keyboard_interrupt = True
        except BaseException as exc:
            stop_evt.set()
            # Preserve the original exception so we can avoid masking it during
            # the cleanup phase.
            exception = exc
            raise
        finally:
            self._shutdown_processes(processes, stop_evt)
            if processes:
                try:
                    self._check_runner_exitcodes(processes)
                except RuntimeError as err:
                    if exception is not None or keyboard_interrupt:
                        # We already propagate/handled a different failure, so
                        # emit a warning instead of raising a secondary error.
                        logger.warning("Runner processes ended abnormally during shutdown: %s", err)
                    else:
                        raise
