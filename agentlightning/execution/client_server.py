# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import multiprocessing
import os
import signal
import time
from multiprocessing.context import BaseContext
from typing import Callable, Iterable, Literal, cast

from agentlightning.store.base import LightningStore
from agentlightning.store.client_server import LightningStoreClient, LightningStoreServer

from .base import AlgorithmBundle, ExecutionStrategy, RunnerBundle
from .events import Event, MultiprocessingEvent

logger = logging.getLogger(__name__)


class ClientServerExecutionStrategy(ExecutionStrategy):
    """Run algorithm (server) and runners (clients) as separate processes over HTTP.

    **Execution Roles:**

    - "algorithm": Start the HTTP server (`LightningStoreServer`) in-process and run the
      algorithm bundle against it.
    - "runner": Connect to an already running server via `LightningStoreClient` and
      execute runner bundles (optionally in multiple processes).
    - "both": Spawn the runner processes first, then launch the algorithm/server
      bundle on the main process. This mode orchestrates the full loop locally.

    When role == "both", you may choose which side runs on the main process via
    `main_process` (debug helper). Running the runner bundle on the main process
    is only supported with `n_runners == 1`.

    Important: When `main_process == "runner"`, the algorithm runs in a subprocess
    with the LightningStore server. This means any state modifications made during
    execution remain in that subprocess and are NOT reflected in the original store
    object passed to `execute()`. The main process runner accesses the store only
    through the HTTP client interface.

    **Abort / Stop Model (four-step escalation):**

    1. Cooperative stop:
       A shared :class:`~agentlightning.execution.events.MultiprocessingEvent`
       (`stop_evt`) is passed to *all* bundles. Bundles should check it to exit.
       Any crash (algorithm or runner) sets `stop_evt` so the other side can
       stop cooperatively. Ctrl+C on the main process also flips the event.
    2. KeyboardInterrupt synth:
       Remaining subprocesses receive `SIGINT` to trigger `KeyboardInterrupt`
       handlers.
    3. Termination:
       Stubborn subprocesses get `terminate()` (SIGTERM on POSIX).
    4. Kill:
        As a last resort we call `kill()` (SIGKILL on POSIX).

    Notes:
        This mirrors the semantics implemented in :mod:`shared_memory`, but adapted
        to multiple processes and the HTTP client/server boundary.
    """

    alias: str = "cs"

    def __init__(
        self,
        role: Literal["algorithm", "runner", "both"] | None = None,
        server_host: str | None = None,
        server_port: int | None = None,
        n_runners: int = 1,
        graceful_timeout: float = 5.0,
        terminate_timeout: float = 5.0,
        main_process: Literal["algorithm", "runner"] = "algorithm",
    ) -> None:
        """Configure the strategy.

        Args:
            role: Which side(s) to run in this process. When omitted, the
                :envvar:`AGL_CURRENT_ROLE` environment variable is used.
            server_host: Interface the HTTP server binds to when running the
                algorithm bundle locally. Defaults to :envvar:`AGL_SERVER_HOST`
                or ``"localhost"`` if unset.
            server_port: Port for the HTTP server in "algorithm"/"both" modes.
                Defaults to :envvar:`AGL_SERVER_PORT` or ``4747`` if unset.
            n_runners: Number of runner processes to spawn in "runner"/"both".
            graceful_timeout: How long to wait (seconds) after setting the stop
                event before escalating to signals.
            terminate_timeout: How long to wait between escalation steps beyond
                the cooperative phase (re-used for SIGINT, terminate, and kill).
            main_process: Which bundle runs on the main process when
                `role == "both"`. `"runner"` requires `n_runners == 1` and is
                primarily intended for debugging.
        """
        if role is None:
            role_env = os.getenv("AGL_CURRENT_ROLE")
            if role_env is None:
                raise ValueError("role must be provided via argument or AGL_CURRENT_ROLE env var")
            if role_env not in ("algorithm", "runner", "both"):
                raise ValueError("role must be one of 'algorithm', 'runner', or 'both'")
            role = role_env

        if server_host is None:
            server_host = os.getenv("AGL_SERVER_HOST", "localhost")

        if server_port is None:
            server_port_env = os.getenv("AGL_SERVER_PORT")
            if server_port_env is None:
                server_port = 4747
            else:
                try:
                    server_port = int(server_port_env)
                except ValueError as exc:
                    raise ValueError("AGL_SERVER_PORT must be an integer") from exc

        self.role = role
        self.n_runners = n_runners
        self.server_host = server_host
        self.server_port = server_port
        self.graceful_timeout = graceful_timeout
        self.terminate_timeout = terminate_timeout
        if main_process not in ("algorithm", "runner"):
            raise ValueError("main_process must be 'algorithm' or 'runner'")
        if main_process == "runner":
            if role != "both":
                raise ValueError("main_process='runner' is only supported when role='both'")
            if n_runners != 1:
                raise ValueError("main_process='runner' requires n_runners to be 1")
        self.main_process = main_process

    async def _execute_algorithm(self, algorithm: AlgorithmBundle, store: LightningStore, stop_evt: Event) -> None:
        logger.info("Starting LightningStore server on %s:%s", self.server_host, self.server_port)
        server_store = LightningStoreServer(store, host=self.server_host, port=self.server_port)
        server_started = False

        try:
            await server_store.start()
            server_started = True
            logger.debug("Algorithm bundle starting against endpoint %s", server_store.endpoint)
            await algorithm(server_store, stop_evt)
            logger.debug("Algorithm bundle completed successfully")
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
                else:
                    logger.debug("LightningStore server shutdown completed")

    async def _execute_runner(self, runner: RunnerBundle, worker_id: int, stop_evt: Event) -> None:
        client_store = LightningStoreClient(f"http://{self.server_host}:{self.server_port}")
        try:
            logger.debug("Runner %s connecting to server at %s:%s", worker_id, self.server_host, self.server_port)
            await runner(client_store, worker_id, stop_evt)
            logger.debug("Runner %s completed successfully", worker_id)
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
                logger.exception("Error closing LightningStore client for runner %s", worker_id)
            else:
                logger.debug("Runner %s closed LightningStore client", worker_id)

    def _spawn_runners(
        self,
        runner: RunnerBundle,
        stop_evt: Event,
        *,
        ctx: BaseContext,
    ) -> list[multiprocessing.Process]:
        """Used when `role == "runner"` or `role == "both"` and `n_runners > 1`."""
        processes: list[multiprocessing.Process] = []

        def _runner_sync(runner: RunnerBundle, worker_id: int, stop_evt: Event) -> None:
            # Runners are executed in child processes; each process owns its own
            # event loop to keep the asyncio scheduler isolated.
            asyncio.run(self._execute_runner(runner, worker_id, stop_evt))

        for i in range(self.n_runners):
            process = cast(
                multiprocessing.Process,
                ctx.Process(target=_runner_sync, args=(runner, i, stop_evt), name=f"runner-{i}"),  # type: ignore
            )
            process.start()
            logger.debug("Spawned runner process %s (pid=%s)", process.name, process.pid)
            processes.append(process)

        return processes

    def _spawn_algorithm_process(
        self,
        algorithm: AlgorithmBundle,
        store: LightningStore,
        stop_evt: Event,
        *,
        ctx: BaseContext,
    ) -> multiprocessing.Process:
        """Used when `main_process == "runner"`."""

        def _algorithm_sync(algorithm: AlgorithmBundle, store: LightningStore, stop_evt: Event) -> None:
            asyncio.run(self._execute_algorithm(algorithm, store, stop_evt))

        process = cast(
            multiprocessing.Process,
            ctx.Process(target=_algorithm_sync, args=(algorithm, store, stop_evt), name="algorithm"),  # type: ignore
        )
        process.start()
        logger.debug("Spawned algorithm process %s (pid=%s)", process.name, process.pid)
        return process

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
            try:
                action(process)
            except Exception:
                logger.exception("Error signaling process %s (pid=%s)", process.name, process.pid)

    def _shutdown_processes(
        self,
        processes: list[multiprocessing.Process],
        stop_evt: Event,
    ) -> None:
        """4-step escalation shutdown of ``processes``."""
        if not processes:
            logger.debug("No subprocesses to shutdown")
            return

        if not stop_evt.is_set():
            logger.debug("Sending cooperative stop signal to subprocesses")
            stop_evt.set()
        else:
            logger.debug("Stop event already set; waiting for subprocesses to exit")

        alive = self._join_until_deadline(processes, self.graceful_timeout)
        if not alive:
            return

        logger.warning(
            "Subprocesses still alive after cooperative wait; sending SIGINT to %s",
            ", ".join(p.name or str(p.pid) for p in alive),
        )
        # SIGINT is not reliable on Windows, but we do not consider such case yet.
        self._signal_processes(alive, lambda p: os.kill(cast(int, p.pid), signal.SIGINT))
        alive = self._join_until_deadline(alive, self.terminate_timeout)
        if not alive:
            return

        logger.warning(
            "Subprocesses still alive after SIGINT wait; sending terminate() to %s",
            ", ".join(p.name or str(p.pid) for p in alive),
        )
        self._signal_processes(alive, lambda p: p.terminate())

        alive = self._join_until_deadline(alive, self.terminate_timeout)
        if not alive:
            return

        logger.error(
            "Subprocesses still alive after terminate(); sending kill() to %s",
            ", ".join(p.name or str(p.pid) for p in alive),
        )
        self._signal_processes(alive, lambda p: p.kill())
        alive = self._join_until_deadline(alive, self.terminate_timeout)

        if alive:
            logger.error(
                "Subprocesses failed to exit even after kill(): %s", ", ".join(p.name or str(p.pid) for p in alive)
            )

    def _check_process_exitcodes(self, processes: Iterable[multiprocessing.Process]) -> None:
        """Raise an error if any managed process exited with a non-zero status."""
        failed = [p for p in processes if p.exitcode not in (0, None)]
        if failed:
            formatted = ", ".join(f"{p.name or p.pid} (exitcode={p.exitcode})" for p in failed)
            raise RuntimeError(f"Subprocesses failed: {formatted}")

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info(
            "Starting client-server execution with %d runner(s) [role=%s, main_process=%s]",
            self.n_runners,
            self.role,
            self.main_process,
        )

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
                logger.info("Running algorithm solely...")
                asyncio.run(self._execute_algorithm(algorithm, store, stop_evt))
            elif self.role == "runner":
                if self.n_runners == 1:
                    logger.info("Running runner solely...")
                    asyncio.run(self._execute_runner(runner, 0, stop_evt))
                else:
                    logger.info("Spawning runner processes...")
                    processes = self._spawn_runners(runner, stop_evt, ctx=ctx)
                    # Wait for the processes to finish naturally.
                    for process in processes:
                        process.join()
                    self._check_process_exitcodes(processes)
            elif self.role == "both":
                if self.main_process == "algorithm":
                    logger.info("Spawning runner processes...")
                    processes = self._spawn_runners(runner, stop_evt, ctx=ctx)
                    try:
                        logger.info("Running algorithm...")
                        asyncio.run(self._execute_algorithm(algorithm, store, stop_evt))
                    finally:
                        # Always request the runner side to unwind once the
                        # algorithm/server portion finishes (successfully or not).
                        stop_evt.set()
                else:  # main_process == "runner"
                    if self.n_runners > 1:
                        raise ValueError("main_process='runner' requires n_runners to be 1")

                    logger.info("Spawning algorithm process...")
                    algorithm_process = self._spawn_algorithm_process(algorithm, store, stop_evt, ctx=ctx)
                    processes = [algorithm_process]

                    # Run the lone runner cooperatively in-process so users can
                    # attach a debugger. The algorithm + HTTP server live in
                    # the background process spawned above (the provided
                    # store must therefore be picklable when using spawn).
                    logger.info("Running runner...")
                    asyncio.run(self._execute_runner(runner, 0, stop_evt))

                    # Wait for the algorithm process to finish.
                    algorithm_process.join()
            else:
                raise ValueError(f"Unknown role: {self.role}")
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received; initiating shutdown")
            stop_evt.set()
            keyboard_interrupt = True
        except BaseException as exc:
            logger.exception("Unhandled exception in execute method")
            stop_evt.set()
            # Preserve the original exception so we can avoid masking it during
            # the cleanup phase.
            exception = exc
            raise
        finally:
            logger.info("Shutting down subprocesses")
            self._shutdown_processes(processes, stop_evt)
            if processes:
                try:
                    self._check_process_exitcodes(processes)
                except RuntimeError as err:
                    if exception is not None or keyboard_interrupt:
                        # We already propagate/handled a different failure, so
                        # emit a warning instead of raising a secondary error.
                        logger.warning("Subprocesses ended abnormally during shutdown: %s", err)
                    else:
                        raise
