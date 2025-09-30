# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import threading
from contextlib import suppress
from typing import Any, Awaitable, Callable, List, Literal, Tuple

from agentlightning.store.base import LightningStore
from agentlightning.store.threading import LightningStoreThreaded

from .base import AlgorithmBundle, ExecutionStrategy, RunnerBundle
from .events import Event, ThreadingEvent

logger = logging.getLogger(__name__)


class SharedMemoryExecutionStrategy(ExecutionStrategy):
    """Run algorithm and runners in a single process with threads sharing memory.

    Termination & abort model:

    - One shared ThreadingEvent (`stop_evt`) is passed to *all* bundles.
    - The main thread (only) receives KeyboardInterrupt on Ctrl+C; we set `stop_evt` there.
    - If any bundle raises, we set `stop_evt` from that thread to stop the rest.
    - After the main-thread bundle finishes normally:
      - If main_thread is "algorithm", we also set `stop_evt` to stop the runners.
      - If main_thread is "runner", we do not set `stop_evt` to stop the algorithm.
        We instead wait for the algorithm to finish naturally.
    - Background threads are daemons; we join briefly and log any stragglers.

    Notes: Signals other than SIGINT (e.g., SIGTERM) are not intercepted; we respect
    Python's default behavior for them.
    """

    alias: str = "shm"

    def __init__(
        self,
        n_runners: int = 1,
        main_thread: Literal["algorithm", "runner"] = "runner",
        join_timeout: float = 10.0,
        graceful_delay: float = 5.0,
    ) -> None:
        if main_thread not in ("algorithm", "runner"):
            raise ValueError("main_thread must be 'algorithm' or 'runner'")
        if main_thread == "runner" and n_runners != 1:
            raise ValueError("When main_thread is 'runner', n_runners must be 1")
        self.n_runners = n_runners
        self.main_thread = main_thread
        self.join_timeout = join_timeout
        self.graceful_delay = graceful_delay

    async def _run_until_completed_or_canceled(self, coro: Awaitable[Any], stop_evt: Event) -> Any:
        """Run `coro` until it finishes or a cooperative stop is requested.

        Control flow:
          1) Start the bundle coroutine as `task`.
          2) Start a watcher task that waits for `stop_evt` *without blocking* the loop
             (using `asyncio.to_thread(stop_evt.wait)`).
          3) When the stop event flips:
               a) Give the bundle *graceful_delay* seconds to finish on its own,
                  because well-behaved bundles will check the event and return.
               b) If still running after the grace period, cancel the bundle task.
          4) Ensure both tasks are awaited; swallow `CancelledError` where appropriate.

        This is a *backup* mechanism for bundles that might not poll the event
        frequently; cooperative shutdown (checking `stop_evt` yourself) is still preferred.
        """
        task: asyncio.Task[Any] = asyncio.create_task(coro)  # type: ignore

        async def watcher() -> None:
            # Block in a thread so we don't block the event loop.
            await asyncio.to_thread(stop_evt.wait)

            # Grace period: let a cooperative bundle exit on its own.
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=self.graceful_delay)  # type: ignore
                logger.debug("Bundle finished by itself during grace period.")
                return  # bundle finished by itself during grace period
            except asyncio.TimeoutError:
                # Still running after the grace window.
                pass
            except asyncio.CancelledError:
                # If someone else canceled the task already, we're done.
                logger.debug("Bundle already canceled by someone else; exiting watcher.")
                return

            # Still running after the grace window: cancel it.
            if not task.done():
                logger.debug("Graceful delay elapsed; canceling bundle task...")
                task.cancel()

        watcher_task = asyncio.create_task(watcher())
        result: Any = None

        try:
            # We don't wait on FIRST_COMPLETED here, because we want the watcher
            # to be able to grant a grace window after stop_evt flips.
            await asyncio.wait(
                {task, watcher_task}, return_when=asyncio.FIRST_COMPLETED
            )  # pyright: ignore[reportUnknownArgumentType]
        finally:
            # If the main task hasn't completed yet (e.g., watcher scheduled cancel),
            # finish the cancellation handshake.
            if not task.done():
                with suppress(asyncio.CancelledError):
                    await task
            else:
                # Task completed naturally; retrieve result.
                with suppress(asyncio.CancelledError):
                    result = await task  # type: ignore

            watcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await watcher_task

        return result  # type: ignore

    def _run_algorithm(self, algorithm: AlgorithmBundle, store: LightningStore, stop_evt: Event) -> None:
        try:
            asyncio.run(self._run_until_completed_or_canceled(algorithm(store, stop_evt), stop_evt))
        except asyncio.CancelledError:
            logger.info("Algorithm bundle canceled due to stop signal.")
        except BaseException:
            logger.exception("Algorithm bundle crashed; signaling stop to others.")
            stop_evt.set()
            raise

    def _run_runner(self, runner: RunnerBundle, store: LightningStore, worker_id: int, stop_evt: Event) -> None:
        try:
            asyncio.run(self._run_until_completed_or_canceled(runner(store, worker_id, stop_evt), stop_evt))
        except asyncio.CancelledError:
            logger.info("Runner bundle (worker_id=%s) canceled due to stop signal.", worker_id)
        except BaseException:
            logger.exception("Runner bundle crashed (worker_id=%s); signaling stop to others.", worker_id)
            stop_evt.set()
            raise

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info(
            "Starting shm execution with %d runner(s); main thread runs '%s'",
            self.n_runners,
            self.main_thread,
        )

        stop_evt = ThreadingEvent()
        thread_safe_store = LightningStoreThreaded(store)

        def make_thread(name: str, target: Callable[..., Any], args: Tuple[Any, ...]) -> threading.Thread:
            t = threading.Thread(name=name, target=target, args=args, daemon=True)
            t.start()
            return t

        threads: List[threading.Thread] = []

        try:
            if self.main_thread == "algorithm":
                # Start runner threads; algorithm runs on main thread.
                for i in range(self.n_runners):
                    thread = make_thread(
                        name=f"runner-{i}",
                        target=self._run_runner,
                        args=(runner, thread_safe_store, i, stop_evt),
                    )
                    threads.append(thread)

                # Ctrl+C here raises KeyboardInterrupt on this stack.
                self._run_algorithm(algorithm, thread_safe_store, stop_evt)

                # If algo finishes naturally, request runners to stop.
                stop_evt.set()

            else:  # main_thread == "runner"
                # Start algorithm in background; runner runs on main thread.
                thread = make_thread(
                    name="algorithm",
                    target=self._run_algorithm,
                    args=(algorithm, thread_safe_store, stop_evt),
                )
                threads.append(thread)

                # Ctrl+C here raises KeyboardInterrupt on this stack.
                self._run_runner(runner, thread_safe_store, 0, stop_evt)

                # If runner finishes naturally, WAIT FOR ALGORITHM TO FINISH.
                thread.join()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received on main thread; initiating cooperative shutdown...")
            stop_evt.set()
        finally:
            # Attempt a clean join; if some threads don't comply, log and move on.
            for t in threads:
                logger.debug("Joining thread %s...", t.name)
                t.join(timeout=self.join_timeout)

            alive = [t.name for t in threads if t.is_alive()]
            if alive:
                logger.error(
                    "Threads still alive after %.1fs: %s. They are daemons; continuing shutdown.",
                    self.join_timeout,
                    ", ".join(alive),
                )
