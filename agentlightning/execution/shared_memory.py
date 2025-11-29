# Copyright (c) Microsoft. All rights reserved.

import asyncio
import logging
import threading
from contextlib import suppress
from queue import SimpleQueue
from typing import Any, Awaitable, Callable, List, Literal, Optional, Tuple

from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var
from agentlightning.store.base import LightningStore
from agentlightning.store.threading import LightningStoreThreaded

from .base import AlgorithmBundle, ExecutionStrategy, RunnerBundle
from .events import ExecutionEvent, ThreadingEvent

logger = logging.getLogger(__name__)


class SharedMemoryExecutionStrategy(ExecutionStrategy):
    """Execute bundles in a single process with cooperative worker threads.

    Stop Model:

    - All bundles share one [`ThreadingEvent`][agentlightning.ThreadingEvent]
      named `stop_evt`.
    - Only the main thread receives `KeyboardInterrupt`. When Ctrl+C occurs we
      set `stop_evt`.
    - Any exception raised inside a bundle sets `stop_evt` so other threads can
      unwind cooperatively.
    - Once the bundle running on the main thread exits successfully the
      treatment depends on `main_thread`:
        - `"algorithm"`: the runners are asked to stop by setting `stop_evt`.
        - `"runner"`: the algorithm keeps running until it exits naturally.
    - Background threads are marked as daemons. We join them briefly and log any
      stragglers before shutting down.

    !!! note
        Signals other than `SIGINT` (such as `SIGTERM`) are not intercepted;
        Python's default behavior for those signals is preserved.
    """

    alias: str = "shm"

    def __init__(
        self,
        n_runners: int = 1,
        main_thread: Literal["algorithm", "runner"] = "runner",
        join_timeout: float = 15.0,
        graceful_delay: float = 5.0,
        poll_interval: float = 0.05,
        managed_store: bool | None = None,
    ) -> None:
        if main_thread not in ("algorithm", "runner"):
            raise ValueError("main_thread must be 'algorithm' or 'runner'")
        if main_thread == "runner" and n_runners != 1:
            raise ValueError(
                "When main_thread is 'runner', n_runners must be 1. "
                "Either use 'algorithm' on the main thread or set n_runners to 1."
            )
        self.n_runners = n_runners
        self.main_thread = main_thread
        self.join_timeout = join_timeout
        self.graceful_delay = graceful_delay
        self.poll_interval = poll_interval
        self.managed_store = resolve_bool_env_var(
            LightningEnvVar.AGL_MANAGED_STORE, override=managed_store, fallback=True
        )

    async def _run_until_completed_or_canceled(self, coro: Awaitable[Any], stop_evt: ExecutionEvent) -> Any:
        """Run `coro` until it finishes or a cooperative stop is requested.

        Control flow:

        1. Start the bundle coroutine as `task`.
        2. Launch a watcher that polls `stop_evt` without blocking the loop.
        3. When the stop event flips:
            a. Give the bundle `graceful_delay` seconds to finish on its own,
               because well-behaved bundles will check the event and return.
            b. Cancel the bundle task if it is still running after the grace
               period.
        4. Await both tasks and swallow `CancelledError` where appropriate.

        This is a *backup* mechanism for bundles that might not poll the event
        frequently; cooperative shutdown (checking `stop_evt` inside the
        bundle) remains the preferred approach.
        """
        task: asyncio.Task[Any] = asyncio.create_task(coro)  # type: ignore
        task_exception: Optional[BaseException] = None

        async def watcher() -> None:
            # Poll the threading event without blocking the event loop. Using a
            # background thread via ``asyncio.to_thread`` makes cancellation
            # difficult because ``ThreadingEvent.wait`` is not interruptible.
            # Instead we cooperatively check the flag from the loop so the
            # watcher task stays cancellable and tests don't hang when the
            # bundle finishes naturally before the stop event is set.
            while not stop_evt.is_set():
                await asyncio.sleep(self.poll_interval)

            # Grace period: let a cooperative bundle exit on its own.
            try:
                # At this point of waiting, the main task should already see the stop event.
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
                try:
                    await asyncio.wait_for(task, timeout=self.graceful_delay)  # second chance
                except asyncio.TimeoutError:
                    logger.error(
                        "Bundle task did not stop after cancellation; abandoning task."
                        "This thread could live until the process exits."
                    )
                    # We return without awaiting it. asyncio.run will still try to cancel
                    # pending tasks on loop close; if the task ignores cancellation, this
                    # thread may still stick. It's the best we can do in Python.
                    # We don't raise an exception here, but the thread could be a zombie.
                    return result
            else:
                # Task completed naturally; retrieve result.
                try:
                    result = await task  # type: ignore
                except asyncio.CancelledError:
                    pass
                except BaseException as exc:
                    task_exception = exc

            watcher_task.cancel()
            with suppress(asyncio.CancelledError):
                await watcher_task

        if task_exception is not None:
            raise task_exception

        return result  # type: ignore

    def _run_algorithm(
        self,
        algorithm: AlgorithmBundle,
        store: LightningStore,
        stop_evt: ExecutionEvent,
        thread_exceptions: Optional[SimpleQueue[BaseException]],
    ) -> None:
        try:
            asyncio.run(self._run_until_completed_or_canceled(algorithm(store, stop_evt), stop_evt))
        except asyncio.CancelledError:
            logger.info("Algorithm bundle canceled due to stop signal.")
        except BaseException as exc:
            logger.exception("Algorithm bundle crashed; signaling stop to others.")
            if thread_exceptions is not None:
                thread_exceptions.put(exc)
            stop_evt.set()
            raise

    def _run_runner(
        self,
        runner: RunnerBundle,
        store: LightningStore,
        worker_id: int,
        stop_evt: ExecutionEvent,
        thread_exceptions: Optional[SimpleQueue[BaseException]],
    ) -> None:
        try:
            asyncio.run(self._run_until_completed_or_canceled(runner(store, worker_id, stop_evt), stop_evt))
        except asyncio.CancelledError:
            logger.info("Runner bundle (worker_id=%s) canceled due to stop signal.", worker_id)
        except BaseException as exc:
            logger.exception("Runner bundle crashed (worker_id=%s); signaling stop to others.", worker_id)
            if thread_exceptions is not None:
                thread_exceptions.put(exc)
            stop_evt.set()
            raise

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        logger.info(
            "Starting shm execution with %d runner(s); main thread runs '%s'",
            self.n_runners,
            self.main_thread,
        )

        # Create stop event and thread-safe store.
        stop_evt = ThreadingEvent()
        if self.managed_store:
            thread_safe_store = LightningStoreThreaded(store)
        else:
            thread_safe_store = store

        thread_exceptions: SimpleQueue[BaseException] = SimpleQueue()
        raised_from_thread: Optional[BaseException] = None

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
                        args=(runner, thread_safe_store, i, stop_evt, thread_exceptions),
                    )
                    threads.append(thread)

                # Ctrl+C here raises KeyboardInterrupt on this stack.
                # Main thread doesn't need to collect exceptions.
                self._run_algorithm(algorithm, thread_safe_store, stop_evt, None)

                # If algo finishes naturally, request runners to stop.
                stop_evt.set()

            else:  # main_thread == "runner"
                # Start algorithm in background; runner runs on main thread.
                thread = make_thread(
                    name="algorithm",
                    target=self._run_algorithm,
                    args=(algorithm, thread_safe_store, stop_evt, thread_exceptions),
                )
                threads.append(thread)

                # Ctrl+C here raises KeyboardInterrupt on this stack.
                # Main thread doesn't need to collect exceptions.
                self._run_runner(runner, thread_safe_store, 0, stop_evt, None)

                # If runner finishes naturally, WAIT FOR ALGORITHM TO FINISH.
                thread.join()

            if not thread_exceptions.empty():
                raised_from_thread = thread_exceptions.get()

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

            if raised_from_thread is None and not thread_exceptions.empty():
                raised_from_thread = thread_exceptions.get()

        if raised_from_thread is not None:
            raise raised_from_thread
