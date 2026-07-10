# Copyright (c) Microsoft. All rights reserved.

"""Async helpers for submitting coroutines to a dedicated event loop.

The store-backed tracing/exporter path in this codebase submits I/O work from
sync callbacks into a dedicated background loop. Both ``LightningSpanProcessor``
and ``LightningSpanExporter`` need the same loop/thread/shutdown behavior, so this
utility centralizes that logic.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Coroutine, Optional, TypeVar

logger = logging.getLogger(__name__)

STORE_WRITE_TIMEOUT_SECONDS = 10.0

T_co = TypeVar("T_co")


class AsyncStoreSpanWriter:
    """Submit coroutine callbacks from sync code on a dedicated event-loop thread."""

    def __init__(
        self,
        *,
        thread_name: str,
        clear_on_fork: bool = False,
        startup_timeout: float = 30.0,
        shutdown_timeout: float = 5.0,
    ) -> None:
        self._thread_name = thread_name
        self._clear_on_fork = clear_on_fork
        self._startup_timeout = startup_timeout
        self._shutdown_timeout = shutdown_timeout
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_ready = threading.Event()
        self._loop_init_lock = threading.Lock()
        self._loop_lock_pid: Optional[int] = None
        self._lock: Optional[threading.Lock] = None

    @property
    def loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """The worker event loop currently owned by the writer."""
        return self._loop

    @property
    def loop_thread(self) -> Optional[threading.Thread]:
        """The worker loop thread."""
        return self._loop_thread

    @property
    def lock(self) -> Optional[threading.Lock]:
        """The optional non-loop shared lock."""
        return self._lock

    def _ensure_fork_context(self) -> None:
        if not self._clear_on_fork:
            if self._loop_lock_pid is None:
                self._loop_lock_pid = os.getpid()
            return

        current_pid = os.getpid()
        if self._loop_lock_pid is None:
            self._loop_lock_pid = current_pid
            return

        if self._loop_lock_pid != current_pid:
            logger.warning("Loop and lock are not owned by the current process. Clearing them.")
            self._loop = None
            self._loop_thread = None
            self._loop_ready.clear()
            self._loop_lock_pid = current_pid
            self._lock = None

    def _run_loop(self) -> None:
        loop = self._loop
        assert loop is not None, "Loop should be initialized before thread starts"
        asyncio.set_event_loop(loop)
        self._loop_ready.set()
        loop.run_forever()

    def ensure_loop(self) -> asyncio.AbstractEventLoop:
        """Initialize and return the writer loop/thread."""
        self._ensure_fork_context()
        if self._loop_thread is not None and self._loop is not None:
            return self._loop

        with self._loop_init_lock:
            if self._loop_thread is not None and self._loop is not None:
                return self._loop

            self._loop_ready.clear()
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(target=self._run_loop, name=self._thread_name, daemon=True)
            self._loop_thread.start()
            if not self._loop_ready.wait(timeout=self._startup_timeout):
                raise RuntimeError("Timed out waiting for async loop thread to start")
            return self._loop

    def ensure_lock(self) -> threading.Lock:
        """Initialize and return a shared lock."""
        self._ensure_fork_context()
        if self._lock is None:
            self._lock = threading.Lock()
        return self._lock

    def run_in_loop(self, coro: Coroutine[object, object, T_co], *, timeout: Optional[float] = None) -> Optional[T_co]:
        """Submit a coroutine to the loop and wait for completion."""
        loop = self.ensure_loop()
        if threading.current_thread() is self._loop_thread:
            loop.call_soon_threadsafe(asyncio.create_task, coro)
            return None

        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        """Stop the loop thread and release loop resources."""
        if self._loop is None or self._loop_thread is None:
            return

        try:
            if threading.current_thread() is self._loop_thread:
                self._loop.stop()
            else:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._loop_thread.join(timeout=self._shutdown_timeout)
        except Exception:
            logger.exception("Error while shutting down async loop writer.")
        finally:
            try:
                self._loop.close()
            except Exception:
                logger.warning("Failed to close async loop.", exc_info=True)
            self._loop = None
            self._loop_thread = None
