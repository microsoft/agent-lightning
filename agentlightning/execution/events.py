# Copyright (c) Microsoft. All rights reserved.

import multiprocessing as mp
import threading
from multiprocessing.context import BaseContext
from typing import Optional, Protocol


class ExecutionEvent(Protocol):
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


class ThreadingEvent:
    """
    An Event implementation using threading.Event.

    Provides a thread-safe event object for signaling between threads.
    """

    __slots__ = ("_evt",)

    def __init__(self) -> None:
        self._evt = threading.Event()

    def set(self) -> None:
        self._evt.set()

    def clear(self) -> None:
        self._evt.clear()

    def is_set(self) -> bool:
        return self._evt.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._evt.wait(timeout)


class MultiprocessingEvent:
    """
    An Event implementation using multiprocessing.Event.

    Provides a process-safe event object for signaling between processes.
    Optionally accepts a multiprocessing context for custom process start methods.
    """

    __slots__ = ("_evt",)

    def __init__(self, *, ctx: Optional[BaseContext] = None) -> None:
        self._evt = (ctx or mp).Event()

    def set(self) -> None:
        self._evt.set()

    def clear(self) -> None:
        self._evt.clear()

    def is_set(self) -> bool:
        return self._evt.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._evt.wait(timeout)
