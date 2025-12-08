# Copyright (c) Microsoft. All rights reserved.

import multiprocessing as mp
import threading
from multiprocessing.context import BaseContext
from typing import Optional, Protocol


class ExecutionEvent(Protocol):
    """Protocol capturing the cooperative stop contract shared by strategies.

    Implementations mirror the API of ``threading.Event`` and
    ``multiprocessing.Event`` so the rest of the execution layer can remain
    agnostic to the underlying concurrency primitive.

    Methods:

        set: Signal cancellation. The call must be idempotent.
        clear: Reset the event to the unsignaled state.
        is_set: Return ``True`` when cancellation has been requested.
        wait: Block until the event is signaled or an optional timeout elapses.
    """

    def set(self) -> None: ...
    def clear(self) -> None: ...
    def is_set(self) -> bool: ...
    def wait(self, timeout: Optional[float] = None) -> bool: ...


class ThreadingEvent:
    """Thread-safe implementation of [`ExecutionEvent`][agentlightning.ExecutionEvent]."""

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
    """Process-safe implementation of [`ExecutionEvent`][agentlightning.ExecutionEvent]."""

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
