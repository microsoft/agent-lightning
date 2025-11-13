# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
CrossSync provides a toolset for sharing logic between async and sync codebases, including:
- A set of decorators for annotating async classes and functions
    (@CrossSync.export_sync, @CrossSync.convert, @CrossSync.drop_method, ...)
- A set of wrappers to wrap common objects and types that have corresponding async and sync implementations
    (CrossSync.Queue, CrossSync.Condition, CrossSync.Future, ...)
- A set of function implementations for common async operations that can be used in both async and sync codebases
    (CrossSync.gather_partials, CrossSync.wait, CrossSync.condition_wait, ...)
- CrossSync.rm_aio(), which is used to annotate regions of the code containing async keywords to strip

A separate module will use CrossSync annotations to generate a corresponding sync
class based on a decorated async class.

Usage Example:
```python
@CrossSync.export_sync(path="path/to/sync_module.py")

    @CrossSync.convert
    async def async_func(self, arg: int) -> int:
        await CrossSync.sleep(1)
        return arg
```
"""

from __future__ import annotations

from typing import (
    TypeVar,
    Any,
    Callable,
    Coroutine,
    Sequence,
    Union,
    AsyncIterable,
    AsyncIterator,
    AsyncGenerator,
    TYPE_CHECKING,
)
import typing

import asyncio
import sys
import concurrent.futures
import google.api_core.retry as retries
import queue
import threading
import time
from ._decorators import (
    ConvertClass,
    Convert,
    Drop,
    Pytest,
    PytestFixture,
)
from ._mapping_meta import MappingMeta

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

T = TypeVar("T")


class CrossSync(metaclass=MappingMeta):
    # support CrossSync.is_async to check if the current environment is async
    is_async = True

    # provide aliases for common async functions and types
    sleep = asyncio.sleep
    retry_target = retries.retry_target_async
    retry_target_stream = retries.retry_target_stream_async
    Retry = retries.AsyncRetry
    Queue: TypeAlias = asyncio.Queue
    Condition: TypeAlias = asyncio.Condition
    Future: TypeAlias = asyncio.Future
    Task: TypeAlias = asyncio.Task
    Event: TypeAlias = asyncio.Event
    Semaphore: TypeAlias = asyncio.Semaphore
    StopIteration: TypeAlias = StopAsyncIteration
    # provide aliases for common async type annotations
    Awaitable: TypeAlias = typing.Awaitable
    Iterable: TypeAlias = AsyncIterable
    Iterator: TypeAlias = AsyncIterator
    Generator: TypeAlias = AsyncGenerator

    # decorators
    convert_class = ConvertClass.decorator  # decorate classes to convert
    convert = Convert.decorator  # decorate methods to convert from async to sync
    drop = Drop.decorator  # decorate methods to remove from sync version
    pytest = Pytest.decorator  # decorate test methods to run with pytest-asyncio
    pytest_fixture = (
        PytestFixture.decorator
    )  # decorate test methods to run with pytest fixture

    @classmethod
    def next(cls, iterable):
        return iterable.__anext__()

    @classmethod
    def Mock(cls, *args, **kwargs):
        """
        Alias for AsyncMock, importing at runtime to avoid hard dependency on mock
        """
        try:
            from unittest.mock import AsyncMock  # type: ignore
        except ImportError:  # pragma: NO COVER
            from mock import AsyncMock  # type: ignore
        return AsyncMock(*args, **kwargs)

    @staticmethod
    async def gather_partials(
        partial_list: Sequence[Callable[[], Awaitable[T]]],
        return_exceptions: bool = False,
        sync_executor: concurrent.futures.ThreadPoolExecutor | None = None,
    ) -> list[T | BaseException]:
        """
        abstraction over asyncio.gather, but with a set of partial functions instead
        of coroutines, to work with sync functions.
        To use gather with a set of futures instead of partials, use CrpssSync.wait

        In the async version, the partials are expected to return an awaitable object. Patials
        are unpacked and awaited in the gather call.

        Sync version implemented with threadpool executor

        Returns:
          - a list of results (or exceptions, if return_exceptions=True) in the same order as partial_list
        """
        if not partial_list:
            return []
        awaitable_list = [partial() for partial in partial_list]
        return await asyncio.gather(
            *awaitable_list, return_exceptions=return_exceptions
        )

    @staticmethod
    async def wait(
        futures: Sequence[CrossSync.Future[T]], timeout: float | None = None
    ) -> tuple[set[CrossSync.Future[T]], set[CrossSync.Future[T]]]:
        """
        abstraction over asyncio.wait

        Return:
            - a tuple of (done, pending) sets of futures
        """
        if not futures:
            return set(), set()
        return await asyncio.wait(futures, timeout=timeout)

    @staticmethod
    async def event_wait(
        event: CrossSync.Event,
        timeout: float | None = None,
        async_break_early: bool = True,
    ) -> None:
        """
        abstraction over asyncio.Event.wait

        Args:
            - event: event to wait for
            - timeout: if set, will break out early after `timeout` seconds
            - async_break_early: if False, the async version will wait for
                the full timeout even if the event is set before the timeout.
                This avoids creating a new background task
        """
        if timeout is None:
            await event.wait()
        elif not async_break_early:
            if not event.is_set():
                await asyncio.sleep(timeout)
        else:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    @staticmethod
    def create_task(
        fn: Callable[..., Coroutine[Any, Any, T]],
        *fn_args,
        sync_executor: concurrent.futures.ThreadPoolExecutor | None = None,
        task_name: str | None = None,
        **fn_kwargs,
    ) -> CrossSync.Task[T]:
        """
        abstraction over asyncio.create_task. Sync version implemented with threadpool executor

        sync_executor: ThreadPoolExecutor to use for sync operations. Ignored in async version
        """
        task: CrossSync.Task[T] = asyncio.create_task(fn(*fn_args, **fn_kwargs))
        if task_name and sys.version_info >= (3, 8):
            task.set_name(task_name)
        return task

    @staticmethod
    async def yield_to_event_loop() -> None:
        """
        Call asyncio.sleep(0) to yield to allow other tasks to run
        """
        await asyncio.sleep(0)

    @staticmethod
    def verify_async_event_loop() -> None:
        """
        Raises RuntimeError if the event loop is not running
        """
        asyncio.get_running_loop()

    @staticmethod
    def rm_aio(statement: T) -> T:
        """
        Used to annotate regions of the code containing async keywords to strip

        All async keywords inside an rm_aio call are removed, along with
        `async with` and `async for` statements containing CrossSync.rm_aio() in the body
        """
        return statement

    class _Sync_Impl(metaclass=MappingMeta):
        """
        Provide sync versions of the async functions and types in CrossSync
        """

        is_async = False

        sleep = time.sleep
        next = next
        retry_target = retries.retry_target
        retry_target_stream = retries.retry_target_stream
        Retry = retries.Retry
        Queue: TypeAlias = queue.Queue
        Condition: TypeAlias = threading.Condition
        Future: TypeAlias = concurrent.futures.Future
        Task: TypeAlias = concurrent.futures.Future
        Event: TypeAlias = threading.Event
        Semaphore: TypeAlias = threading.Semaphore
        StopIteration: TypeAlias = StopIteration
        # type annotations
        Awaitable: TypeAlias = Union[T]
        Iterable: TypeAlias = typing.Iterable
        Iterator: TypeAlias = typing.Iterator
        Generator: TypeAlias = typing.Generator

        @classmethod
        def Mock(cls, *args, **kwargs):
            from unittest.mock import Mock

            return Mock(*args, **kwargs)

        @staticmethod
        def event_wait(
            event: CrossSync._Sync_Impl.Event,
            timeout: float | None = None,
            async_break_early: bool = True,
        ) -> None:
            event.wait(timeout=timeout)

        @staticmethod
        def gather_partials(
            partial_list: Sequence[Callable[[], T]],
            return_exceptions: bool = False,
            sync_executor: concurrent.futures.ThreadPoolExecutor | None = None,
        ) -> list[T | BaseException]:
            if not partial_list:
                return []
            if not sync_executor:
                raise ValueError("sync_executor is required for sync version")
            futures_list = [sync_executor.submit(partial) for partial in partial_list]
            results_list: list[T | BaseException] = []
            for future in futures_list:
                found_exc = future.exception()
                if found_exc is not None:
                    if return_exceptions:
                        results_list.append(found_exc)
                    else:
                        raise found_exc
                else:
                    results_list.append(future.result())
            return results_list

        @staticmethod
        def wait(
            futures: Sequence[CrossSync._Sync_Impl.Future[T]],
            timeout: float | None = None,
        ) -> tuple[
            set[CrossSync._Sync_Impl.Future[T]], set[CrossSync._Sync_Impl.Future[T]]
        ]:
            if not futures:
                return set(), set()
            return concurrent.futures.wait(futures, timeout=timeout)

        @staticmethod
        def create_task(
            fn: Callable[..., T],
            *fn_args,
            sync_executor: concurrent.futures.ThreadPoolExecutor | None = None,
            task_name: str | None = None,
            **fn_kwargs,
        ) -> CrossSync._Sync_Impl.Task[T]:
            """
            abstraction over asyncio.create_task. Sync version implemented with threadpool executor

            sync_executor: ThreadPoolExecutor to use for sync operations. Ignored in async version
            """
            if not sync_executor:
                raise ValueError("sync_executor is required for sync version")
            return sync_executor.submit(fn, *fn_args, **fn_kwargs)

        @staticmethod
        def yield_to_event_loop() -> None:
            """
            No-op for sync version
            """
            pass

        @staticmethod
        def verify_async_event_loop() -> None:
            """
            No-op for sync version
            """
            pass
