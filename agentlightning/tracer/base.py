# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncContextManager, Awaitable, Callable, ContextManager, List, Optional, TypeVar

from agentlightning.store.base import LightningStore
from agentlightning.types import Attributes, ParallelWorkerBase, Span, SpanCoreFields, SpanRecordingContext, TraceStatus

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore

logger = logging.getLogger(__name__)


T = TypeVar("T")


_active_tracer: Optional[Tracer] = None

T_func = Callable[..., Awaitable[Any]]


class Tracer(ParallelWorkerBase):
    """
    An abstract base class for tracers.

    This class defines a standard interface for tracing code execution,
    capturing the resulting spans, and providing them for analysis. It is
    designed to be backend-agnostic, allowing for different implementations
    (e.g., for AgentOps, OpenTelemetry, Docker, etc.).

    The primary interaction pattern is through the [`trace_context`][agentlightning.Tracer.trace_context]
    context manager, which ensures that traces are properly started and captured,
    even in the case of exceptions.

    A typical workflow:

    ```python
    tracer = YourTracerImplementation()

    try:
        async with tracer.trace_context(name="my_traced_task"):
            # ... code to be traced ...
            await run_my_agent_logic()
    except Exception as e:
        print(f"An error occurred: {e}")

    # Retrieve the trace data after the context block
    spans: list[ReadableSpan] = tracer.get_last_trace()

    # Process the trace data
    if trace_tree:
        rl_triplets = TracerTraceToTriplet().adapt(spans)
        # ... do something with the triplets
    ```
    """

    _store: Optional[LightningStore] = None

    def init_worker(self, worker_id: int, store: Optional[LightningStore] = None) -> None:
        """Initialize the tracer for a worker.

        Args:
            worker_id: The ID of the worker.
            store: The store to add the spans to. If it's provided, traces will be added to the store when tracing.
        """
        super().init_worker(worker_id)
        self._store = store

    def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> AsyncContextManager[Any]:
        """
        Starts a new tracing context. This should be used as a context manager.

        The implementation should handle the setup and teardown of the tracing
        for the enclosed code block. It must ensure that any spans generated
        within the `with` block are collected and made available via
        [`get_last_trace`][agentlightning.Tracer.get_last_trace].

        Args:
            name: The name for the root span of this trace context.
            store: The store to add the spans to. Deprecated in favor of passing store to init_worker().
            rollout_id: The rollout ID to add the spans to.
            attempt_id: The attempt ID to add the spans to.
        """
        raise NotImplementedError()

    def _trace_context_sync(
        self,
        name: Optional[str] = None,
        *,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> ContextManager[Any]:
        """Internal API for CI backward compatibility."""
        raise NotImplementedError()

    def get_last_trace(self) -> List[Span]:
        """
        Retrieves the raw list of captured spans from the most recent trace.

        Returns:
            A list of OpenTelemetry `ReadableSpan` objects.
        """
        raise NotImplementedError()

    def trace_run(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        A convenience wrapper to trace the execution of a single synchronous function.

        Deprecated in favor of customizing Runners.

        Args:
            func: The synchronous function to execute and trace.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        with self._trace_context_sync(name=func.__name__):
            return func(*args, **kwargs)

    def create_span(
        self,
        name: str,
        attributes: Optional[Attributes] = None,
        timestamp: Optional[float] = None,
        status: Optional[TraceStatus] = None,
    ) -> SpanCoreFields:
        """Notify the tracer that a span should be created here.

        It uses a fire-and-forget approach and doesn't wait for the span to be created.

        Args:
            name: The name of the span.
            attributes: The attributes of the span.
            timestamp: The timestamp of the span.
            status: The status of the span.

        Returns:
            The core fields of the span.
        """
        raise NotImplementedError()

    def operation_context(
        self, name: str, attributes: Optional[Attributes] = None, start_time: Optional[float] = None
    ) -> ContextManager[SpanRecordingContext]:
        """Start to record an operation to a span.

        Args:
            name: The name of the operation.
            attributes: The attributes of the operation.
            start_time: The start time of the operation.

        Returns:
            A context for recording the operation on the span.
            Call [`finish()`][agentlightning.SpanRecordingContext.finish] when the recording is complete.
        """
        raise NotImplementedError()

    async def trace_run_async(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """
        A convenience wrapper to trace the execution of a single asynchronous function.

        Deprecated in favor of customizing Runners.

        Args:
            func: The asynchronous function to execute and trace.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        async with self.trace_context(name=func.__name__):
            return await func(*args, **kwargs)

    def get_langchain_handler(self) -> Optional[BaseCallbackHandler]:  # type: ignore
        """Get a handler to install in langchain agent callback.

        Agents are expected to use this handler in their agents to enable tracing.
        """
        logger.warning(f"{self.__class__.__name__} does not provide a LangChain callback handler.")
        return None

    @contextmanager
    def lifespan(self, store: Optional[LightningStore] = None):
        """A context manager to manage the lifespan of the tracer.

        This can be used to set up and tear down any necessary resources
        for the tracer, useful for debugging purposes.

        Args:
            store: The store to add the spans to. If it's provided, traces will be added to the store when tracing.
        """
        has_init = False
        has_init_worker = False
        try:
            self.init()
            has_init = True

            self.init_worker(0, store)
            has_init_worker = True

            yield

        finally:
            if has_init_worker:
                self.teardown_worker(0)
            if has_init:
                self.teardown()


def set_active_tracer(tracer: Tracer):
    """Set the active tracer for the current process.

    Args:
        tracer: The tracer to set as active.
    """
    global _active_tracer
    if _active_tracer is not None:
        raise ValueError("An active tracer is already set. Cannot set a new one.")
    _active_tracer = tracer


def clear_active_tracer():
    """Clear the active tracer for the current process."""
    global _active_tracer
    _active_tracer = None


def get_active_tracer() -> Optional[Tracer]:
    """Get the active tracer for the current process.

    Returns:
        The active tracer, or None if no tracer is active.
    """
    global _active_tracer
    return _active_tracer


class _ActiveTracerAsyncCM(AsyncContextManager[T]):
    def __init__(self, tracer: Tracer, inner: AsyncContextManager[T]):
        self._tracer = tracer
        self._inner = inner

    async def __aenter__(self) -> T:
        set_active_tracer(self._tracer)  # will raise if nested
        try:
            return await self._inner.__aenter__()
        except Exception:
            clear_active_tracer()
            raise

    async def __aexit__(self, *args: Any, **kwargs: Any) -> Optional[bool]:
        try:
            return await self._inner.__aexit__(*args, **kwargs)
        finally:
            clear_active_tracer()


def with_active_tracer_context(
    func: Callable[..., AsyncContextManager[T]],
) -> Callable[..., AsyncContextManager[T]]:
    """Decorate a method returning an AsyncContextManager so tracer is active for the whole `async with`."""

    @functools.wraps(func)
    def wrapper(self: Tracer, *args: Any, **kwargs: Any) -> AsyncContextManager[T]:
        cm = func(self, *args, **kwargs)
        return _ActiveTracerAsyncCM(self, cm)

    return wrapper
