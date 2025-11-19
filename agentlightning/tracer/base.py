# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, AsyncContextManager, Awaitable, Callable, ContextManager, List, Optional

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store.base import LightningStore
from agentlightning.types import ParallelWorkerBase

if TYPE_CHECKING:
    from langchain_core.callbacks.base import BaseCallbackHandler  # type: ignore

logger = logging.getLogger(__name__)


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

    def get_last_trace(self) -> List[ReadableSpan]:
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
