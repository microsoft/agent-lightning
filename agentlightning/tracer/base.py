# Copyright (c) Microsoft. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Any, Awaitable, Callable, Iterator, List, Optional

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.store.base import LightningStore
from agentlightning.types import ParallelWorkerBase

logger = logging.getLogger(__name__)


class Tracer(ParallelWorkerBase):
    """
    An abstract base class for tracers.

    This class defines a standard interface for tracing code execution,
    capturing the resulting spans, and providing them for analysis. It is
    designed to be backend-agnostic, allowing for different implementations
    (e.g., for AgentOps, OpenTelemetry, Docker, etc.).

    The primary interaction pattern is through the `trace_context`
    context manager, which ensures that traces are properly started and captured,
    even in the case of exceptions.

    A typical workflow:

    ```python
    tracer = YourTracerImplementation()

    try:
        with tracer.trace_context(name="my_traced_task"):
            # ... code to be traced ...
            run_my_agent_logic()
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

    @contextmanager
    def trace_context(
        self,
        name: Optional[str] = None,
        *,
        store: Optional[LightningStore] = None,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> Iterator[Any]:
        """
        Starts a new tracing context. This should be used as a context manager.

        The implementation should handle the setup and teardown of the tracing
        for the enclosed code block. It must ensure that any spans generated
        within the `with` block are collected and made available via
        `get_last_trace`.

        If a store is provided, the spans will be added to the store when tracing.

        Args:
            name: The name for the root span of this trace context.
            store: The store to add the spans to.
            rollout_id: The rollout ID to add the spans to.
            attempt_id: The attempt ID to add the spans to.
        """
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

        Args:
            func: The synchronous function to execute and trace.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        with self.trace_context(name=func.__name__):
            return func(*args, **kwargs)

    async def trace_run_async(self, func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any) -> Any:
        """
        A convenience wrapper to trace the execution of a single asynchronous function.

        Args:
            func: The asynchronous function to execute and trace.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The return value of the function.
        """
        with self.trace_context(name=func.__name__):
            return await func(*args, **kwargs)
