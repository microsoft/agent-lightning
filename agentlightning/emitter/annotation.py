# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting annotation/operation spans."""

import asyncio
import functools
import inspect
import logging
from types import TracebackType
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)

from agentlightning.semconv import AGL_ANNOTATION, AGL_OPERATION, LightningSpanAttributes
from agentlightning.tracer.base import get_active_tracer
from agentlightning.tracer.dummy import DummyTracer
from agentlightning.types import SpanCoreFields, SpanRecordingContext, TraceStatus
from agentlightning.utils.otel import check_attributes_sanity, flatten_attributes, sanitize_attributes

_FnType = TypeVar("_FnType", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def emit_annotation(annotation: Dict[str, Any], propagate: bool = True) -> SpanCoreFields:
    """Emit a new annotation span.

    This is the underlying implementation of [`emit_reward`][agentlightning.emit_reward].

    Annotation spans are used to annotate a specific event or a part of rollout.
    See [semconv][agentlightning.semconv] for conventional annotation keys in Agent-lightning.

    If annotations contain nested dicts, they will be flattened before emitting.
    Complex objects will lead to emitting failures.

    Args:
        annotation: Dictionary containing annotation key-value pairs.
            Representatives are rewards, tags, and metadata.
        propagate: Whether to propagate the span to tracers automatically.
    """
    annotation_attributes = flatten_attributes(annotation, expand_leaf_lists=False)
    check_attributes_sanity(annotation_attributes)
    sanitized_attributes = sanitize_attributes(annotation_attributes)
    logger.debug("Emitting annotation span with keys %s", sanitized_attributes.keys())

    if propagate:
        tracer = get_active_tracer()
        if tracer is None:
            raise RuntimeError("No active tracer found. Cannot emit annotation span.")
    else:
        tracer = DummyTracer()

    return tracer.create_span(
        name=AGL_ANNOTATION,
        attributes=sanitized_attributes,
        status=TraceStatus(status_code="OK"),
    )


class OperationContext:
    """Context manager and decorator for tracing operations.

    This class manages a tracer-backed span for a logical unit of work. It can be
    used either:

    * As a decorator, in which case inputs and outputs are inferred
      automatically from the wrapped function's signature.
    * As a context manager, in which case inputs and outputs can be recorded
      explicitly via [`set_input`][agentlightning.emitter.annotation.OperationContext.set_input]
      and [`set_output`][agentlightning.emitter.annotation.OperationContext.set_output].

    Attributes:
        name: Human-readable span name.
        initial_attributes: Attributes applied when the span is created.
        tracer: Tracer implementation used to create spans.
    """

    def __init__(self, name: str, attributes: Dict[str, Any], propagate: bool = True) -> None:
        """Initialize a new operation context.

        Args:
            name: Human-readable name of the span.
            attributes: Initial attributes attached to the span. Values are
                JSON-serialized where necessary.
            propagate: Whether the span should be sent to active exporters.
        """
        self.name = name
        self.initial_attributes = attributes
        self.propagate = propagate
        if propagate:
            tracer = get_active_tracer()
            if tracer is None:
                raise RuntimeError("No active tracer found. Cannot trace operation spans.")
            self.tracer = tracer
        else:
            self.tracer = DummyTracer()
        self._ctx_manager: Optional[ContextManager[SpanRecordingContext]] = None
        self._recording_context: Optional[SpanRecordingContext] = None
        self._span: Optional[SpanCoreFields] = None

    def __enter__(self) -> "OperationContext":
        """Enter the context manager and start a new span.

        Returns:
            The current :class:`OperationContext` instance with an active span.
        """
        sanitized_attrs = sanitize_attributes(self.initial_attributes)
        self._ctx_manager = self.tracer.operation_context(self.name, attributes=sanitized_attrs)
        recording_context = self._ctx_manager.__enter__()
        self._recording_context = recording_context
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the context manager and finish the span."""
        if self._ctx_manager:
            self._ctx_manager.__exit__(exc_type, exc_val, exc_tb)
        if self._recording_context:
            self._span = self._recording_context.get_recorded_span()
        self._ctx_manager = None
        self._recording_context = None

    def span(self) -> SpanCoreFields:
        """Get the span that was created by this context manager."""
        if self._span is None:
            raise RuntimeError("Span is not ready yet.")
        return self._span

    def set_input(self, *args: Any, **kwargs: Any) -> None:
        """Record input arguments on the current span.

        Positional arguments are stored under the `input.args` attribute,
        and keyword arguments are stored under `input.<name>` attributes.

        This is intended for use inside a `with operation(...) as op` block.

        Args:
            *args: Positional arguments to record.
            **kwargs: Keyword arguments to record.
        """
        if not self._recording_context:
            raise RuntimeError("No recording context found. Cannot set input.")

        attributes: Dict[str, Any] = {}
        if args:
            attributes[f"{LightningSpanAttributes.OPERATION_INPUT.value}.args"] = args
        if kwargs:
            for k, v in kwargs.items():
                attributes[f"{LightningSpanAttributes.OPERATION_INPUT.value}.{k}"] = v
        if attributes:
            self._recording_context.record_attributes(sanitize_attributes(attributes))

    def set_output(self, output: Any) -> None:
        """Record the output value on the current span.

        This is intended for use inside a `with operation(...) as op` block.

        Args:
            output: The output value to record.
        """
        if not self._recording_context:
            raise RuntimeError("No recording context found. Cannot set output.")

        self._recording_context.record_attributes(
            sanitize_attributes({LightningSpanAttributes.OPERATION_OUTPUT.value: output})
        )

    def __call__(self, fn: _FnType) -> _FnType:
        """Wrap a callable so its execution is traced in a span.

        When used as a decorator, a new span is created for each call to
        the wrapped function. The bound arguments are recorded as input
        attributes, the return value is recorded as an output attribute,
        and any exception is recorded and marks the span as an error.

        Args:
            fn: The function or coroutine function to wrap.

        Returns:
            The wrapped callable.
        """
        function_name = fn.__name__

        sig = inspect.signature(fn)

        sanitized_init_attrs = sanitize_attributes(
            {LightningSpanAttributes.OPERATION_NAME.value: function_name, **self.initial_attributes}
        )

        def _record_auto_inputs(
            recording_ctx: SpanRecordingContext, args: Tuple[Any, ...], kwargs: Dict[str, Any]
        ) -> None:
            """Bind arguments to signature and log them on the span."""
            attributes: Dict[str, Any] = {}
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for k, v in bound.arguments.items():
                    attributes[f"{LightningSpanAttributes.OPERATION_INPUT.value}.{k}"] = v
            except Exception:
                attributes[f"{LightningSpanAttributes.OPERATION_INPUT.value}.args"] = args
                attributes[f"{LightningSpanAttributes.OPERATION_INPUT.value}.kwargs"] = kwargs
            if attributes:
                recording_ctx.record_attributes(sanitize_attributes(attributes))

        def _record_auto_outputs(recording_ctx: SpanRecordingContext, result: Any) -> None:
            """Record the output value on the span."""
            recording_ctx.record_attributes(
                sanitize_attributes({LightningSpanAttributes.OPERATION_OUTPUT.value: result})
            )

        if asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper that traces the wrapped coroutine."""
                with self.tracer.operation_context(self.name, attributes=sanitized_init_attrs) as recording_ctx:
                    _record_auto_inputs(recording_ctx, args, kwargs)
                    result = await fn(*args, **kwargs)
                    _record_auto_outputs(recording_ctx, result)
                    return result

            return cast(_FnType, async_wrapper)

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Sync wrapper that traces the wrapped callable."""
                with self.tracer.operation_context(self.name, attributes=sanitized_init_attrs) as recording_ctx:
                    _record_auto_inputs(recording_ctx, args, kwargs)
                    result = fn(*args, **kwargs)
                    _record_auto_outputs(recording_ctx, result)
                    return result

            return cast(_FnType, sync_wrapper)


@overload
def operation(fn: _FnType, *, propagate: bool = True, **additional_attributes: Any) -> _FnType: ...


@overload
def operation(*, propagate: bool = True, **additional_attributes: Any) -> OperationContext: ...


def operation(
    fn: Optional[_FnType] = None,
    *,
    propagate: bool = True,
    **additional_attributes: Any,
) -> Union[_FnType, OperationContext]:
    """Entry point for tracking operations.

    This helper can be used either as a decorator or as a context manager.
    The span name is fixed to [`AGL_OPERATION`][agentlightning.semconv.AGL_OPERATION];
    custom span names are not supported. Any keyword arguments are recorded as span attributes.

    Usage as a decorator:

    ```python
    @operation
    def func(...):
        ...

    @operation(category="compute")
    def func(...):
        ...
    ```

    Usage as a context manager:

    ```python
    with operation(user_id=123) as op:
        op.set_input(data=data)
        # ... do work ...
        op.set_output(result)
    ```

    Args:
        fn: When used as `@operation`, this is the wrapped function.
            When used as `operation(**attrs)`, this should be omitted (or
            left as `None`) and only keyword attributes are provided.
        propagate: Whether spans should use the active span processor. When False,
            spans will stay local and not be exported.
        **additional_attributes: Additional span attributes to attach at
            creation time.

    Returns:
        Either a wrapped callable (when used as a decorator) or an
        [`OperationContext`][agentlightning.emitter.annotation.OperationContext]
        (when used as a context manager factory).
    """
    # Case 1: Used as @operation (bare decorator or with attributes)
    if callable(fn):
        # Create context with fixed name, then immediately wrap the function
        return OperationContext(AGL_OPERATION, additional_attributes, propagate=propagate)(fn)

    # Case 2: Used as operation(...) / with operation(...)
    # Custom span names are intentionally not supported; use AGL_OPERATION.
    if fn is not None:
        raise ValueError("Custom span names are intentionally not supported when used as a context manager.")
    return OperationContext(AGL_OPERATION, additional_attributes, propagate=propagate)
