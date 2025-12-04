# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting annotation/operation spans."""

import asyncio
import functools
import inspect
import json
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

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import Status, StatusCode

from agentlightning.semconv import AGL_ANNOTATION, AGL_OPERATION, LightningSpanAttributes
from agentlightning.utils.otel import flatten_attributes, get_tracer

_FnType = TypeVar("_FnType", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def emit_annotation(annotation: Dict[str, Any], propagate: bool = True) -> ReadableSpan:
    """Emit a new annotation span.

    This is the underlying implementation of [`emit_reward`][agentlightning.emit_reward].

    Annotation spans are used to annotate a specific event or a part of rollout.
    See [semconv][agentlightning.semconv] for conventional annotation keys in Agent-lightning.

    If annotations contain nested dicts, they will be flattened before emitting.
    Complex objects will lead to emitting failures.

    Args:
        annotation: Dictionary containing annotation key-value pairs.
            Representatives are rewards, tags, and metadata.
        propagate: Whether to propagate the span to exporters automatically.
    """
    annotation_attributes = flatten_attributes(annotation)
    if any(not isinstance(v, (str, int, float, bool, bytes)) for v in annotation_attributes.values()):
        raise TypeError("All annotation attributes must be primitive types (str, int, float, bool, bytes)")

    # TODO: this should use a tracer from current context rather than the singleton
    tracer = get_tracer(use_active_span_processor=propagate)
    span = tracer.start_span(
        AGL_ANNOTATION,
        attributes=annotation_attributes,
    )
    logger.debug("Emitting annotation span with keys %s", annotation_attributes)
    with span:
        pass
    if not isinstance(span, ReadableSpan):
        raise ValueError(f"Span is not a ReadableSpan: {span}")

    return span


def _safe_json_dump(obj: Any) -> str:
    """Serialize an object to JSON, falling back to ``str(obj)`` if needed.

    Args:
        obj: Object to be serialized.

    Returns:
        The JSON-encoded string representation of the object, or its string
        representation if JSON encoding fails.
    """
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


class OperationContext:
    """Context manager and decorator for tracing operations.

    This class manages an OpenTelemetry span for a logical unit of work. It can
    be used either:

    * As a decorator, in which case inputs and outputs are inferred
      automatically from the wrapped function's signature.
    * As a context manager, in which case inputs and outputs can be recorded
      explicitly via :meth:`set_input` and :meth:`set_output`.

    Attributes:
        name: Human-readable span name.
        initial_attributes: Attributes applied when the span is created.
        tracer: OpenTelemetry tracer used to create spans.
        span: The currently active span, if any.
    """

    def __init__(self, name: str, attributes: Dict[str, Any], *, propagate: bool = True) -> None:
        """Initialize a new operation context.

        Args:
            name: Human-readable name of the span.
            attributes: Initial attributes attached to the span. Values are
                JSON-serialized where necessary.
            propagate: Whether the span should be sent to active exporters.
        """
        self.name: str = name
        self.initial_attributes: Dict[str, Any] = attributes
        self.propagate: bool = propagate
        self.tracer: trace.Tracer = get_tracer(use_active_span_processor=propagate)
        self.span: Optional[trace.Span] = None
        self._ctx_token: Optional[ContextManager[Any]] = None

    def __enter__(self) -> "OperationContext":
        """Enter the context manager and start a new span.

        Returns:
            The current :class:`OperationContext` instance with an active span.
        """
        # 1. Start the span with initial attributes (JSON serialized)
        sanitized_attrs = {
            k: _safe_json_dump(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in self.initial_attributes.items()
        }

        self.span = self.tracer.start_span(self.name, attributes=sanitized_attrs)
        self._ctx_token = trace.use_span(self.span, end_on_exit=True)
        self._ctx_token.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Exit the context manager and finish the span.

        Any exception raised inside the context is recorded on the span and the
        span status is set to error.

        Args:
            exc_type: Exception type, if an exception occurred.
            exc_val: Exception instance, if an exception occurred.
            exc_tb: Traceback object, if an exception occurred.
        """
        # 1. Record Exception if present
        if exc_val and self.span:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))

        # 2. Close span
        if self._ctx_token:
            self._ctx_token.__exit__(exc_type, exc_val, exc_tb)

    def set_input(self, *args: Any, **kwargs: Any) -> None:
        """Record input arguments on the current span.

        Positional arguments are stored under the ``input.args`` attribute,
        and keyword arguments are stored under ``input.<name>`` attributes.

        This is intended for use inside a ``with operation(...) as op`` block.

        Args:
            *args: Positional arguments to record.
            **kwargs: Keyword arguments to record.
        """
        if not self.span:
            return

        if args:
            self.span.set_attribute("input.args", _safe_json_dump(args))
        if kwargs:
            for k, v in kwargs.items():
                self.span.set_attribute(f"input.{k}", _safe_json_dump(v))

    def set_output(self, output: Any) -> None:
        """Record the output value on the current span.

        This is intended for use inside a ``with operation(...) as op`` block.

        Args:
            output: The output value to record.
        """
        if not self.span:
            return
        self.span.set_attribute("output", _safe_json_dump(output))

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

        def _record_auto_inputs(span: trace.Span, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> None:
            """Bind arguments to signature and log them on the span.

            Args:
                span: Span on which to record attributes.
                args: Positional arguments passed to the wrapped callable.
                kwargs: Keyword arguments passed to the wrapped callable.
            """
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for k, v in bound.arguments.items():
                    span.set_attribute(
                        f"{LightningSpanAttributes.OPERATION_INPUT.value}.{k}",
                        _safe_json_dump(v),
                    )
            except Exception:
                span.set_attribute(
                    f"{LightningSpanAttributes.OPERATION_INPUT.value}.args",
                    _safe_json_dump(args),
                )
                span.set_attribute(
                    f"{LightningSpanAttributes.OPERATION_INPUT.value}.kwargs",
                    _safe_json_dump(kwargs),
                )

        if asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Async wrapper that traces the wrapped coroutine."""
                # Reuse __enter__ logic via 'with self' would share state incorrectly
                # across concurrent calls. We must create a new span per call.
                # So we manually reimplement the span logic for the wrapper here.

                sanitized_attrs = {
                    k: _safe_json_dump(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in self.initial_attributes.items()
                }

                with self.tracer.start_as_current_span(self.name, attributes=sanitized_attrs) as span:
                    span.set_attribute(LightningSpanAttributes.OPERATION_NAME.value, function_name)
                    _record_auto_inputs(span, args, kwargs)
                    try:
                        result = await fn(*args, **kwargs)
                        span.set_attribute(
                            LightningSpanAttributes.OPERATION_OUTPUT.value,
                            _safe_json_dump(result),
                        )
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return cast(_FnType, async_wrapper)

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                """Sync wrapper that traces the wrapped callable."""
                sanitized_attrs = {
                    k: _safe_json_dump(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in self.initial_attributes.items()
                }

                with self.tracer.start_as_current_span(self.name, attributes=sanitized_attrs) as span:
                    span.set_attribute(LightningSpanAttributes.OPERATION_NAME.value, function_name)
                    _record_auto_inputs(span, args, kwargs)
                    try:
                        result = fn(*args, **kwargs)
                        span.set_attribute(
                            LightningSpanAttributes.OPERATION_OUTPUT.value,
                            _safe_json_dump(result),
                        )
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

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
