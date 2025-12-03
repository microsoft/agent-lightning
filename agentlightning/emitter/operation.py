# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting operation spans."""

import asyncio
import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from agentlightning.semconv import LightningSpanAttributes
from agentlightning.utils.otel import get_tracer

_FnType = TypeVar("_FnType", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def _safe_json_dump(obj: Any) -> str:
    """Serialize object to JSON, falling back to string representation if needed."""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


class OperationContext:
    """
    A context manager and decorator for tracing operations.

    Acts as the controller for the OTel span, allowing dynamic setting of
    inputs/outputs when used in a 'with' block, or automatic inference
    when used as a decorator.
    """

    def __init__(self, name: str, attributes: Dict[str, Any]):
        self.name = name
        self.initial_attributes = attributes
        self.tracer = get_tracer()
        self.span: Optional[trace.Span] = None
        self._ctx_token = None

    def __enter__(self):
        # 1. Start the span with initial attributes (JSON serialized)
        sanitized_attrs = {
            k: _safe_json_dump(v) if not isinstance(v, (str, int, float, bool)) else v
            for k, v in self.initial_attributes.items()
        }

        self.span = self.tracer.start_span(self.name, attributes=sanitized_attrs)
        self._ctx_token = trace.use_span(self.span, end_on_exit=True)
        self._ctx_token.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):
        # 1. Record Exception if present
        if exc_val and self.span:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))

        # 2. Close span
        if self._ctx_token:
            self._ctx_token.__exit__(exc_type, exc_val, exc_tb)

    def set_input(self, *args: Any, **kwargs: Any) -> None:
        """
        Manually record inputs in the span attributes.
        Used inside 'with operation(...) as op'.
        """
        if not self.span:
            return

        if args:
            self.span.set_attribute("input.args", _safe_json_dump(args))
        if kwargs:
            for k, v in kwargs.items():
                self.span.set_attribute(f"input.{k}", _safe_json_dump(v))

    def set_output(self, output: Any) -> None:
        """
        Manually record output in the span attributes.
        Used inside 'with operation(...) as op'.
        """
        if not self.span:
            return
        self.span.set_attribute("output", _safe_json_dump(output))

    def __call__(self, fn: _FnType) -> _FnType:
        """Decorator implementation (@operation)."""
        # If the class is called, it means it's being used as a decorator
        # We override the name with the function name if it was default
        if self.name == "operation":
            self.name = fn.__name__

        sig = inspect.signature(fn)

        def _record_auto_inputs(span: trace.Span, args: Tuple[Any, ...], kwargs: Dict[str, Any]):
            """Bind arguments to signature and log them."""
            try:
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                for k, v in bound.arguments.items():
                    span.set_attribute(f"{LightningSpanAttributes.OPERATION_INPUT.value}.{k}", _safe_json_dump(v))
            except Exception:
                span.set_attribute(f"{LightningSpanAttributes.OPERATION_INPUT.value}.args", _safe_json_dump(args))
                span.set_attribute(f"{LightningSpanAttributes.OPERATION_INPUT.value}.kwargs", _safe_json_dump(kwargs))

        if asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                # Reuse __enter__ logic via 'with self' would share state incorrectly
                # across concurrent calls. We must create a new span per call.
                # So we manually reimplement the span logic for the wrapper here.

                sanitized_attrs = {
                    k: _safe_json_dump(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in self.initial_attributes.items()
                }

                with self.tracer.start_as_current_span(self.name, attributes=sanitized_attrs) as span:
                    _record_auto_inputs(span, args, kwargs)
                    try:
                        result = await fn(*args, **kwargs)
                        span.set_attribute(LightningSpanAttributes.OPERATION_OUTPUT.value, _safe_json_dump(result))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return cast(_FnType, async_wrapper)

        else:

            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                sanitized_attrs = {
                    k: _safe_json_dump(v) if not isinstance(v, (str, int, float, bool)) else v
                    for k, v in self.initial_attributes.items()
                }

                with self.tracer.start_as_current_span(self.name, attributes=sanitized_attrs) as span:
                    _record_auto_inputs(span, args, kwargs)
                    try:
                        result = fn(*args, **kwargs)
                        span.set_attribute(LightningSpanAttributes.OPERATION_OUTPUT.value, _safe_json_dump(result))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            return cast(_FnType, sync_wrapper)


def operation(
    fn_or_name: Optional[Union[_FnType, str]] = None, **additional_attributes: Any
) -> Union[_FnType, OperationContext]:
    """
    Entry point for tracking operations.

    Usage 1: Decorator
        @operation
        def func(): ...

        @operation(category="compute")
        def func(): ...

    Usage 2: Context Manager
        with operation(name="complex_step", user_id=123) as op:
            op.set_input(data=data)
            # ... do work ...
            op.set_output(result)
    """

    # Case 1: Used as @operation (bare decorator)
    if callable(fn_or_name):
        func = fn_or_name
        # Create context with default name, then immediately wrap the function
        return OperationContext("operation", additional_attributes)(func)

    # Case 2: Used as @operation(...) or with operation(...)
    # fn_or_name is likely a string name, or None
    name = fn_or_name if isinstance(fn_or_name, str) else "operation"
    return OperationContext(name, additional_attributes)
