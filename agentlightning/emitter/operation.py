# Copyright (c) Microsoft. All rights reserved.

import asyncio
import functools
import inspect
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from agentlightning.utils.otel import flatten_attributes, get_tracer

_FnType = TypeVar("_FnType", bound=Callable[..., Any])

def operation(fn: Optional[_FnType] = None, **decorator_attrs: Any) -> Any:
    """
    Decorate a function to wrap its execution in an OpenTelemetry span.

    This creates a "long" span that measures duration.
    - Inputs are recorded as span attributes at the start.
    - Outputs are recorded as span attributes at the end.
    - Exceptions are recorded as events and span status is set to Error.

    Usage:
        @operation
        def my_func(x): ...

        @operation(category="core", crucial=True)
        async def my_async_func(x): ...
    """
    
    # Handle usage as @operation(key="val")
    if fn is None:
        return functools.partial(operation, **decorator_attrs)

    sig = inspect.signature(fn)

    def _get_sanitized_attributes(args: tuple, kwargs: dict) -> Dict[str, Any]:
        """
        Binds arguments, merges with decorator attributes, and flattens
        everything into OTel-compatible primitives.
        """
        raw_attributes = {**decorator_attrs}

        # 1. Bind Arguments to Parameter Names
        try:
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            for name, value in bound_args.arguments.items():
                raw_attributes[f"input.{name}"] = value
        except Exception:
            # Fallback if binding fails
            raw_attributes["input.args"] = str(args)
            raw_attributes["input.kwargs"] = str(kwargs)

        # 2. Flatten and Sanitize (converts nested dicts to dot.notation and objs to str)
        # using the existing utility from your project
        return flatten_attributes(raw_attributes)

    def _record_output(span: trace.Span, result: Any):
        """Helper to safely record the output."""
        # We manually sanitize here because flatten_attributes might be heavy 
        # for just a single value, and we need to handle None/Types carefully.
        if result is None:
            return 
        
        if isinstance(result, (str, int, float, bool)):
            span.set_attribute("output", result)
        else:
            span.set_attribute("output", str(result))

    # --- Async Wrapper ---
    if asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer() 
            attributes = _get_sanitized_attributes(args, kwargs)
            
            # Start the span
            with tracer.start_as_current_span(fn.__name__, attributes=attributes) as span:
                try:
                    result = await fn(*args, **kwargs)
                    _record_output(span, result)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise e
        return cast(_FnType, async_wrapper)

    # --- Sync Wrapper ---
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            attributes = _get_sanitized_attributes(args, kwargs)

            with tracer.start_as_current_span(fn.__name__, attributes=attributes) as span:
                try:
                    result = fn(*args, **kwargs)
                    _record_output(span, result)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise e
        return cast(_FnType, sync_wrapper)