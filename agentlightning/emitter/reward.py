# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting reward spans and integrating with AgentOps telemetry."""

import asyncio
import inspect
import json
import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    TypeVar,
    cast,
)

import agentops
from agentops.sdk.decorators import operation
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import SpanLike, SpanNames

from .utils import get_tracer

logger = logging.getLogger(__name__)

__all__ = [
    "reward",
    "emit_reward",
    "get_reward_value",
    "is_reward_span",
    "find_reward_spans",
    "find_final_reward",
]


class RewardSpanData(TypedDict):
    type: Literal["reward"]
    value: Optional[float]


FnType = TypeVar("FnType", bound=Callable[..., Any])


def _agentops_initialized() -> bool:
    """Return `True` when the AgentOps client has been configured."""
    return agentops.get_client().initialized


def reward(fn: FnType) -> FnType:
    """Decorate a reward function so its outputs are tracked as spans.

    The decorator integrates with AgentOps when it is available and falls back to
    the built-in telemetry otherwise. Both synchronous and asynchronous functions
    are supported transparently.

    Deprecated:
        This decorator is deprecated. Use [`emit_reward`][agentlightning.emit_reward] instead.

    Args:
        fn: Callable that produces a numeric reward.

    Returns:
        Wrapped callable that preserves the original signature.
    """

    def wrap_result(result: Optional[float]) -> RewardSpanData:
        """Normalize the reward value into the span payload format."""
        if result is None:
            return {"type": "reward", "value": None}
        if not isinstance(result, (float, int)):  # type: ignore
            warnings.warn(f"Reward is ignored because it is not a number: {result}")
            return {"type": "reward", "value": None}
        return {"type": "reward", "value": float(result)}

    # Check if the function is async
    is_async = asyncio.iscoroutinefunction(fn) or inspect.iscoroutinefunction(fn)

    if is_async:

        async def wrapper_async(*args: Any, **kwargs: Any) -> Any:
            if not _agentops_initialized():
                # Track the reward without AgentOps
                result = await fn(*args, **kwargs)
                emit_reward(cast(float, result))
                return result

            result: Optional[float] = None

            @operation
            async def agentops_reward_operation() -> RewardSpanData:
                # The reward function we are interested in tracing
                # It takes zero inputs and return a formatted dict
                nonlocal result
                result = await fn(*args, **kwargs)
                return wrap_result(result)

            await agentops_reward_operation()
            return result

        return wrapper_async  # type: ignore

    else:

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _agentops_initialized():
                # Track the reward without AgentOps
                result = fn(*args, **kwargs)
                emit_reward(cast(float, result))
                return result

            result: Optional[float] = None

            @operation
            def agentops_reward_operation() -> RewardSpanData:
                nonlocal result
                result = fn(*args, **kwargs)
                return wrap_result(result)

            agentops_reward_operation()
            return result

        return wrapper  # type: ignore


def emit_reward(reward: float, auto_export: bool = True) -> ReadableSpan:
    """Emit a reward value as an OpenTelemetry span.

    Args:
        reward: Numeric reward to record. Integers and booleans are converted to
            floating point numbers for consistency.
        auto_export: Whether to export the span automatically.

    Returns:
        Readable span capturing the recorded reward.

    Raises:
        ValueError: If the provided reward cannot be interpreted as a float or the
            resulting span is not a [`ReadableSpan`](https://opentelemetry.io/docs/concepts/signals/traces/) instance.
    """
    logger.debug(f"Emitting reward: {reward}")
    if isinstance(reward, (int, bool)):
        reward = float(reward)
    if not isinstance(reward, float):
        raise ValueError(f"Reward must be a number, got: {type(reward)}")

    # TODO: This should use the tracer from current context by tracer
    tracer = get_tracer(use_active_span_processor=auto_export)
    span = tracer.start_span(SpanNames.REWARD.value, attributes={"reward": reward})
    # Do nothing; it's just a number
    with span:
        pass
    if not isinstance(span, ReadableSpan):
        raise ValueError(f"Span is not a ReadableSpan: {span}")
    return span


def get_reward_value(span: SpanLike) -> Optional[float]:
    """Extract the reward value from a span, if available.

    Args:
        span: Span object produced by AgentOps or Agent Lightning emitters.

    Returns:
        The reward encoded in the span or `None` when the span does not represent a reward.
    """
    for key in [
        "agentops.task.output",  # newer versions of agentops
        "agentops.entity.output",
    ]:
        reward_dict: Dict[str, Any] | None = None
        if span.attributes:
            output = span.attributes.get(key)
            if output:
                if isinstance(output, dict):
                    reward_dict = cast(Dict[str, Any], output)
                elif isinstance(output, str):
                    try:
                        reward_dict = cast(Dict[str, Any], json.loads(output))
                    except json.JSONDecodeError:
                        reward_dict = None

        if reward_dict and reward_dict.get("type") == "reward":
            reward_value = reward_dict.get("value", None)
            if reward_value is None:
                return None
            if not isinstance(reward_value, float):
                logger.error(f"Reward is not a number, got: {type(reward_value)}. This may cause undefined behaviors.")
            return cast(float, reward_value)

    # Latest emit reward format
    if span.name == SpanNames.REWARD.value and span.attributes:
        reward_value = span.attributes.get("reward", None)
        if reward_value is None:
            return None
        if not isinstance(reward_value, float):
            logger.error(f"Reward is not a number, got: {type(reward_value)}. This may cause undefined behaviors.")
        return cast(float, reward_value)
    return None


def is_reward_span(span: SpanLike) -> bool:
    """Return ``True`` when the provided span encodes a reward value."""
    maybe_reward = get_reward_value(span)
    return maybe_reward is not None


def find_reward_spans(spans: Sequence[SpanLike]) -> List[SpanLike]:
    """Return all reward spans in the provided sequence.

    Args:
        spans: Sequence containing [`ReadableSpan`](https://opentelemetry.io/docs/concepts/signals/traces/) objects or mocked span-like values.

    Returns:
        List of spans that could be parsed as rewards.
    """
    return [span for span in spans if is_reward_span(span)]


def find_final_reward(spans: Sequence[SpanLike]) -> Optional[float]:
    """Return the last reward value present in the provided spans.

    Args:
        spans: Sequence containing [`ReadableSpan`](https://opentelemetry.io/docs/concepts/signals/traces/) objects or mocked span-like values.

    Returns:
        Reward value from the latest reward span, or `None` when none are found.
    """
    for span in reversed(spans):
        reward = get_reward_value(span)
        if reward is not None:
            return reward
    return None
