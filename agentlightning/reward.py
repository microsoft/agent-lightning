# Copyright (c) Microsoft. All rights reserved.

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
    Union,
    cast,
)

import agentops
import opentelemetry.trace as trace_api
from agentops.sdk.decorators import operation
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import get_tracer_provider

from agentlightning.types import Span, SpanNames

logger = logging.getLogger(__name__)


class RewardSpanData(TypedDict):
    type: Literal["reward"]
    value: Optional[float]


FnType = TypeVar("FnType", bound=Callable[..., Any])


def _agentops_initialized() -> bool:
    """Check if AgentOps is initialized in the current context."""
    return agentops.get_client().initialized


def reward(fn: FnType) -> FnType:
    """
    A decorator to wrap a function that computes rewards.
    It will automatically handle the input and output of the function.
    """

    def wrap_result(result: Optional[float]) -> RewardSpanData:
        """
        Wrap the result of the function in a dict.
        """
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


def emit_reward(reward: float) -> ReadableSpan:
    """
    Record a new reward as a new span.
    """
    logger.debug(f"Emitting reward: {reward}")
    if isinstance(reward, (int, bool)):
        reward = float(reward)
    if not isinstance(reward, float):
        raise ValueError(f"Reward must be a number, got: {type(reward)}")

    # Check for tracer initialization
    if hasattr(trace_api, "_TRACER_PROVIDER") and trace_api._TRACER_PROVIDER is None:  # type: ignore
        raise RuntimeError("Tracer is not initialized. Cannot emit a meaningful span.")

    tracer_provider = get_tracer_provider()

    tracer = tracer_provider.get_tracer("agentlightning")
    span = tracer.start_span(SpanNames.REWARD.value, attributes={"reward": reward})
    # Do nothing; it's just a number
    with span:
        pass
    if not isinstance(span, ReadableSpan):
        raise ValueError(f"Span is not a ReadableSpan: {span}")
    return span


SpanLike = Union[ReadableSpan, Span]


def get_reward_value(span: SpanLike) -> Optional[float]:
    """
    Get the reward value from a span.
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
    """
    Check if a span is a reward span.
    """
    maybe_reward = get_reward_value(span)
    return maybe_reward is not None


def find_reward_spans(spans: Sequence[SpanLike]) -> List[SpanLike]:
    """
    Find all reward spans in the given list of spans.

    Args:
        spans: A list of spans (either ReadableSpan or Span).

    Returns:
        A list of spans whose name matches the reward span name.
    """
    return [span for span in spans if is_reward_span(span)]


def get_last_reward(spans: Sequence[SpanLike]) -> Optional[float]:
    """
    Get the last reward value from a list of spans.

    Args:
        spans: A list of spans (either ReadableSpan or Span).

    Returns:
        The reward value from the last reward span, or None if not found.
    """
    for span in reversed(spans):
        reward = get_reward_value(span)
        if reward is not None:
            return reward
    return None
