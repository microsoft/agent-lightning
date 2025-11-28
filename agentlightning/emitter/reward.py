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
from pydantic import TypeAdapter

from agentlightning.semconv import AGL_ANNOTATION, LightningSpanAttributes, RewardPydanticModel
from agentlightning.types import SpanLike
from agentlightning.utils.otel import filter_and_unflatten_attributes

from .annotation import emit_annotation

logger = logging.getLogger(__name__)

__all__ = [
    "reward",
    "emit_reward",
    "get_reward_value",
    "get_rewards_from_span",
    "is_reward_span",
    "find_reward_spans",
    "find_final_reward",
]


class RewardDimension(TypedDict):
    """Type representing a single dimension in a multi-dimensional reward."""

    name: str
    value: float


class _RewardSpanData(TypedDict):
    type: Literal["reward"]
    value: Optional[float]


_FnType = TypeVar("_FnType", bound=Callable[..., Any])


def _agentops_initialized() -> bool:
    """Return `True` when the AgentOps client has been configured."""
    return agentops.get_client().initialized


def reward(fn: _FnType) -> _FnType:
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

    def wrap_result(result: Optional[float]) -> _RewardSpanData:
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
            async def agentops_reward_operation() -> _RewardSpanData:
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
            def agentops_reward_operation() -> _RewardSpanData:
                nonlocal result
                result = fn(*args, **kwargs)
                return wrap_result(result)

            agentops_reward_operation()
            return result

        return wrapper  # type: ignore


def emit_reward(
    reward: float | Dict[str, Any],
    *,
    primary_key: str | None = None,
    attributes: Dict[str, Any] | None = None,
    propagate: bool = True,
) -> ReadableSpan:
    """Emit a reward value as an OpenTelemetry span.

    Examples:
        Emit a single-dimensional reward:
        >>> emit_reward(1.0)

        Emit multi-dimensional rewards:
        >>> emit_reward({"task_completion": 1.0, "efficiency": 0.8}, primary_key="task_completion")

        Emit a reward with additional attributes (for example linking to another response span):
        >>> from agentlightning.utils.otel import make_link_attributes
        >>> emit_reward(0.5, attributes=make_link_attributes({"gen_ai.response.id": "response-123"}))

        Or adding tags onto the reward span:
        >>> from agentlightning.utils.otel import make_tag_attributes
        >>> emit_reward(0.7, attributes=make_tag_attributes(["fast", "reliable"]))

    Args:
        reward: Numeric reward to record. Integers and booleans are converted to
            floating point numbers for consistency.
            Use a dictionary to represent a multi-dimensional reward.
        attributes: Other optional span attributes.
        propagate: Whether to propagate the span to exporters automatically.

    Returns:
        Readable span capturing the recorded reward.

    Raises:
        ValueError: If the provided reward cannot be interpreted as a float or the
            resulting span is not a [`ReadableSpan`](https://opentelemetry.io/docs/concepts/signals/traces/) instance.
    """
    logger.debug(f"Emitting reward: {reward}")
    reward_dimensions: List[RewardDimension] = []
    if isinstance(reward, dict):
        reward_dict: Dict[str, float] = {}
        for k, v in reward.items():
            if isinstance(v, (int, bool)):
                reward_dict[k] = float(v)
            elif isinstance(v, float):
                reward_dict[k] = v
            else:
                raise ValueError(f"Reward value must be a number, got: {type(v)} for key {k}")
        if primary_key is None:
            raise ValueError("When emitting a multi-dimensional reward as a dict, primary_key must be provided.")
        if primary_key not in reward_dict:
            raise ValueError(f"Primary key '{primary_key}' not found in reward dict keys: {list(reward_dict.keys())}")
        reward_dimensions.append(RewardDimension(name=primary_key, value=reward_dict[primary_key]))
        for k, v in reward_dict.items():
            if k != primary_key:
                reward_dimensions.append(RewardDimension(name=k, value=v))
    else:
        if isinstance(reward, (int, bool)):
            reward = float(reward)
        elif not isinstance(reward, float):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Reward must be a number, got: {type(reward)}")
        reward_dimensions.append(RewardDimension(name="primary", value=reward))

    return emit_annotation(
        {LightningSpanAttributes.REWARD.value: reward_dimensions, **(attributes or {})}, propagate=propagate
    )


def get_reward_value(span: SpanLike) -> Optional[float]:
    """Extract the reward value from a span, if available.

    Args:
        span: Span object produced by AgentOps or Agent Lightning emitters.

    Returns:
        The primary reward encoded in the span or `None` when the span does not represent a reward.
    """
    # v0.3+ emit reward format
    reward_list = get_rewards_from_span(span)
    if reward_list:
        # Reward list is ordered and the first element is the primary reward
        return reward_list[0].value

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
            logger.warning(
                f"Extracted reward {reward_value} from AgentOps. This format is deprecated, please migrate to using `emit_reward`."
            )
            return cast(float, reward_value)

    # v0.2 emit reward format
    if span.name == AGL_ANNOTATION and span.attributes:
        reward_value = span.attributes.get("reward", None)
        if reward_value is None:
            return None
        if not isinstance(reward_value, float):
            logger.error(f"Reward is not a number, got: {type(reward_value)}. This may cause undefined behaviors.")
        logger.warning(
            f"Extracted reward {reward_value} from a legacy version of reward span. You might have inconsistent agent-lightning versions."
        )
        return cast(float, reward_value)

    return None


def get_rewards_from_span(span: SpanLike) -> List[RewardPydanticModel]:
    """Extract the reward as a list from a span, if available.

    Args:
        span: Span object produced by AgentOps or Agent Lightning emitters.

    Returns:
        A list of reward dimensions encoded in the span or an empty list when the span does not represent a reward.
    """
    if span.attributes and any(key.startswith(LightningSpanAttributes.REWARD.value) for key in span.attributes):
        reward_attr = filter_and_unflatten_attributes(
            cast(Any, span.attributes or {}), LightningSpanAttributes.REWARD.value
        )
        recovered_rewards = TypeAdapter(List[RewardPydanticModel]).validate_python(reward_attr)
        return recovered_rewards
    else:
        return []


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
