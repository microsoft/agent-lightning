# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting reward spans."""

import logging
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    TypedDict,
    cast,
)

from pydantic import TypeAdapter

from agentlightning.semconv import (
    AGL_ANNOTATION,
    AGL_OPERATION,
    AGL_REWARD,
    LightningSpanAttributes,
    RewardPydanticModel,
)
from agentlightning.types import SpanCoreFields, SpanLike
from agentlightning.utils.otel import filter_and_unflatten_attributes

from .annotation import emit_annotation

logger = logging.getLogger(__name__)

__all__ = [
    "emit_reward",
    "get_reward_value",
    "get_rewards_from_span",
    "is_reward_span",
    "find_reward_spans",
    "find_final_reward",
]


def _ensure_numeric_reward(reward: object) -> float:
    if isinstance(reward, float):
        return reward
    raise TypeError(f"Reward must be a float, got: {type(reward)}")


class RewardDimension(TypedDict):
    """Type representing a single dimension in a multi-dimensional reward."""

    name: str
    value: float


def emit_reward(
    reward: float | Dict[str, float],
    *,
    primary_key: str | None = None,
    attributes: Dict[str, Any] | None = None,
    propagate: bool = True,
) -> SpanCoreFields:
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
        reward: Floating point reward to record.
            Use a dictionary to represent a multi-dimensional reward.
        attributes: Other optional span attributes.
        propagate: Whether to propagate the span to exporters automatically.

    Returns:
        Span core fields capturing the recorded reward.
    """
    logger.debug(f"Emitting reward: {reward}")
    reward_dimensions: List[RewardDimension] = []
    if isinstance(reward, dict):
        reward_dict = {key: _ensure_numeric_reward(value) for key, value in reward.items()}
        if primary_key is None:
            raise ValueError("When emitting a multi-dimensional reward as a dict, primary_key must be provided.")
        if primary_key not in reward_dict:
            raise ValueError(f"Primary key '{primary_key}' not found in reward dict keys: {list(reward_dict.keys())}")
        reward_dimensions.append(RewardDimension(name=primary_key, value=reward_dict[primary_key]))
        for k, v in reward_dict.items():
            if k != primary_key:
                reward_dimensions.append(RewardDimension(name=k, value=v))
    else:
        reward = _ensure_numeric_reward(reward)
        reward_dimensions.append(RewardDimension(name="primary", value=reward))

    return emit_annotation(
        {LightningSpanAttributes.REWARD.value: reward_dimensions, **(attributes or {})}, propagate=propagate
    )


def get_reward_value(span: SpanLike) -> Optional[float]:
    """Extract the reward value from a span, if available.

    Args:
        span: Span object produced by Agent Lightning emitters.

    Returns:
        The primary reward encoded in the span or `None` when the span does not represent a reward.
    """
    if span.name == AGL_OPERATION and span.attributes:
        operation_name = span.attributes.get(LightningSpanAttributes.OPERATION_NAME.value)
        if operation_name != AGL_REWARD:
            return None
    elif span.name != AGL_ANNOTATION:
        return None
    reward_list = get_rewards_from_span(span)
    if reward_list:
        return reward_list[0].value
    return None


def get_rewards_from_span(span: SpanLike) -> List[RewardPydanticModel]:
    """Extract the reward as a list from a span, if available.

    Args:
        span: Span object produced by Agent Lightning emitters.

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
