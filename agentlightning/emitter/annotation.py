# Copyright (c) Microsoft. All rights reserved.

"""Helpers for emitting annotation spans."""

from typing import Any, Dict

import agentops
from agentops.sdk.decorators import operation
from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.semconv import LightningSpanAttributes
from agentlightning.types import SpanLike, SpanNames

from .utils import get_tracer


def emit_annotation(annotation: Dict[str, Any]) -> None:
    """Emit a new annotation span.

    Annotation spans are used to annotate a specific event or a part of rollout.
    See semconv for conventional annotation keys in Agent-lightning.

    Args:
        annotation: ...

    !!! note
        The payload must be JSON serializable. Non-serializable objects are ignored and
        an error is logged to aid debugging.
    """
    try:
        serialized = json.dumps(object)
    except (TypeError, ValueError):
        logger.error(f"Object must be JSON serializable, got: {type(object)}. Skip emit_object.")
        return

    tracer = get_tracer()
    span = tracer.start_span(
        SpanNames.OBJECT.value,
        attributes={SpanAttributeNames.OBJECT.value: serialized},
    )
    logger.debug("Emitting object span with payload size %d characters", len(serialized))
    with span:
        pass


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


def emit_reward(reward: float, *, metadata: Dict[str, Any] | None = None, auto_export: bool = True) -> ReadableSpan:
    """Emit a reward value as an OpenTelemetry span.

    Args:
        reward: Numeric reward to record. Integers and booleans are converted to
            floating point numbers for consistency.
        metadata: Optional additional metadata to attach to the span.
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
