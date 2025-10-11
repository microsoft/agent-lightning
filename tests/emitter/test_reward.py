# Copyright (c) Microsoft. All rights reserved.

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

from agentlightning.reward import (
    find_final_reward,
    find_reward_spans,
    get_reward_value,
    is_reward_span,
)
from agentlightning.types import SpanLike, SpanNames


@dataclass
class FakeSpan:
    name: str
    attributes: Optional[Dict[str, Any]] = None


def make_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanLike:
    return cast(SpanLike, FakeSpan(name=name, attributes=attributes))


def test_get_reward_value_from_agentops_dict() -> None:
    span = make_span(
        name="any",
        attributes={
            "agentops.task.output": {"type": "reward", "value": 3.5},
        },
    )

    assert get_reward_value(span) == 3.5


def test_get_reward_value_from_agentops_json_string() -> None:
    payload = json.dumps({"type": "reward", "value": 1.25})
    span = make_span(name="any", attributes={"agentops.entity.output": payload})

    assert get_reward_value(span) == 1.25


def test_get_reward_value_from_reward_span_attributes() -> None:
    span = make_span(
        name=SpanNames.REWARD.value,
        attributes={"reward": 0.75},
    )

    assert get_reward_value(span) == 0.75


def test_get_reward_value_returns_none_when_not_reward() -> None:
    span = make_span(name="any", attributes={"agentops.task.output": {"foo": "bar"}})

    assert get_reward_value(span) is None


def test_is_reward_span_matches_reward_value() -> None:
    span = make_span(
        name="whatever",
        attributes={"agentops.task.output": {"type": "reward", "value": 4.2}},
    )

    assert is_reward_span(span) is True


def test_is_reward_span_false_when_no_reward() -> None:
    span = make_span(name="absent", attributes={"agentops.entity.output": {"value": 1}})

    assert is_reward_span(span) is False


def test_find_reward_spans_filters_correctly() -> None:
    reward_span = make_span(
        name=SpanNames.REWARD.value,
        attributes={"reward": 2.0},
    )
    non_reward_span = make_span(name="other", attributes={})

    spans = find_reward_spans([non_reward_span, reward_span, non_reward_span])

    assert spans == [reward_span]


def test_find_final_reward_returns_last_reward_value() -> None:
    spans = [
        make_span(name="first", attributes={}),
        make_span(name=SpanNames.REWARD.value, attributes={"reward": 1.0}),
        make_span(name="agentops", attributes={"agentops.task.output": {"type": "reward", "value": 5.5}}),
    ]

    assert find_final_reward(spans) == 5.5


def test_find_final_reward_returns_none_when_no_reward() -> None:
    spans = [
        make_span(name="first", attributes={}),
        make_span(name="second", attributes={"agentops.task.output": {"foo": "bar"}}),
    ]

    assert find_final_reward(spans) is None
