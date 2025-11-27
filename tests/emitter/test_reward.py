# Copyright (c) Microsoft. All rights reserved.

import importlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import pytest

reward_module = importlib.import_module("agentlightning.emitter.reward")
from agentlightning.emitter.reward import emit_reward, get_rewards_from_span
from agentlightning.reward import find_final_reward, find_reward_spans, get_reward_value, is_reward_span
from agentlightning.semconv import AGL_ANNOTATION, LightningSpanAttributes, RewardPydanticModel
from agentlightning.types import SpanLike
from agentlightning.utils.otel import make_link_attributes, make_tag_attributes


@dataclass
class FakeSpan:
    name: str
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class AttributeSpan:
    attributes: Optional[Dict[str, Any]]


def make_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> SpanLike:
    return cast(SpanLike, FakeSpan(name=name, attributes=attributes))


def _capture_emit_annotation(monkeypatch: pytest.MonkeyPatch) -> tuple[Dict[str, Any], object]:
    captured: Dict[str, Any] = {}
    sentinel = object()

    def fake_emit_annotation(payload: Dict[str, Any], *, propagate: bool) -> object:
        captured["payload"] = payload
        captured["propagate"] = propagate
        return sentinel

    monkeypatch.setattr(reward_module, "emit_annotation", fake_emit_annotation)
    return captured, sentinel


def test_emit_reward_example_scalar(monkeypatch: pytest.MonkeyPatch) -> None:
    captured, sentinel = _capture_emit_annotation(monkeypatch)

    result = emit_reward(1.0)

    assert result is sentinel
    assert captured["propagate"] is True
    dimensions = captured["payload"][LightningSpanAttributes.REWARD.value]
    assert dimensions == [{"name": "primary", "value": 1.0}]


def test_emit_reward_example_multi_dimensional(monkeypatch: pytest.MonkeyPatch) -> None:
    captured, _ = _capture_emit_annotation(monkeypatch)

    emit_reward({"task_completion": 1.0, "efficiency": 0.8}, primary_key="task_completion")

    dimensions = captured["payload"][LightningSpanAttributes.REWARD.value]
    assert dimensions == [
        {"name": "task_completion", "value": 1.0},
        {"name": "efficiency", "value": 0.8},
    ]


def test_emit_reward_example_with_links(monkeypatch: pytest.MonkeyPatch) -> None:
    captured, _ = _capture_emit_annotation(monkeypatch)

    link_attrs = make_link_attributes({"gen_ai.response.id": "response-123", "span_id": "abcd-efgh"})
    emit_reward(0.5, attributes=link_attrs)

    payload = captured["payload"]
    assert payload[f"{LightningSpanAttributes.LINK.value}.0.key_match"] == "gen_ai.response.id"
    assert payload[f"{LightningSpanAttributes.LINK.value}.0.value_match"] == "response-123"
    reward_dimensions = payload[LightningSpanAttributes.REWARD.value]
    assert reward_dimensions == [{"name": "primary", "value": 0.5}]


def test_emit_reward_example_with_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    captured, _ = _capture_emit_annotation(monkeypatch)

    tag_attrs = make_tag_attributes(["fast", "reliable"])
    emit_reward(0.7, attributes=tag_attrs)

    payload = captured["payload"]
    assert payload[f"{LightningSpanAttributes.TAG.value}.0"] == "fast"
    assert payload[f"{LightningSpanAttributes.TAG.value}.1"] == "reliable"
    reward_dimensions = payload[LightningSpanAttributes.REWARD.value]
    assert reward_dimensions == [{"name": "primary", "value": 0.7}]


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
        name=AGL_ANNOTATION,
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
        name=AGL_ANNOTATION,
        attributes={"reward": 2.0},
    )
    non_reward_span = make_span(name="other", attributes={})

    spans = find_reward_spans([non_reward_span, reward_span, non_reward_span])

    assert spans == [reward_span]


def test_find_final_reward_returns_last_reward_value() -> None:
    spans = [
        make_span(name="first", attributes={}),
        make_span(name=AGL_ANNOTATION, attributes={"reward": 1.0}),
        make_span(name="agentops", attributes={"agentops.task.output": {"type": "reward", "value": 5.5}}),
    ]

    assert find_final_reward(spans) == 5.5


def test_find_final_reward_returns_none_when_no_reward() -> None:
    spans = [
        make_span(name="first", attributes={}),
        make_span(name="second", attributes={"agentops.task.output": {"foo": "bar"}}),
    ]

    assert find_final_reward(spans) is None


def test_emit_reward_scalar_converts_to_primary_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}
    sentinel_span = object()

    def fake_emit_annotation(payload: Dict[str, Any], *, propagate: bool) -> object:
        captured["payload"] = payload
        captured["propagate"] = propagate
        return sentinel_span

    monkeypatch.setattr(reward_module, "emit_annotation", fake_emit_annotation)

    result = emit_reward(2, attributes={"extra": "value"}, propagate=False)

    assert result is sentinel_span
    assert captured["propagate"] is False
    rewards = captured["payload"][LightningSpanAttributes.REWARD.value]
    assert rewards == [{"name": "primary", "value": 2.0}]
    assert captured["payload"]["extra"] == "value"


def test_emit_reward_dict_requires_primary_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Dict[str, Any] = {}

    def fake_emit_annotation(payload: Dict[str, Any], *, propagate: bool) -> Dict[str, Any]:
        captured["payload"] = payload
        return payload

    monkeypatch.setattr(reward_module, "emit_annotation", fake_emit_annotation)

    emit_reward({"score": 0.8, "other": 0.2}, primary_key="score")

    rewards = captured["payload"][LightningSpanAttributes.REWARD.value]
    assert [dim["name"] for dim in rewards] == ["score", "other"]

    with pytest.raises(ValueError):
        emit_reward({"score": 0.8}, primary_key=None)

    with pytest.raises(ValueError):
        emit_reward({"score": 0.8}, primary_key="missing")

    with pytest.raises(ValueError):
        emit_reward({"score": "bad"}, primary_key="score")


def test_emit_reward_rejects_non_numeric() -> None:
    with pytest.raises(TypeError):
        emit_reward("bad")  # type: ignore[arg-type]


def test_get_rewards_from_span_roundtrip() -> None:
    attributes = {
        f"{LightningSpanAttributes.REWARD.value}.0.name": "primary",
        f"{LightningSpanAttributes.REWARD.value}.0.value": 1.0,
        f"{LightningSpanAttributes.REWARD.value}.1.name": "aux",
        f"{LightningSpanAttributes.REWARD.value}.1.value": 0.25,
    }
    span = AttributeSpan(attributes=attributes)

    rewards = get_rewards_from_span(cast(SpanLike, span))

    assert rewards == [
        RewardPydanticModel(name="primary", value=1.0),
        RewardPydanticModel(name="aux", value=0.25),
    ]


def test_get_rewards_from_span_returns_empty_when_missing() -> None:
    span = AttributeSpan(attributes={})

    assert get_rewards_from_span(cast(SpanLike, span)) == []
