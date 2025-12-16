# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any, Dict, List, cast

import opentelemetry.trace as trace_api
import pytest
from opentelemetry.sdk.trace import ReadableSpan, SynchronousMultiSpanProcessor
from opentelemetry.semconv.attributes import exception_attributes
from opentelemetry.trace import TraceFlags
from pydantic import ValidationError

from agentlightning.semconv import LightningSpanAttributes, LinkPydanticModel
from agentlightning.types.tracer import Span
from agentlightning.utils import otel
from agentlightning.utils.otel import (
    check_attributes_sanity,
    extract_links_from_attributes,
    extract_tags_from_attributes,
    filter_and_unflatten_attributes,
    filter_attributes,
    flatten_attributes,
    format_exception_attributes,
    full_qualified_name,
    get_tracer,
    get_tracer_provider,
    make_link_attributes,
    make_tag_attributes,
    query_linked_spans,
    sanitize_attribute_value,
    sanitize_attributes,
    sanitize_list_attribute_sanity,
    unflatten_attributes,
)

pytestmark = pytest.mark.utils


def _span_context(trace_id_hex: str, span_id_hex: str) -> trace_api.SpanContext:
    return trace_api.SpanContext(
        trace_id=int(trace_id_hex, 16),
        span_id=int(span_id_hex, 16),
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
        trace_state=trace_api.TraceState(),
    )


def test_flatten_simple_nested_dict_and_list() -> None:
    data = {"a": {"b": 1, "c": [2, 3]}}
    result = flatten_attributes(data)
    assert result == {
        "a.b": 1,
        "a.c": [2, 3],
    }


def test_flatten_simple_nested_dict_and_list_with_leaf_expansion() -> None:
    data = {"a": {"b": 1, "c": [2, 3]}}
    result = flatten_attributes(data, expand_leaf_lists=True)
    assert result == {
        "a.b": 1,
        "a.c.0": 2,
        "a.c.1": 3,
    }


def test_full_qualified_name_handles_builtin_and_custom_classes() -> None:
    class LocalClass:
        pass

    builtin_result = full_qualified_name(int)
    local_result = full_qualified_name(LocalClass)

    assert builtin_result == "int"
    assert local_result.endswith("LocalClass")
    assert local_result.startswith("tests.utils.test_otel")


def test_flatten_empty_dict() -> None:
    data: Dict[str, Any] = {}
    assert flatten_attributes(data) == {}


def test_flatten_empty_list() -> None:
    data: List[Any] = []
    # No elements -> no keys
    assert flatten_attributes(data) == {}


def test_flatten_root_list_of_primitives() -> None:
    data = [10, 20, 30]
    result = flatten_attributes(data)
    assert result == {
        "0": 10,
        "1": 20,
        "2": 30,
    }


def test_flatten_nested_lists_and_dicts() -> None:
    data: Dict[str, Any] = {
        "users": [
            {"name": "Alice", "tags": ["admin", "staff"]},
            {"name": "Bob", "tags": []},
        ]
    }
    result = flatten_attributes(data)
    assert result == {
        "users.0.name": "Alice",
        "users.0.tags": ["admin", "staff"],
        "users.1.name": "Bob",
        # Empty leaf lists remain implicit
    }


def test_flatten_leaf_lists_can_stay_compact() -> None:
    data = {"tags": ["fast", "reliable"]}
    result = flatten_attributes(data, expand_leaf_lists=False)
    assert result == {"tags": ["fast", "reliable"]}


def test_flatten_non_leaf_lists_still_expand_when_leaf_expansion_disabled() -> None:
    data = {"users": [{"name": "Alice"}, {"name": "Bob"}]}
    result = flatten_attributes(data, expand_leaf_lists=False)
    assert result == {
        "users.0.name": "Alice",
        "users.1.name": "Bob",
    }


def test_flatten_leaf_lists_with_mixed_types_warns_and_expands(caplog: pytest.LogCaptureFixture) -> None:
    data = {"values": [1, "two"]}
    caplog.set_level(logging.WARNING, logger="agentlightning.utils.otel")

    result = flatten_attributes(data, expand_leaf_lists=False)

    assert result == {
        "values.0": 1,
        "values.1": "two",
    }
    assert "mixed primitive types" in caplog.text


def test_flatten_leaf_lists_with_mixed_numbers_warns_and_expands(caplog: pytest.LogCaptureFixture) -> None:
    data = {"values": [1, 2.0, 3]}
    caplog.set_level(logging.WARNING, logger="agentlightning.utils.otel")

    result = flatten_attributes(data, expand_leaf_lists=False)

    assert result == {
        "values.0": 1,
        "values.1": 2.0,
        "values.2": 3,
    }
    assert "mixed primitive types" in caplog.text


def test_flatten_mixed_types_and_none() -> None:
    data = {
        "a": True,
        "b": None,
        "c": 3.14,
        "d": "hello",
        "e": {"f": False},
    }
    result = flatten_attributes(data)
    assert result == {
        "a": True,
        "b": None,
        "c": 3.14,
        "d": "hello",
        "e.f": False,
    }


def test_flatten_non_string_key_raises_value_error() -> None:
    data = {
        "a": {
            1: "bad",  # non-string key inside nested dict
        }
    }
    with pytest.raises(ValueError) as excinfo:
        flatten_attributes(data)

    msg = str(excinfo.value)
    assert "Only string keys are supported in dictionaries" in msg
    # Ensure the offending key is mentioned
    assert "'1'" in msg
    assert "type <class 'int'>" in msg


def test_flatten_root_primitive_is_allowed() -> None:
    # Even though the type hint says Dict/List, function behavior supports primitives.
    data = 42
    result = flatten_attributes(data)  # type: ignore[arg-type]
    assert result == {"": 42}


def test_unflatten_simple_nested_dict() -> None:
    flat = {
        "a.b": 1,
        "a.c": 2,
    }
    result = unflatten_attributes(flat)
    assert result == {"a": {"b": 1, "c": 2}}


def test_unflatten_consecutive_numeric_keys_to_list() -> None:
    flat = {
        "a.0": "x",
        "a.1": "y",
        "a.2": "z",
    }
    result = unflatten_attributes(flat)
    assert result == {
        "a": ["x", "y", "z"],
    }


def test_unflatten_non_consecutive_numeric_keys_stays_dict() -> None:
    flat = {
        "a.0": "first",
        "a.2": "third",
    }
    result = unflatten_attributes(flat)
    # Keys are numeric but not consecutive -> remains dict
    assert result == {
        "a": {
            "0": "first",
            "2": "third",
        }
    }


def test_unflatten_mixed_numeric_and_non_numeric_keys_stays_dict() -> None:
    flat = {
        "a.0": "zero",
        "a.foo": "bar",
    }
    result = unflatten_attributes(flat)
    assert result == {
        "a": {
            "0": "zero",
            "foo": "bar",
        }
    }


def test_unflatten_root_list_from_numeric_keys() -> None:
    flat = {
        "0": "a",
        "1": "b",
        "2": "c",
    }
    result = unflatten_attributes(flat)
    # Root dict with all numeric keys 0..n-1 becomes list
    assert result == ["a", "b", "c"]


def test_unflatten_empty_flat_dict_returns_empty_dict() -> None:
    flat: Dict[str, Any] = {}
    result = unflatten_attributes(flat)
    assert result == {}


def test_unflatten_nested_lists_and_dicts() -> None:
    flat = {
        "users.0.name": "Alice",
        "users.0.tags.0": "admin",
        "users.0.tags.1": "staff",
        "users.1.name": "Bob",
        "users.1.tags.0": "guest",
    }
    result = unflatten_attributes(flat)
    assert result == {
        "users": [
            {"name": "Alice", "tags": ["admin", "staff"]},
            {"name": "Bob", "tags": ["guest"]},
        ]
    }


def test_unflatten_list_of_lists() -> None:
    flat = {
        "a.0.0": 1,
        "a.0.1": 2,
        "a.1.0": 3,
    }
    result = unflatten_attributes(flat)
    assert result == {
        "a": [
            [1, 2],
            [3],
        ]
    }


def test_unflatten_conflicting_primitive_and_nested_path_prefers_nested() -> None:
    # "a" is first set to a primitive, then to a nested dict via "a.b"
    flat = {
        "a": 1,
        "a.b": 2,
    }
    result = unflatten_attributes(flat)
    # Primitive is overwritten by nested dict structure
    assert result == {"a": {"b": 2}}


@pytest.mark.parametrize(
    "value",
    [
        {"a": {"b": 1, "c": [2, 3]}},
        {"x": [1, 2, {"y": 3}]},
        {"root": [{"k": "v"}, {"k": "w"}]},
        [{"name": "Alice"}, {"name": "Bob", "scores": [10, 20]}],
    ],
)
def test_round_trip_flatten_then_unflatten_preserves_structure(value: Dict[str, Any] | List[Any]) -> None:
    flat = flatten_attributes(value)  # type: ignore[arg-type]
    reconstructed = unflatten_attributes(flat)
    assert reconstructed == value


@pytest.mark.parametrize(
    "flat",
    [
        {"a.b": 1, "a.c": 2},
        {"0": "x", "1": "y"},
        {
            "users.0.name": "Alice",
            "users.1.name": "Bob",
        },
    ],
)
def test_round_trip_unflatten_then_flatten_preserves_flat_structure(flat: Dict[str, Any]) -> None:
    nested = unflatten_attributes(flat)
    re_flat = flatten_attributes(nested)
    # Order of items in dict shouldn't matter
    assert re_flat == flat


def test_round_trip_with_empty_list_information_loss_is_expected() -> None:
    """This documents the corner case: empty list flattens to {},
    which unflattens back to {} (empty dict), losing the distinction.
    """
    data: List[Any] = []
    flat = flatten_attributes(data)
    assert flat == {}
    reconstructed = unflatten_attributes(flat)
    assert reconstructed == {}
    assert reconstructed != data  # explicit documentation of the behavior


def test_sanitize_attribute_value_handles_primitives_and_lists() -> None:
    assert sanitize_attribute_value("text") == "text"
    assert sanitize_attribute_value(42) == 42
    assert sanitize_attribute_value(3.14) == 3.14
    assert sanitize_attribute_value(True) is True
    assert sanitize_attribute_value([True, 2]) == [1, 2]


def test_sanitize_attribute_value_falls_back_to_json_for_hybrid_lists(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="agentlightning.utils.otel")
    result = sanitize_attribute_value([1, "two"])
    assert result == json.dumps([1, "two"])
    assert "Failed to sanitize list attribute" in caplog.text


def test_sanitize_attribute_value_serializes_non_serializable_objects_when_forced() -> None:
    class Unserializable:
        def __str__(self) -> str:
            return "forced-serialization"

    result = sanitize_attribute_value(Unserializable())
    assert json.loads(cast(str, result)) == "forced-serialization"


def test_sanitize_attribute_value_respects_force_flag() -> None:
    class Unserializable:
        pass

    with pytest.raises(ValueError, match="Object must be JSON serializable"):
        sanitize_attribute_value(Unserializable(), force=False)


def test_sanitize_attributes_returns_clean_values() -> None:
    attributes = {
        "name": "agent",
        "enabled": True,
        "scores": [1, True],
        "flags": [True, False],
    }

    result = sanitize_attributes(attributes)
    assert result["name"] == "agent"
    assert result["enabled"] is True
    assert result["scores"] == [1, 1]
    assert result["flags"] == [True, False]


def test_sanitize_attributes_serializes_non_serializable_values_by_default() -> None:
    class CustomValue:
        def __str__(self) -> str:
            return "custom-payload"

    attributes = {"payload": CustomValue()}

    result = sanitize_attributes(attributes)
    assert json.loads(cast(str, result["payload"])) == "custom-payload"


def test_sanitize_attributes_respects_force_flag() -> None:
    class CustomValue:
        pass

    attributes = {"payload": CustomValue()}

    with pytest.raises(ValueError, match="Failed to sanitize attribute 'payload'"):
        sanitize_attributes(attributes, force=False)


def test_sanitize_attributes_exposes_key_in_error() -> None:
    class Bad:
        pass

    attributes = {"ok": "value", "bad": Bad()}
    with pytest.raises(ValueError, match="Failed to sanitize attribute 'bad'"):
        sanitize_attributes(attributes, force=False)


def test_format_exception_attributes_captures_metadata() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as err:
        attrs = format_exception_attributes(err)

    assert attrs[exception_attributes.EXCEPTION_TYPE] == "RuntimeError"
    assert attrs[exception_attributes.EXCEPTION_MESSAGE] == "boom"
    assert attrs[exception_attributes.EXCEPTION_ESCAPED] is True
    assert exception_attributes.EXCEPTION_STACKTRACE in attrs
    assert "RuntimeError: boom" in attrs[exception_attributes.EXCEPTION_STACKTRACE]  # type: ignore


def test_sanitize_list_attribute_sanity_supports_primitive_lists() -> None:
    assert sanitize_list_attribute_sanity(["a", "b"]) == ["a", "b"]
    assert sanitize_list_attribute_sanity([True, False]) == [True, False]
    assert sanitize_list_attribute_sanity([1, False]) == [1, 0]
    assert sanitize_list_attribute_sanity([1.0, 2, True]) == [1.0, 2.0, 1.0]


def test_sanitize_list_attribute_sanity_rejects_mixed_types() -> None:
    with pytest.raises(ValueError, match="List must contain only one type of primitive values"):
        sanitize_list_attribute_sanity([1, "two"])


def test_check_attributes_sanity_accepts_valid_payload() -> None:
    attributes: Dict[str, Any] = {
        "name": "agent",
        "count": 3,
        "ratio": 0.5,
        "enabled": False,
        "flags": [True, False],
        "scores": [1, True],
    }

    check_attributes_sanity(attributes)


def test_check_attributes_sanity_requires_string_keys() -> None:
    with pytest.raises(ValueError, match="Attribute key must be a string"):
        check_attributes_sanity({1: "value"})  # type: ignore[arg-type]


def test_check_attributes_sanity_wraps_list_errors() -> None:
    with pytest.raises(ValueError, match="Failed to sanitize list attribute 'mixed'"):
        check_attributes_sanity({"mixed": [1, "two"]})


def test_check_attributes_sanity_rejects_non_primitive_values() -> None:
    with pytest.raises(ValueError, match="Attribute value must be a string"):
        check_attributes_sanity({"bad": {"nested": "value"}})


def test_make_and_extract_link_attributes_round_trip() -> None:
    flattened = make_link_attributes(
        {
            "gen_ai.response.id": "response-123",
            "span_id": "abcd1234abcd1234",
        }
    )
    assert flattened == {
        f"{LightningSpanAttributes.LINK.value}.0.key_match": "gen_ai.response.id",
        f"{LightningSpanAttributes.LINK.value}.0.value_match": "response-123",
        f"{LightningSpanAttributes.LINK.value}.1.key_match": "span_id",
        f"{LightningSpanAttributes.LINK.value}.1.value_match": "abcd1234abcd1234",
    }

    extracted = extract_links_from_attributes(flattened)
    assert [link.model_dump() for link in extracted] == [
        {"key_match": "gen_ai.response.id", "value_match": "response-123"},
        {"key_match": "span_id", "value_match": "abcd1234abcd1234"},
    ]


def test_make_link_attributes_rejects_non_string_values() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_link_attributes({"span_id": 123})  # type: ignore

    assert "Link value must be a string" in str(excinfo.value)


def test_make_tag_attributes_and_extract_round_trip() -> None:
    flattened = make_tag_attributes(["fast", "reliable"])
    assert flattened == {
        f"{LightningSpanAttributes.TAG.value}.0": "fast",
        f"{LightningSpanAttributes.TAG.value}.1": "reliable",
    }

    assert extract_tags_from_attributes(flattened) == ["fast", "reliable"]


def test_extract_tags_from_attributes_rejects_non_strings() -> None:
    attributes = {
        f"{LightningSpanAttributes.TAG.value}.0": 1,
    }

    with pytest.raises(ValidationError):
        extract_tags_from_attributes(attributes)


def test_filter_attributes_keeps_exact_matches_and_children() -> None:
    attributes = {
        "agentlightning.link": "root",
        "agentlightning.link.0.key_match": "trace_id",
        "agentlightning.other": "discard",
        "agentlightning.link_extra": "different_prefix",
    }

    filtered = filter_attributes(attributes, LightningSpanAttributes.LINK.value)
    assert filtered == {
        "agentlightning.link": "root",
        "agentlightning.link.0.key_match": "trace_id",
    }


def test_filter_and_unflatten_attributes_strips_prefix_and_rebuilds_nested_structure() -> None:
    attributes = {
        f"{LightningSpanAttributes.LINK.value}.0.key_match": "trace_id",
        f"{LightningSpanAttributes.LINK.value}.0.value_match": "aaa",
        f"{LightningSpanAttributes.LINK.value}.1.key_match": "span_id",
        f"{LightningSpanAttributes.LINK.value}.1.value_match": "bbb",
    }

    result = filter_and_unflatten_attributes(attributes, LightningSpanAttributes.LINK.value)
    assert result == [
        {"key_match": "trace_id", "value_match": "aaa"},
        {"key_match": "span_id", "value_match": "bbb"},
    ]


def test_filter_and_unflatten_attributes_rejects_exact_prefix_key() -> None:
    attributes = {LightningSpanAttributes.LINK.value: "invalid"}

    with pytest.raises(ValueError):
        filter_and_unflatten_attributes(attributes, LightningSpanAttributes.LINK.value)


def test_query_linked_spans_matches_trace_id_on_readable_span() -> None:
    readable_span = ReadableSpan(
        name="upstream",
        context=_span_context("a" * 32, "b" * 16),
        attributes={},
    )

    assert readable_span.context is not None
    links = [
        LinkPydanticModel(key_match="trace_id", value_match=trace_api.format_trace_id(readable_span.context.trace_id)),
    ]

    matches = query_linked_spans([readable_span], links)
    assert matches == [readable_span]


def test_query_linked_spans_matches_custom_span_attributes() -> None:
    custom_span = Span.from_attributes(
        attributes={"gen_ai.response.id": "response-123", "custom": "needle"},
        trace_id="c" * 32,
        span_id="d" * 16,
    )

    links = [
        LinkPydanticModel(key_match="gen_ai.response.id", value_match="response-123"),
        LinkPydanticModel(key_match="custom", value_match="needle"),
    ]

    matches = query_linked_spans([custom_span], links)
    assert matches == [custom_span]


def test_query_linked_spans_excludes_span_with_mismatched_span_id() -> None:
    span = Span.from_attributes(
        attributes={"marker": "x"},
        trace_id="e" * 32,
        span_id="f" * 16,
    )

    links = [
        LinkPydanticModel(key_match="span_id", value_match="deadbeefdeadbeef"),
        LinkPydanticModel(key_match="marker", value_match="x"),
    ]

    assert query_linked_spans([span], links) == []


def test_query_linked_spans_requires_all_links_to_match() -> None:
    span = Span.from_attributes(
        attributes={"marker": "x", "other": "y"},
        trace_id="1" * 32,
        span_id="2" * 16,
    )

    links = [
        LinkPydanticModel(key_match="marker", value_match="x"),
        LinkPydanticModel(key_match="other", value_match="z"),
    ]

    assert query_linked_spans([span], links) == []


def test_query_linked_spans_handles_readable_span_without_context() -> None:
    readable_span = ReadableSpan(name="orphan", context=None, attributes={"marker": "x"})

    links = [LinkPydanticModel(key_match="marker", value_match="x")]

    matches = query_linked_spans([readable_span], links)
    assert matches == [readable_span]


def test_get_tracer_provider_raises_when_tracer_uninitialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", None, raising=False)

    with pytest.raises(RuntimeError):
        get_tracer_provider(inspect=False)


def test_get_tracer_provider_logs_when_provider_not_sdk(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel_provider = object()
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", sentinel_provider, raising=False)
    monkeypatch.setattr(otel, "otel_get_tracer_provider", lambda: sentinel_provider)

    caplog.set_level(logging.ERROR, logger=otel.logger.name)

    returned = get_tracer_provider(inspect=False)

    assert returned is sentinel_provider
    assert any("Tracer provider is expected" in rec.getMessage() for rec in caplog.records)


def test_get_tracer_delegates_to_active_span_processor(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyProvider:
        def __init__(self) -> None:
            self.calls: List[str] = []

        def get_tracer(self, name: str) -> str:
            self.calls.append(name)
            return f"tracer:{name}"

    provider = DummyProvider()

    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", object(), raising=False)
    monkeypatch.setattr(otel, "get_tracer_provider", lambda inspect=True: provider)

    tracer = get_tracer()

    assert tracer == "tracer:agentlightning"
    assert provider.calls == ["agentlightning"]


def test_get_tracer_without_active_span_processor_builds_isolated_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SimpleNamespace(
        sampler="sampler",
        resource="resource",
        id_generator="id_gen",
    )

    created: Dict[str, Any] = {}

    class DummyTracer:
        def __init__(
            self,
            sampler: Any,
            resource: Any,
            span_processor: SynchronousMultiSpanProcessor,
            id_generator: Any,
            instrumentation_info: Any,
            span_limits: Any,
            instrumentation_scope: Any,
        ) -> None:
            created["args"] = (
                sampler,
                resource,
                span_processor,
                id_generator,
                instrumentation_info,
                span_limits,
                instrumentation_scope,
            )

    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", object(), raising=False)
    monkeypatch.setattr(otel, "get_tracer_provider", lambda inspect=True: provider)
    monkeypatch.setattr(otel, "Tracer", DummyTracer)

    tracer = get_tracer(use_active_span_processor=False)

    assert isinstance(created.get("args"), tuple)
    assert created["args"][0] == "sampler"
    assert created["args"][1] == "resource"
    assert isinstance(created["args"][2], SynchronousMultiSpanProcessor)
    assert created["args"][3] == "id_gen"
    assert created["args"][4].name == "agentlightning"
    assert tracer is not None


def test_get_tracer_raises_when_provider_not_initialized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trace_api, "_TRACER_PROVIDER", None, raising=False)

    with pytest.raises(RuntimeError):
        get_tracer()
