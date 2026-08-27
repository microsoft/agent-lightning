# Copyright (c) Microsoft. All rights reserved.

"""Tests for the annotation module adapters."""

import itertools
from typing import Any, Dict, Optional

import pytest

from agentlightning.adapter.annotation import (
    IdentifyAnnotations,
    RepairMissingLinks,
    SelectByAnnotation,
)
from agentlightning.semconv import (
    AGL_ANNOTATION,
    AGL_EXCEPTION,
    AGL_MESSAGE,
    AGL_OBJECT,
    AGL_OPERATION,
    LightningSpanAttributes,
    LinkPydanticModel,
)
from agentlightning.types import Span
from agentlightning.types.adapter import (
    AdaptingSequence,
    AdaptingSpan,
    AgentAnnotation,
    ExceptionAnnotation,
    GeneralAnnotation,
    MessageAnnotation,
    ObjectAnnotation,
    OperationAnnotation,
    Tree,
)

_SEQ = itertools.count()


def make_span(
    span_id: str,
    name: str,
    *,
    parent_id: Optional[str],
    start_time: float,
    end_time: float,
    attributes: Optional[Dict[str, Any]] = None,
    rollout_id: str = "rollout-1",
    attempt_id: str = "attempt-1",
    sequence_id: Optional[int] = None,
) -> Span:
    """Create a test span with sensible defaults."""
    return Span.from_attributes(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
        sequence_id=sequence_id if sequence_id is not None else next(_SEQ),
        trace_id="trace-1",
        span_id=span_id,
        parent_id=parent_id,
        name=name,
        attributes=attributes or {},
        start_time=start_time,
        end_time=end_time,
    )


def make_adapting_span(
    span_id: str,
    name: str,
    *,
    parent_id: Optional[str] = None,
    start_time: float = 0.0,
    end_time: float = 1.0,
    attributes: Optional[Dict[str, Any]] = None,
    data: Any = None,
) -> AdaptingSpan:
    """Create a test AdaptingSpan with sensible defaults."""
    span = make_span(
        span_id=span_id,
        name=name,
        parent_id=parent_id,
        start_time=start_time,
        end_time=end_time,
        attributes=attributes,
    )
    return AdaptingSpan.from_span(span, data)


# Tests for IdentifyAnnotations._filter_custom_attributes


def test_filter_custom_attributes_filters_reserved_fields() -> None:
    """Reserved fields should be filtered out."""
    adapter = IdentifyAnnotations()
    attributes = {
        LightningSpanAttributes.REWARD.value: 1.0,
        LightningSpanAttributes.TAG.value: ["tag1"],
        "custom.field": "value",
    }

    result = adapter._filter_custom_attributes(attributes)  # pyright: ignore[reportPrivateUsage]

    assert "custom.field" in result
    assert LightningSpanAttributes.REWARD.value not in result
    assert LightningSpanAttributes.TAG.value not in result


def test_filter_custom_attributes_filters_nested_reserved_fields() -> None:
    """Nested reserved fields are only filtered when base field is present."""
    adapter = IdentifyAnnotations()
    # When the base field IS present, nested fields are also filtered
    attributes = {
        LightningSpanAttributes.REWARD.value: "base",
        f"{LightningSpanAttributes.REWARD.value}.0.value": 1.0,
        f"{LightningSpanAttributes.REWARD.value}.0.name": "quality",
        "custom.field": "value",
    }

    result = adapter._filter_custom_attributes(attributes)  # pyright: ignore[reportPrivateUsage]

    assert "custom.field" in result
    assert len(result) == 1


def test_filter_custom_attributes_nested_fields_not_filtered_without_base() -> None:
    """Nested reserved fields pass through if base field is not present.

    Note: This is the current behavior - nested fields like 'agentlightning.reward.0.value'
    are NOT filtered unless the base field 'agentlightning.reward' is also present.
    """
    adapter = IdentifyAnnotations()
    attributes = {
        f"{LightningSpanAttributes.REWARD.value}.0.value": 1.0,
        "custom.field": "value",
    }

    result = adapter._filter_custom_attributes(attributes)  # pyright: ignore[reportPrivateUsage]

    # Nested fields pass through since base field is not present
    assert len(result) == 2


def test_filter_custom_attributes_preserves_custom_fields() -> None:
    """Custom fields not matching reserved prefixes should be preserved."""
    adapter = IdentifyAnnotations()
    attributes = {
        "my.custom.attribute": "value1",
        "another_attribute": "value2",
    }

    result = adapter._filter_custom_attributes(attributes)  # pyright: ignore[reportPrivateUsage]

    assert result == attributes


# Tests for IdentifyAnnotations.extract_links


def test_extract_links_extracts_valid_links() -> None:
    """Valid links should be extracted from span attributes."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "span",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            f"{LightningSpanAttributes.LINK.value}.0.key_match": "span_id",
            f"{LightningSpanAttributes.LINK.value}.0.value_match": "target-span",
        },
    )

    links = adapter.extract_links(span)

    assert len(links) == 1
    assert links[0].key_match == "span_id"
    assert links[0].value_match == "target-span"


def test_extract_links_returns_empty_for_no_links() -> None:
    """Should return empty list when no links are present."""
    adapter = IdentifyAnnotations()
    span = make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0)

    links = adapter.extract_links(span)

    assert links == []


def test_extract_links_handles_malformed_links() -> None:
    """Malformed links should return empty list without raising."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "span",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            f"{LightningSpanAttributes.LINK.value}.0.invalid_field": "value",
        },
    )

    # Should not raise, returns empty list
    links = adapter.extract_links(span)
    assert links == []


# Tests for IdentifyAnnotations.identify_general


def test_identify_general_creates_general_annotation() -> None:
    """Should create GeneralAnnotation from annotation span."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_ANNOTATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "custom.field": "value",
            # Need to provide empty list for tags since extract_tags_from_attributes
            # doesn't handle missing tags gracefully
            f"{LightningSpanAttributes.TAG.value}.0": "test-tag",
        },
    )

    result = adapter.identify_general(span)

    assert result is not None
    assert result.annotation_type == "general"
    # custom.field is preserved, tag is filtered out
    assert "custom.field" in result.custom_fields


def test_identify_general_extracts_rewards() -> None:
    """Should extract rewards from annotation span."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_ANNOTATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            f"{LightningSpanAttributes.REWARD.value}.0.name": "quality",
            f"{LightningSpanAttributes.REWARD.value}.0.value": 0.9,
            f"{LightningSpanAttributes.TAG.value}.0": "test-tag",
        },
    )

    result = adapter.identify_general(span)

    assert result is not None
    assert len(result.rewards) == 1
    assert result.rewards[0].name == "quality"
    assert result.rewards[0].value == 0.9
    assert result.primary_reward == 0.9


def test_identify_general_extracts_tags() -> None:
    """Should extract tags from annotation span."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_ANNOTATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            f"{LightningSpanAttributes.TAG.value}.0": "important",
            f"{LightningSpanAttributes.TAG.value}.1": "reviewed",
        },
    )

    result = adapter.identify_general(span)

    assert result is not None
    assert "important" in result.tags
    assert "reviewed" in result.tags


# Tests for IdentifyAnnotations.identify_message


def test_identify_message_creates_message_annotation() -> None:
    """Should create MessageAnnotation from message span."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_MESSAGE,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.MESSAGE_BODY.value: "Hello, world!",
        },
    )

    result = adapter.identify_message(span)

    assert result is not None
    assert result.annotation_type == "message"
    assert result.message == "Hello, world!"


def test_identify_message_returns_none_for_missing_body() -> None:
    """Should return None when message body is missing."""
    adapter = IdentifyAnnotations()
    span = make_span("s1", AGL_MESSAGE, parent_id=None, start_time=0.0, end_time=1.0)

    result = adapter.identify_message(span)

    assert result is None


# Tests for IdentifyAnnotations.identify_object


def test_identify_object_creates_object_annotation_from_json() -> None:
    """Should create ObjectAnnotation from JSON object."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OBJECT,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.OBJECT_JSON.value: '{"key": "value"}',
        },
    )

    result = adapter.identify_object(span)

    assert result is not None
    assert result.annotation_type == "object"
    assert result.object == {"key": "value"}


def test_identify_object_creates_object_annotation_from_literal() -> None:
    """Should create ObjectAnnotation from literal value."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OBJECT,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.OBJECT_LITERAL.value: "simple string",
        },
    )

    result = adapter.identify_object(span)

    assert result is not None
    assert result.annotation_type == "object"


def test_identify_object_handles_invalid_json() -> None:
    """Should handle invalid JSON gracefully."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OBJECT,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.OBJECT_JSON.value: "invalid json {",
        },
    )

    # Should not raise, returns None due to error handling
    result = adapter.identify_object(span)
    assert result is None


# Tests for IdentifyAnnotations.identify_exception


def test_identify_exception_creates_exception_annotation() -> None:
    """Should create ExceptionAnnotation from exception span."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_EXCEPTION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "exception.type": "ValueError",
            "exception.message": "Invalid input",
            "exception.stacktrace": "Traceback...",
        },
    )

    result = adapter.identify_exception(span)

    assert result is not None
    assert result.annotation_type == "exception"
    assert result.type == "ValueError"
    assert result.message == "Invalid input"
    assert result.stacktrace == "Traceback..."


def test_identify_exception_uses_defaults_for_missing_fields() -> None:
    """Should use default values when exception fields are missing."""
    adapter = IdentifyAnnotations()
    span = make_span("s1", AGL_EXCEPTION, parent_id=None, start_time=0.0, end_time=1.0)

    result = adapter.identify_exception(span)

    assert result is not None
    assert result.type == "UnknownException"
    assert result.message == ""


# Tests for IdentifyAnnotations.identify_operation


def test_identify_operation_creates_operation_annotation() -> None:
    """Should create OperationAnnotation from operation span."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OPERATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.OPERATION_NAME.value: "process_data",
            LightningSpanAttributes.OPERATION_INPUT.value: "input_value",
            LightningSpanAttributes.OPERATION_OUTPUT.value: "output_value",
        },
    )

    result = adapter.identify_operation(span)

    assert result is not None
    assert result.annotation_type == "operation"
    assert result.name == "process_data"
    assert result.input == "input_value"
    assert result.output == "output_value"


def test_identify_operation_extracts_nested_input_output() -> None:
    """Should extract nested input/output from flattened attributes."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OPERATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.OPERATION_NAME.value: "process_data",
            f"{LightningSpanAttributes.OPERATION_INPUT.value}.arg1": "value1",
            f"{LightningSpanAttributes.OPERATION_INPUT.value}.arg2": "value2",
        },
    )

    result = adapter.identify_operation(span)

    assert result is not None
    assert result.input == {"arg1": "value1", "arg2": "value2"}


# Tests for IdentifyAnnotations.detect_agent_annotation


def test_detect_agent_annotation_detects_otel_agent_span() -> None:
    """Should detect agent from OpenTelemetry agent spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "agent.run",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "agent.name": "MyAgent",
            "agent.id": "agent-123",
            "agent.description": "A helpful agent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.annotation_type == "agent"
    assert result.name == "MyAgent"
    assert result.id == "agent-123"
    assert result.description == "A helpful agent"


def test_detect_agent_annotation_detects_agentops_agent() -> None:
    """Should detect agent from AgentOps spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "some.operation",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "agentops.span.kind": "agent",
            "operation.name": "AgentOpsAgent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.name == "AgentOpsAgent"


def test_detect_agent_annotation_detects_autogen_agent() -> None:
    """Should detect agent from Autogen spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "autogen.task",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "recipient_agent_type": "AssistantAgent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.name == "AssistantAgent"


def test_detect_agent_annotation_detects_langgraph_agent() -> None:
    """Should detect agent from LangGraph spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "langgraph.node",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "langchain.chain.type": "ReActAgent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.name == "ReActAgent"


def test_detect_agent_annotation_detects_weave_agent() -> None:
    """Should detect agent from Weave spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "weave.call",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "type": "agent",
            "agentlightning.operation.input.name": "WeaveAgent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.name == "WeaveAgent"


def test_detect_agent_annotation_detects_langchain_weave_agent() -> None:
    """Should detect agent from LangChain + Weave spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "langchain.Chain.MyChain",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "lc_name": "LangChainAgent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.name == "LangChainAgent"


def test_detect_agent_annotation_detects_agent_framework() -> None:
    """Should detect agent from agent-framework spans."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "executor.run",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "executor.id": "ExecutorAgent",
        },
    )

    result = adapter.detect_agent_annotation(span)

    assert result is not None
    assert result.name == "ExecutorAgent"


def test_detect_agent_annotation_returns_none_for_non_agent_span() -> None:
    """Should return None for spans without agent indicators."""
    adapter = IdentifyAnnotations()
    span = make_span("s1", "some.operation", parent_id=None, start_time=0.0, end_time=1.0)

    result = adapter.detect_agent_annotation(span)

    assert result is None


# Tests for IdentifyAnnotations.adapt_one


def test_adapt_one_identifies_annotation_span() -> None:
    """Should identify AGL_ANNOTATION spans."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span(
        "s1",
        AGL_ANNOTATION,
        attributes={
            "custom": "value",
            f"{LightningSpanAttributes.TAG.value}.0": "test-tag",
        },
    )

    result = adapter.adapt_one(source)

    assert isinstance(result.data, GeneralAnnotation)


def test_adapt_one_identifies_message_span() -> None:
    """Should identify AGL_MESSAGE spans."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span(
        "s1",
        AGL_MESSAGE,
        attributes={LightningSpanAttributes.MESSAGE_BODY.value: "Hello"},
    )

    result = adapter.adapt_one(source)

    assert isinstance(result.data, MessageAnnotation)


def test_adapt_one_identifies_exception_span() -> None:
    """Should identify AGL_EXCEPTION spans."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span(
        "s1",
        AGL_EXCEPTION,
        attributes={"exception.type": "Error"},
    )

    result = adapter.adapt_one(source)

    assert isinstance(result.data, ExceptionAnnotation)


def test_adapt_one_identifies_operation_span() -> None:
    """Should identify AGL_OPERATION spans."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span(
        "s1",
        AGL_OPERATION,
        attributes={LightningSpanAttributes.OPERATION_NAME.value: "op"},
    )

    result = adapter.adapt_one(source)

    assert isinstance(result.data, OperationAnnotation)


def test_adapt_one_falls_back_to_agent_detection() -> None:
    """Should fall back to agent detection for unknown span names."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span(
        "s1",
        "agent.task",
        attributes={"agent.name": "MyAgent"},
    )

    result = adapter.adapt_one(source)

    assert isinstance(result.data, AgentAnnotation)


def test_adapt_one_returns_unchanged_for_unrecognized_span() -> None:
    """Should return unchanged span when no annotation is detected."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span("s1", "some.operation")

    result = adapter.adapt_one(source)

    assert result.data is None


# Tests for SelectByAnnotation


def test_select_by_annotation_include_mode_selects_annotated_spans() -> None:
    """Include mode should select only annotated spans and their links."""
    adapter = SelectByAnnotation(mode="include")

    # Create spans with one annotation
    annotation_span = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[LinkPydanticModel(key_match="span_id", value_match="linked")],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    linked_span = make_adapting_span("linked", "some.operation")
    unrelated_span = make_adapting_span("unrelated", "other.operation")

    source = AdaptingSequence([annotation_span, linked_span, unrelated_span])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    assert "annotation" in span_ids
    assert "linked" in span_ids
    assert "unrelated" not in span_ids


def test_select_by_annotation_exclude_mode_removes_annotated_spans() -> None:
    """Exclude mode should remove annotated spans and their links."""
    adapter = SelectByAnnotation(mode="exclude")

    annotation_span = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[LinkPydanticModel(key_match="span_id", value_match="linked")],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    linked_span = make_adapting_span("linked", "some.operation")
    unrelated_span = make_adapting_span("unrelated", "other.operation")

    source = AdaptingSequence([annotation_span, linked_span, unrelated_span])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    assert "annotation" not in span_ids
    assert "linked" not in span_ids
    assert "unrelated" in span_ids


def test_select_by_annotation_include_mode_with_no_annotations() -> None:
    """Include mode with no annotations should return empty result."""
    adapter = SelectByAnnotation(mode="include")

    span1 = make_adapting_span("s1", "operation")
    span2 = make_adapting_span("s2", "operation")

    source = AdaptingSequence([span1, span2])
    result = adapter.adapt(source)

    assert len(list(result)) == 0


def test_select_by_annotation_include_mode_with_empty_links() -> None:
    """Include mode with annotation having empty links should only include the annotation itself.

    An annotation with links=[] should apply only to itself, not to all spans.
    This tests that check_linked_span returning True for empty links doesn't
    cause all spans to be incorrectly selected.
    """
    adapter = SelectByAnnotation(mode="include")

    # Annotation with no links - should only include itself
    annotation_span = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],  # Empty links!
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    unrelated_span1 = make_adapting_span("unrelated1", "some.operation")
    unrelated_span2 = make_adapting_span("unrelated2", "other.operation")

    source = AdaptingSequence([annotation_span, unrelated_span1, unrelated_span2])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    # Only the annotation span should be selected, not the unrelated spans
    assert span_ids == ["annotation"]


def test_select_by_annotation_exclude_mode_with_empty_links() -> None:
    """Exclude mode with annotation having empty links should only exclude the annotation itself.

    An annotation with links=[] should apply only to itself, so only the annotation
    span should be excluded, not all spans.
    """
    adapter = SelectByAnnotation(mode="exclude")

    annotation_span = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],  # Empty links!
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    unrelated_span1 = make_adapting_span("unrelated1", "some.operation")
    unrelated_span2 = make_adapting_span("unrelated2", "other.operation")

    source = AdaptingSequence([annotation_span, unrelated_span1, unrelated_span2])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    # Only the annotation span should be excluded, unrelated spans should remain
    assert "annotation" not in span_ids
    assert "unrelated1" in span_ids
    assert "unrelated2" in span_ids


def test_select_by_annotation_exclude_mode_with_no_annotations() -> None:
    """Exclude mode with no annotations should return all spans."""
    adapter = SelectByAnnotation(mode="exclude")

    span1 = make_adapting_span("s1", "operation")
    span2 = make_adapting_span("s2", "operation")

    source = AdaptingSequence([span1, span2])
    result = adapter.adapt(source)

    assert len(list(result)) == 2


# Tests for RepairMissingLinks


def test_repair_missing_links_backward() -> None:
    """Should repair missing links by searching backward."""
    adapter = RepairMissingLinks(scan_direction="backward")

    # Create sequence: candidate -> annotation (without link)
    candidate = make_adapting_span("candidate", "some.operation", start_time=0.0, end_time=1.0)
    annotation = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],  # No links
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([candidate, annotation])
    result = adapter.adapt(source)

    # Find the annotation span in result
    annotation_result = next(s for s in result if s.span_id == "annotation")
    assert isinstance(annotation_result.data, GeneralAnnotation)
    assert len(annotation_result.data.links) == 1
    assert annotation_result.data.links[0].value_match == "candidate"


def test_repair_missing_links_forward() -> None:
    """Should repair missing links by searching forward."""
    adapter = RepairMissingLinks(scan_direction="forward")

    # Create sequence: annotation (without link) -> candidate
    annotation = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        start_time=0.0,
        end_time=1.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    candidate = make_adapting_span("candidate", "some.operation", start_time=1.0, end_time=2.0)

    source = AdaptingSequence([annotation, candidate])
    result = adapter.adapt(source)

    annotation_result = next(s for s in result if s.span_id == "annotation")
    assert isinstance(annotation_result.data, GeneralAnnotation)
    assert len(annotation_result.data.links) == 1
    assert annotation_result.data.links[0].value_match == "candidate"


def test_repair_missing_links_respects_candidate_predicate() -> None:
    """Should only consider candidates matching the predicate."""
    adapter = RepairMissingLinks(
        scan_direction="backward",
        candidate_predicate=lambda span: span.name == "valid.candidate",
    )

    invalid_candidate = make_adapting_span("invalid", "invalid.operation", start_time=0.0, end_time=1.0)
    valid_candidate = make_adapting_span("valid", "valid.candidate", start_time=1.0, end_time=2.0)
    annotation = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        start_time=2.0,
        end_time=3.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([invalid_candidate, valid_candidate, annotation])
    result = adapter.adapt(source)

    annotation_result = next(s for s in result if s.span_id == "annotation")
    assert annotation_result.data.links[0].value_match == "valid"


def test_repair_missing_links_does_not_reuse_linked_spans_by_default() -> None:
    """By default, linked spans should not be reused for other annotations."""
    adapter = RepairMissingLinks(scan_direction="backward", allow_reuse_linked_spans=False)

    candidate = make_adapting_span("candidate", "operation", start_time=0.0, end_time=1.0)
    annotation1 = make_adapting_span(
        "ann1",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    annotation2 = make_adapting_span(
        "ann2",
        AGL_ANNOTATION,
        start_time=2.0,
        end_time=3.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([candidate, annotation1, annotation2])
    result = adapter.adapt(source)

    # Only one annotation should get the link
    linked_annotations = [s for s in result if isinstance(s.data, GeneralAnnotation) and len(s.data.links) > 0]
    assert len(linked_annotations) == 1


def test_repair_missing_links_allows_reuse_linked_spans_when_enabled() -> None:
    """When enabled, linked spans can be reused for multiple annotations."""
    adapter = RepairMissingLinks(scan_direction="backward", allow_reuse_linked_spans=True)

    candidate = make_adapting_span("candidate", "operation", start_time=0.0, end_time=1.0)
    annotation1 = make_adapting_span(
        "ann1",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    annotation2 = make_adapting_span(
        "ann2",
        AGL_ANNOTATION,
        start_time=2.0,
        end_time=3.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([candidate, annotation1, annotation2])
    result = adapter.adapt(source)

    # Both annotations should get links to the same candidate
    linked_annotations = [s for s in result if isinstance(s.data, GeneralAnnotation) and len(s.data.links) > 0]
    assert len(linked_annotations) == 2


def test_repair_missing_links_preserves_existing_links() -> None:
    """Annotations with existing links should not be modified."""
    adapter = RepairMissingLinks(scan_direction="backward")

    candidate = make_adapting_span("candidate", "operation", start_time=0.0, end_time=1.0)
    annotation_with_link = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[LinkPydanticModel(key_match="span_id", value_match="existing-target")],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([candidate, annotation_with_link])
    result = adapter.adapt(source)

    annotation_result = next(s for s in result if s.span_id == "annotation")
    # Original link should be preserved
    assert annotation_result.data.links[0].value_match == "existing-target"


def test_repair_missing_links_siblings_scope_requires_tree() -> None:
    """Siblings scope should raise error for non-tree sequences."""
    adapter = RepairMissingLinks(candidate_scope="siblings")

    span = make_adapting_span("s1", "operation")
    source = AdaptingSequence([span])

    with pytest.raises(ValueError, match="siblings.*only applicable to tree"):
        adapter.adapt(source)


def test_repair_missing_links_siblings_scope_with_tree() -> None:
    """Siblings scope should work correctly with tree sequences."""
    adapter = RepairMissingLinks(candidate_scope="siblings", scan_direction="backward")

    # Build a simple tree: root -> [child1, child2]
    root = make_adapting_span("root", "root", start_time=0.0, end_time=10.0)
    child1 = make_adapting_span("child1", "operation", start_time=1.0, end_time=4.0)
    child2 = make_adapting_span(
        "child2",
        AGL_ANNOTATION,
        start_time=5.0,
        end_time=9.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    # Create tree structure
    child1_tree = Tree(child1, [])
    child2_tree = Tree(child2, [])
    tree = Tree(root, [child1_tree, child2_tree])

    result = adapter.adapt(tree)

    # child2 annotation should link to child1 (its sibling)
    child2_result = next(s for s in result.traverse() if s.span_id == "child2")
    assert isinstance(child2_result.data, GeneralAnnotation)
    assert len(child2_result.data.links) == 1
    assert child2_result.data.links[0].value_match == "child1"


# Edge case tests


def test_identify_annotations_empty_sequence() -> None:
    """Should handle empty sequence."""
    adapter = IdentifyAnnotations()
    result = adapter.adapt(AdaptingSequence([]))
    assert list(result) == []


def test_select_by_annotation_empty_sequence() -> None:
    """Should handle empty sequence."""
    adapter = SelectByAnnotation(mode="include")
    result = adapter.adapt(AdaptingSequence([]))
    assert len(list(result)) == 0


def test_repair_missing_links_empty_sequence() -> None:
    """Should handle empty sequence."""
    adapter = RepairMissingLinks()
    result = adapter.adapt(AdaptingSequence([]))
    assert len(list(result)) == 0


# Additional corner case tests


def test_identify_object_returns_annotation_with_none_when_no_object_attributes() -> None:
    """Should return ObjectAnnotation with None object when no JSON or literal present."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OBJECT,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={},
    )

    result = adapter.identify_object(span)

    assert result is not None
    assert result.annotation_type == "object"
    assert result.object is None


def test_extract_agent_name_returns_none_for_agentops_without_operation_name() -> None:
    """AgentOps span with kind=agent but no operation.name should return None."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "some.operation",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "agentops.span.kind": "agent",
            # Missing operation.name
        },
    )

    result = adapter.extract_agent_name(span)

    assert result is None


def test_extract_agent_name_returns_none_for_weave_without_input_name() -> None:
    """Weave span with type=agent but no input.name should return None."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "weave.call",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            "type": "agent",
            # Missing agentlightning.operation.input.name
        },
    )

    result = adapter.extract_agent_name(span)

    assert result is None


def test_extract_agent_name_returns_none_for_langchain_chain_without_lc_name() -> None:
    """LangChain chain span without lc_name should return None."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        "langchain.Chain.MyChain",
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            # Missing lc_name
        },
    )

    result = adapter.extract_agent_name(span)

    assert result is None


def test_identify_annotations_adapt_processes_multiple_spans() -> None:
    """Should process multiple spans in a sequence."""
    adapter = IdentifyAnnotations()

    span1 = make_adapting_span(
        "s1",
        AGL_ANNOTATION,
        attributes={f"{LightningSpanAttributes.TAG.value}.0": "tag1"},
    )
    span2 = make_adapting_span(
        "s2",
        AGL_MESSAGE,
        attributes={LightningSpanAttributes.MESSAGE_BODY.value: "Hello"},
    )
    span3 = make_adapting_span("s3", "regular.span")

    source = AdaptingSequence([span1, span2, span3])
    result = adapter.adapt(source)

    result_list = list(result)
    assert len(result_list) == 3
    assert isinstance(result_list[0].data, GeneralAnnotation)
    assert isinstance(result_list[1].data, MessageAnnotation)
    assert result_list[2].data is None


def test_select_by_annotation_links_use_and_logic() -> None:
    """Multiple links in an annotation use AND logic - span must match ALL links."""
    adapter = SelectByAnnotation(mode="include")

    # Links specify span_id AND a custom attribute must both match
    annotation_span = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[
                LinkPydanticModel(key_match="span_id", value_match="target"),
                LinkPydanticModel(key_match="custom.tag", value_match="important"),
            ],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    # This span matches both constraints
    target = make_adapting_span("target", "target.operation", attributes={"custom.tag": "important"})
    # This span only matches span_id but not the custom attribute
    partial_match = make_adapting_span("partial", "other.operation", attributes={"custom.tag": "important"})
    unrelated = make_adapting_span("unrelated", "operation3")

    source = AdaptingSequence([annotation_span, target, partial_match, unrelated])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    assert "annotation" in span_ids
    assert "target" in span_ids
    # partial_match doesn't match span_id constraint
    assert "partial" not in span_ids
    assert "unrelated" not in span_ids


def test_select_by_annotation_multiple_annotations_link_different_spans() -> None:
    """Multiple annotations can each link to different spans."""
    adapter = SelectByAnnotation(mode="include")

    ann1 = make_adapting_span(
        "ann1",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[LinkPydanticModel(key_match="span_id", value_match="linked1")],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    ann2 = make_adapting_span(
        "ann2",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[LinkPydanticModel(key_match="span_id", value_match="linked2")],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    linked1 = make_adapting_span("linked1", "operation1")
    linked2 = make_adapting_span("linked2", "operation2")
    unrelated = make_adapting_span("unrelated", "operation3")

    source = AdaptingSequence([ann1, ann2, linked1, linked2, unrelated])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    assert "ann1" in span_ids
    assert "ann2" in span_ids
    assert "linked1" in span_ids
    assert "linked2" in span_ids
    assert "unrelated" not in span_ids


def test_repair_missing_links_multiple_annotations_single_candidate_no_reuse() -> None:
    """Without reuse, only one annotation gets linked to a candidate."""
    adapter = RepairMissingLinks(scan_direction="backward", allow_reuse_linked_spans=False)

    candidate = make_adapting_span("candidate", "operation", start_time=0.0, end_time=1.0)
    ann1 = make_adapting_span(
        "ann1",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    ann2 = make_adapting_span(
        "ann2",
        AGL_ANNOTATION,
        start_time=2.0,
        end_time=3.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    ann3 = make_adapting_span(
        "ann3",
        AGL_ANNOTATION,
        start_time=3.0,
        end_time=4.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([candidate, ann1, ann2, ann3])
    result = adapter.adapt(source)

    # Only one annotation should get linked
    linked = [s for s in result if isinstance(s.data, GeneralAnnotation) and len(s.data.links) > 0]
    assert len(linked) == 1


def test_repair_missing_links_backward_links_nearest_candidate() -> None:
    """Backward scan should link each annotation to its nearest preceding candidate.

    When multiple annotations appear before candidates (in chronological order),
    the annotation closest to the candidate should get that candidate.

    Chronological order: [C1, C2, A1, A2]
    Expected: A2->C2 (nearest), A1->C1 (remaining)
    """
    adapter = RepairMissingLinks(scan_direction="backward", allow_reuse_linked_spans=False)

    c1 = make_adapting_span("c1", "operation", start_time=0.0, end_time=1.0)
    c2 = make_adapting_span("c2", "operation", start_time=1.0, end_time=2.0)
    a1 = make_adapting_span(
        "a1",
        AGL_ANNOTATION,
        start_time=2.0,
        end_time=3.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    a2 = make_adapting_span(
        "a2",
        AGL_ANNOTATION,
        start_time=3.0,
        end_time=4.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([c1, c2, a1, a2])
    result = adapter.adapt(source)

    # A2 (latest annotation) should link to C2 (nearest candidate before it)
    a2_result = next(s for s in result if s.span_id == "a2")
    assert isinstance(a2_result.data, GeneralAnnotation)
    assert len(a2_result.data.links) == 1
    assert a2_result.data.links[0].value_match == "c2"

    # A1 should link to C1 (remaining candidate)
    a1_result = next(s for s in result if s.span_id == "a1")
    assert isinstance(a1_result.data, GeneralAnnotation)
    assert len(a1_result.data.links) == 1
    assert a1_result.data.links[0].value_match == "c1"


def test_repair_missing_links_forward_links_nearest_candidate() -> None:
    """Forward scan should link each annotation to its nearest following candidate.

    When multiple annotations appear before candidates (in chronological order),
    the annotation closest to the candidate should get that candidate.

    Chronological order: [A1, A2, C1, C2]
    Expected: A1->C1 (nearest), A2->C2 (remaining)
    """
    adapter = RepairMissingLinks(scan_direction="forward", allow_reuse_linked_spans=False)

    a1 = make_adapting_span(
        "a1",
        AGL_ANNOTATION,
        start_time=0.0,
        end_time=1.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    a2 = make_adapting_span(
        "a2",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    c1 = make_adapting_span("c1", "operation", start_time=2.0, end_time=3.0)
    c2 = make_adapting_span("c2", "operation", start_time=3.0, end_time=4.0)

    source = AdaptingSequence([a1, a2, c1, c2])
    result = adapter.adapt(source)

    # A1 (earliest annotation) should link to C1 (nearest candidate after it)
    a1_result = next(s for s in result if s.span_id == "a1")
    assert isinstance(a1_result.data, GeneralAnnotation)
    assert len(a1_result.data.links) == 1
    assert a1_result.data.links[0].value_match == "c1"

    # A2 should link to C2 (remaining candidate)
    a2_result = next(s for s in result if s.span_id == "a2")
    assert isinstance(a2_result.data, GeneralAnnotation)
    assert len(a2_result.data.links) == 1
    assert a2_result.data.links[0].value_match == "c2"


def test_repair_missing_links_backward_interleaved() -> None:
    """Backward scan with interleaved annotations and candidates.

    Chronological order: [C1, A1, C2, A2]
    Expected: A1->C1 (nearest before A1), A2->C2 (nearest before A2)
    """
    adapter = RepairMissingLinks(scan_direction="backward", allow_reuse_linked_spans=False)

    c1 = make_adapting_span("c1", "operation", start_time=0.0, end_time=1.0)
    a1 = make_adapting_span(
        "a1",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    c2 = make_adapting_span("c2", "operation", start_time=2.0, end_time=3.0)
    a2 = make_adapting_span(
        "a2",
        AGL_ANNOTATION,
        start_time=3.0,
        end_time=4.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([c1, a1, c2, a2])
    result = adapter.adapt(source)

    # A2 should link to C2 (nearest candidate before A2)
    a2_result = next(s for s in result if s.span_id == "a2")
    assert isinstance(a2_result.data, GeneralAnnotation)
    assert len(a2_result.data.links) == 1
    assert a2_result.data.links[0].value_match == "c2"

    # A1 should link to C1 (nearest candidate before A1)
    a1_result = next(s for s in result if s.span_id == "a1")
    assert isinstance(a1_result.data, GeneralAnnotation)
    assert len(a1_result.data.links) == 1
    assert a1_result.data.links[0].value_match == "c1"


def test_repair_missing_links_forward_interleaved() -> None:
    """Forward scan with interleaved annotations and candidates.

    Chronological order: [A1, C1, A2, C2]
    Expected: A1->C1 (nearest after A1), A2->C2 (nearest after A2)
    """
    adapter = RepairMissingLinks(scan_direction="forward", allow_reuse_linked_spans=False)

    a1 = make_adapting_span(
        "a1",
        AGL_ANNOTATION,
        start_time=0.0,
        end_time=1.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    c1 = make_adapting_span("c1", "operation", start_time=1.0, end_time=2.0)
    a2 = make_adapting_span(
        "a2",
        AGL_ANNOTATION,
        start_time=2.0,
        end_time=3.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    c2 = make_adapting_span("c2", "operation", start_time=3.0, end_time=4.0)

    source = AdaptingSequence([a1, c1, a2, c2])
    result = adapter.adapt(source)

    # A1 should link to C1 (nearest candidate after A1)
    a1_result = next(s for s in result if s.span_id == "a1")
    assert isinstance(a1_result.data, GeneralAnnotation)
    assert len(a1_result.data.links) == 1
    assert a1_result.data.links[0].value_match == "c1"

    # A2 should link to C2 (nearest candidate after A2)
    a2_result = next(s for s in result if s.span_id == "a2")
    assert isinstance(a2_result.data, GeneralAnnotation)
    assert len(a2_result.data.links) == 1
    assert a2_result.data.links[0].value_match == "c2"


def test_repair_missing_links_skips_annotation_spans_as_candidates() -> None:
    """Annotation spans should not be considered as link candidates."""
    adapter = RepairMissingLinks(scan_direction="backward")

    # Two annotations in sequence - second should not link to first
    ann1 = make_adapting_span(
        "ann1",
        AGL_ANNOTATION,
        start_time=0.0,
        end_time=1.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    ann2 = make_adapting_span(
        "ann2",
        AGL_ANNOTATION,
        start_time=1.0,
        end_time=2.0,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )

    source = AdaptingSequence([ann1, ann2])
    result = adapter.adapt(source)

    # Neither annotation should get links since there are no candidates
    for span in result:
        if isinstance(span.data, GeneralAnnotation):
            assert len(span.data.links) == 0


def test_identify_operation_with_no_input_output() -> None:
    """Should handle operation spans with no input or output."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_OPERATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            LightningSpanAttributes.OPERATION_NAME.value: "simple_op",
        },
    )

    result = adapter.identify_operation(span)

    assert result is not None
    assert result.name == "simple_op"
    assert result.input == {}
    assert result.output == {}


def test_identify_general_with_multiple_rewards() -> None:
    """Should extract multiple rewards and use first as primary."""
    adapter = IdentifyAnnotations()
    span = make_span(
        "s1",
        AGL_ANNOTATION,
        parent_id=None,
        start_time=0.0,
        end_time=1.0,
        attributes={
            f"{LightningSpanAttributes.REWARD.value}.0.name": "quality",
            f"{LightningSpanAttributes.REWARD.value}.0.value": 0.9,
            f"{LightningSpanAttributes.REWARD.value}.1.name": "relevance",
            f"{LightningSpanAttributes.REWARD.value}.1.value": 0.7,
            f"{LightningSpanAttributes.TAG.value}.0": "test",
        },
    )

    result = adapter.identify_general(span)

    assert result is not None
    assert len(result.rewards) == 2
    assert result.primary_reward == 0.9


def test_adapt_one_with_object_span() -> None:
    """Should identify AGL_OBJECT spans via adapt_one."""
    adapter = IdentifyAnnotations()
    source = make_adapting_span(
        "s1",
        AGL_OBJECT,
        attributes={LightningSpanAttributes.OBJECT_JSON.value: '{"key": "value"}'},
    )

    result = adapter.adapt_one(source)

    assert isinstance(result.data, ObjectAnnotation)
    assert result.data.object == {"key": "value"}


def test_select_by_annotation_with_link_using_attribute_match() -> None:
    """Should support links that match by custom attribute."""
    adapter = SelectByAnnotation(mode="include")

    annotation_span = make_adapting_span(
        "annotation",
        AGL_ANNOTATION,
        data=GeneralAnnotation(
            annotation_type="general",
            links=[LinkPydanticModel(key_match="custom.tag", value_match="important")],
            rewards=[],
            primary_reward=None,
            tags=[],
            custom_fields={},
        ),
    )
    target = make_adapting_span("target", "target.operation", attributes={"custom.tag": "important"})
    other = make_adapting_span("other", "other.operation", attributes={"custom.tag": "other"})

    source = AdaptingSequence([annotation_span, target, other])
    result = adapter.adapt(source)

    span_ids = [s.span_id for s in result]
    assert "annotation" in span_ids
    assert "target" in span_ids
    assert "other" not in span_ids
