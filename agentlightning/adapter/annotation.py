# Copyright (c) Microsoft. All rights reserved.

"""Find and repair the annotations from spans."""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, TypeVar, cast

from opentelemetry.semconv.attributes import exception_attributes

from agentlightning.emitter.message import get_message_value
from agentlightning.emitter.object import get_object_value
from agentlightning.emitter.reward import get_rewards_from_span
from agentlightning.semconv import (
    AGL_ANNOTATION,
    AGL_EXCEPTION,
    AGL_MESSAGE,
    AGL_OBJECT,
    AGL_OPERATION,
    LightningSpanAttributes,
    LinkPydanticModel,
)
from agentlightning.types.adapter import (
    AgentAnnotation,
    Annotation,
    ExceptionAnnotation,
    GeneralAnnotation,
    MessageAnnotation,
    ObjectAnnotation,
    OperationAnnotation,
    Tree,
)
from agentlightning.types.tracer import Span
from agentlightning.utils.otel import (
    check_linked_span,
    extract_links_from_attributes,
    extract_tags_from_attributes,
    filter_and_unflatten_attributes,
)

from .base import Adapter

T_SpanSequence = TypeVar("T_SpanSequence", bound=Sequence[Span])

logger = logging.getLogger(__name__)


class CurateAnnotations(Adapter[Sequence[Span], Sequence[Annotation]]):
    """Curate the annotations from the spans."""

    def _filter_custom_attributes(self, attributes: Dict[str, Any]) -> Dict[str, Any]:
        reserved_fields = [attr.value for attr in LightningSpanAttributes if attr.value in attributes]
        return {
            key: value
            for key, value in attributes.items()
            if not any(
                # Filter out those that are reserved fields or start with reserved fields (plus ".")
                key == reserved_field or key.startswith(reserved_field + ".")
                for reserved_field in reserved_fields
            )
        }

    def extract_links(self, span: Span) -> Sequence[LinkPydanticModel]:
        try:
            return extract_links_from_attributes(span.attributes)
        except Exception as exc:
            logger.error(f"Link is malformed for span {span.span_id}: {exc}")
            return []

    def adapt_general(self, span: Span) -> Optional[GeneralAnnotation]:
        rewards = get_rewards_from_span(span)
        primary_reward = rewards[0].value if rewards else None
        return GeneralAnnotation(
            annotation_type="general",
            span_id=span.span_id,
            links=self.extract_links(span),
            rewards=rewards,
            primary_reward=primary_reward,
            tags=extract_tags_from_attributes(span.attributes),
            custom_fields=self._filter_custom_attributes(span.attributes),
        )

    def adapt_message(self, span: Span) -> Optional[MessageAnnotation]:
        msg_body = get_message_value(span)
        if msg_body is None:
            logger.warning(f"Message body is missing for message span {span.span_id}")
            return None

        return MessageAnnotation(
            annotation_type="message",
            span_id=span.span_id,
            links=self.extract_links(span),
            message=msg_body,
        )

    def adapt_object(self, span: Span) -> Optional[ObjectAnnotation]:
        try:
            obj_value = get_object_value(span)
        except Exception as exc:
            logger.error(f"Fail to deserialize object for object span {span.span_id}: {exc}")
            return None

        return ObjectAnnotation(
            annotation_type="object",
            span_id=span.span_id,
            links=self.extract_links(span),
            object=obj_value,
        )

    def adapt_exception(self, span: Span) -> Optional[ExceptionAnnotation]:
        exception_type = span.attributes.get(exception_attributes.EXCEPTION_TYPE, "UnknownException")
        exception_message = span.attributes.get(exception_attributes.EXCEPTION_MESSAGE, "")
        exception_stacktrace = span.attributes.get(exception_attributes.EXCEPTION_STACKTRACE, "")

        return ExceptionAnnotation(
            annotation_type="exception",
            span_id=span.span_id,
            links=self.extract_links(span),
            type=str(exception_type),
            message=str(exception_message),
            stacktrace=str(exception_stacktrace),
        )

    def adapt_operation(self, span: Span) -> Optional[OperationAnnotation]:
        try:
            operation_name = span.attributes.get(LightningSpanAttributes.OPERATION_NAME.value, "UnknownOperation")
            if LightningSpanAttributes.OPERATION_INPUT.value in span.attributes:
                operation_input = span.attributes[LightningSpanAttributes.OPERATION_INPUT.value]
            else:
                operation_input = filter_and_unflatten_attributes(
                    span.attributes, LightningSpanAttributes.OPERATION_INPUT.value
                )
            if LightningSpanAttributes.OPERATION_OUTPUT.value in span.attributes:
                operation_output = span.attributes[LightningSpanAttributes.OPERATION_OUTPUT.value]
            else:
                operation_output = filter_and_unflatten_attributes(
                    span.attributes, LightningSpanAttributes.OPERATION_OUTPUT.value
                )
        except Exception as exc:
            logger.error(f"Fail to unpack operation context for operation span {span.span_id}: {exc}")
            return None

        return OperationAnnotation(
            annotation_type="operation",
            span_id=span.span_id,
            links=self.extract_links(span),
            name=str(operation_name),
            input=operation_input,
            output=operation_output,
        )

    def extract_agent_id(self, span: Span) -> Optional[str]:
        # TODO: Support agent id in other formats
        return cast(Optional[str], span.attributes.get("agent.id"))

    def extract_agent_description(self, span: Span) -> Optional[str]:
        # TODO: Support agent description in other formats
        return cast(Optional[str], span.attributes.get("agent.description"))

    def extract_agent_name(self, span: Span) -> Optional[str]:
        # Case 1: OpenTelemetry Agent Spans
        agent_name = cast(Optional[str], span.attributes.get("agent.name"))
        if agent_name is not None:
            return agent_name

        # Case 2: Agentops decorator @agent
        is_agent = span.attributes.get("agentops.span.kind") == "agent"
        if is_agent:
            agent_name = cast(Optional[str], span.attributes.get("operation.name"))
            if agent_name is not None:
                return agent_name

        # Case 3: Autogen team
        agent_name = cast(Optional[str], span.attributes.get("recipient_agent_type"))
        if agent_name is not None:
            return agent_name

        # Case 4: LangGraph
        agent_name = cast(Optional[str], span.attributes.get("langchain.chain.type"))
        if agent_name is not None:
            return agent_name

        # Case 5: agent-framework
        agent_name = cast(Optional[str], span.attributes.get("executor.id"))
        if agent_name is not None:
            return agent_name

        # Case 6: Weave
        is_agent_type = span.attributes.get("type") == "agent"
        if is_agent_type:
            agent_name = cast(Optional[str], span.attributes.get("agentlightning.operation.input.name"))
            if agent_name is not None:
                return agent_name

        # Case 7: Weave + LangChain
        if span.name.startswith("langchain.Chain."):
            attributes_lc_name = cast(Optional[str], span.attributes.get("lc_name"))
            if attributes_lc_name is not None:
                return attributes_lc_name

    def detect_agent_annotation(self, span: Span) -> Optional[AgentAnnotation]:
        agent_id = self.extract_agent_id(span)
        agent_name = self.extract_agent_name(span)
        agent_description = self.extract_agent_description(span)

        if agent_name is not None:
            return AgentAnnotation(
                annotation_type="agent",
                span_id=span.span_id,
                links=self.extract_links(span),
                id=agent_id,
                name=agent_name,
                description=agent_description,
            )
        return None

    def adapt(self, source: Sequence[Span]) -> Sequence[Annotation]:
        annotations: List[Annotation] = []
        for span in source:
            annotation: Optional[Annotation] = None
            if span.name == AGL_ANNOTATION:
                annotation = self.adapt_general(span)
            elif span.name == AGL_MESSAGE:
                annotation = self.adapt_message(span)
            elif span.name == AGL_OBJECT:
                annotation = self.adapt_object(span)
            elif span.name == AGL_EXCEPTION:
                annotation = self.adapt_exception(span)
            elif span.name == AGL_OPERATION:
                annotation = self.adapt_operation(span)
            else:
                # Fallback to agent annotation detection
                annotation = self.detect_agent_annotation(span)
            if annotation is not None:
                annotations.append(annotation)
        return annotations


class SelectByAnnotation(Adapter[Tuple[T_SpanSequence, Sequence[Annotation]], T_SpanSequence]):
    """Select the corresponding spans within the annotation sequence, as well as their linked spans
    (and subtree spans if applicable).

    The effective radius of an annotation is as follows:

    - If the annotation has links, it applies to the linked spans only.
    - If the annotation is on a tree node, it applies to all spans in its subtree.
    - If the annotation has neither links nor tree nodes, it applies to only itself.

    The adapter either selects the union of the effective radius of all annotations,
    or excludes the union of effective radius.

    When the source is a tree, to avoid the tree nodes from becoming fragmented,
    the adapter will also include the ancestors of the tree nodes in "include" mode.

    Args:
        mode: "include" to select spans within the annotations; "exclude" to exclude them.

    """

    def __init__(self, mode: Literal["include", "exclude"]) -> None:
        self.mode = mode

    def _filter_linked_spans(self, source: Sequence[Span], annotation: Sequence[Annotation]) -> Iterable[Span]:
        annotation_span_ids = set(annotation.span_id for annotation in annotation)
        for span in source:
            if span.span_id in annotation_span_ids:
                yield span
            elif any(check_linked_span(span, annotation.links) for annotation in annotation):
                yield span
            # ignore the current span for now

    def adapt(self, source: Tuple[T_SpanSequence, Sequence[Annotation]]) -> T_SpanSequence:
        spans, annotations = source
        linked_spans = self._filter_linked_spans(spans, annotations)
        if isinstance(spans, Tree):
            if self.mode == "include":
                return cast(T_SpanSequence, spans.retain(lambda span: span in linked_spans))
            else:
                return cast(T_SpanSequence, spans.prune(lambda span: span not in linked_spans))
        else:
            if self.mode == "include":
                return cast(T_SpanSequence, list(linked_spans))
            else:
                return cast(T_SpanSequence, [span for span in spans if span not in linked_spans])


class RepairMissingLinks(Adapter[Tuple[Sequence[Span], Sequence[Annotation]], Sequence[Span]]):
    """Populate missing annotation links by searching nearby spans.

    This adapter scans annotations and, for any annotation that has no linked spans, attempts
    to infer and attach link targets using a configurable search strategy.

    Typical use case: upstream extraction produced annotations (e.g., entities, citations)
    but failed to attach their target spans; this adapter backfills those links based on
    proximity and eligibility rules.

    The annotation spans do not necessarily have to appear in the input span sequence.

    Args:
        annotation_span_required:
            If True, only attempt to fill links for annotations whose *span_id* is present
            in the candidate span set. If False, annotations are considered
            regardless of whether their span is in the input span sequence.

        candidate_scope:
            Controls which spans are eligible as link targets:

            - "siblings": search only among sibling spans of the annotation span. Only applicable when input span sequence is a tree.
            - "all": search among all spans provided to the adapter.

        scan_direction:
            Determines both (a) which direction the adapter searches for candidate targets
            relative to an annotation and (b) the order in which annotations are processed:

            - "backward": search earlier spans; process annotations from latest to earliest.
            - "forward": search later spans; process annotations from earliest to latest.

        allow_reuse_linked_spans:
            If False, spans already linked by *any* annotation are not eligible targets for
            additional links (i.e., enforce a one-to-one-ish linking constraint).
            If True, a span may be linked multiple times by different annotations.
    """

    def __init__(
        self,
        require_annotation_span_child: bool = True,
        candidate_scope: Literal["siblings", "all"] = "all",
        scan_direction: Literal["backward", "forward"] = "backward",
        allow_reuse_linked_spans: bool = False,
    ) -> None:
        self.require_annotation_span_child = require_annotation_span_child
        self.candidate_scope = candidate_scope
        self.scan_direction = scan_direction
        self.allow_reuse_linked_spans = allow_reuse_linked_spans

    def adapt(self, source: Tuple[Sequence[Span], Sequence[Annotation]]) -> Sequence[Span]: ...
