# Copyright (c) Microsoft. All rights reserved.

"""Find and repair the annotations from spans."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, TypeVar, cast

from opentelemetry.semconv.attributes import exception_attributes

from agentlightning.adapter.preprocess import default_span_order
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
    AdaptingSpan,
    AgentAnnotation,
    Annotation,
    BaseAdaptingSequence,
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

from .base import SequenceAdapter

T_SpanSequence = TypeVar("T_SpanSequence", bound=Sequence[Span])

logger = logging.getLogger(__name__)


class IdentifyAnnotations(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
    """Identify and parse annotation data from spans based on span name conventions.

    This adapter inspects each span's name to determine if it represents a known
    annotation type (general, message, object, exception, operation) or an agent span.
    When identified, the span's data field is populated with the corresponding annotation
    model containing extracted attributes.

    Supported annotation types:

    - `AGL_ANNOTATION`: General annotations with rewards, tags, and custom fields.
    - `AGL_MESSAGE`: Message annotations containing a message body.
    - `AGL_OBJECT`: Object annotations containing serialized JSON or literal values.
    - `AGL_EXCEPTION`: Exception annotations with type, message, and stacktrace.
    - `AGL_OPERATION`: Operation annotations with name, input, and output.
    - Agent spans: Detected via heuristics for various agent frameworks.
    """

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
        """Extract link specifications from span attributes.

        Args:
            span: The span to extract links from.

        Returns:
            A sequence of link models. Returns an empty list if no links are found
            or if the link attributes are malformed.
        """
        try:
            return extract_links_from_attributes(span.attributes)
        except Exception as exc:
            logger.error(f"Link is malformed for span {span.span_id}: {exc}")
            return []

    def identify_general(self, span: Span) -> Optional[GeneralAnnotation]:
        """Parse a general annotation span into a `GeneralAnnotation` model.

        Extracts rewards, tags, links, and custom fields from the span attributes.

        Args:
            span: A span with name `AGL_ANNOTATION`.

        Returns:
            A `GeneralAnnotation` with extracted data, or None if parsing fails.
        """
        rewards = get_rewards_from_span(span)
        primary_reward = rewards[0].value if rewards else None
        return GeneralAnnotation(
            annotation_type="general",
            links=self.extract_links(span),
            rewards=rewards,
            primary_reward=primary_reward,
            tags=extract_tags_from_attributes(span.attributes),
            custom_fields=self._filter_custom_attributes(span.attributes),
        )

    def identify_message(self, span: Span) -> Optional[MessageAnnotation]:
        """Parse a message span into a `MessageAnnotation` model.

        Args:
            span: A span with name `AGL_MESSAGE`.

        Returns:
            A `MessageAnnotation` containing the message body, or None if the
            message body attribute is missing.
        """
        msg_body = get_message_value(span)
        if msg_body is None:
            logger.warning(f"Message body is missing for message span {span.span_id}")
            return None

        return MessageAnnotation(
            annotation_type="message",
            links=self.extract_links(span),
            message=msg_body,
        )

    def identify_object(self, span: Span) -> Optional[ObjectAnnotation]:
        """Parse an object span into an `ObjectAnnotation` model.

        Supports both JSON-serialized objects and literal values.

        Args:
            span: A span with name `AGL_OBJECT`.

        Returns:
            An `ObjectAnnotation` containing the deserialized object, or None if
            deserialization fails.
        """
        try:
            obj_value = get_object_value(span)
        except Exception as exc:
            logger.error(f"Fail to deserialize object for object span {span.span_id}: {exc}")
            return None

        return ObjectAnnotation(
            annotation_type="object",
            links=self.extract_links(span),
            object=obj_value,
        )

    def identify_exception(self, span: Span) -> Optional[ExceptionAnnotation]:
        """Parse an exception span into an `ExceptionAnnotation` model.

        Uses OpenTelemetry semantic conventions for exception attributes.

        Args:
            span: A span with name `AGL_EXCEPTION`.

        Returns:
            An `ExceptionAnnotation` containing exception type, message, and stacktrace.
            Missing fields default to "UnknownException" for type and empty string for others.
        """
        exception_type = span.attributes.get(exception_attributes.EXCEPTION_TYPE, "UnknownException")
        exception_message = span.attributes.get(exception_attributes.EXCEPTION_MESSAGE, "")
        exception_stacktrace = span.attributes.get(exception_attributes.EXCEPTION_STACKTRACE, "")

        return ExceptionAnnotation(
            annotation_type="exception",
            links=self.extract_links(span),
            type=str(exception_type),
            message=str(exception_message),
            stacktrace=str(exception_stacktrace),
        )

    def identify_operation(self, span: Span) -> Optional[OperationAnnotation]:
        """Parse an operation span into an `OperationAnnotation` model.

        Extracts operation name, input, and output. Input/output can be either
        direct values or nested structures reconstructed from flattened attributes.

        Args:
            span: A span with name `AGL_OPERATION`.

        Returns:
            An `OperationAnnotation` containing operation details, or None if
            attribute unpacking fails.
        """
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
            links=self.extract_links(span),
            name=str(operation_name),
            input=operation_input,
            output=operation_output,
        )

    def extract_agent_id(self, span: Span) -> Optional[str]:
        """Extract agent ID from span attributes.

        Args:
            span: The span to extract the agent ID from.

        Returns:
            The agent ID if found, None otherwise.
        """
        # TODO: Support agent id in other formats
        return cast(Optional[str], span.attributes.get("agent.id"))

    def extract_agent_description(self, span: Span) -> Optional[str]:
        """Extract agent description from span attributes.

        Args:
            span: The span to extract the agent description from.

        Returns:
            The agent description if found, None otherwise.
        """
        # TODO: Support agent description in other formats
        return cast(Optional[str], span.attributes.get("agent.description"))

    def extract_agent_name(self, span: Span) -> Optional[str]:
        """Extract agent name from span attributes using framework-specific heuristics.

        Supports multiple agent frameworks by checking various attribute patterns:

        1. OpenTelemetry agent spans (`agent.name`)
        2. AgentOps decorated agents (`agentops.span.kind` + `operation.name`)
        3. Autogen teams (`recipient_agent_type`)
        4. LangGraph (`langchain.chain.type`)
        5. agent-framework (`executor.id`)
        6. Weave (`type` == "agent" + `agentlightning.operation.input.name`)
        7. Weave + LangChain (`langchain.Chain.*` span names + `lc_name`)

        Args:
            span: The span to extract the agent name from.

        Returns:
            The agent name if detected via any supported pattern, None otherwise.
        """
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

        return None

    def detect_agent_annotation(self, span: Span) -> Optional[AgentAnnotation]:
        """Detect and create an agent annotation from span attributes.

        Uses heuristics to identify spans representing agent executions from
        various frameworks (OpenTelemetry, AgentOps, Autogen, LangGraph, etc.).

        Args:
            span: The span to check for agent indicators.

        Returns:
            An `AgentAnnotation` if an agent is detected, None otherwise.
        """
        agent_id = self.extract_agent_id(span)
        agent_name = self.extract_agent_name(span)
        agent_description = self.extract_agent_description(span)

        if agent_name is not None:
            return AgentAnnotation(
                annotation_type="agent",
                links=self.extract_links(span),
                id=agent_id,
                name=agent_name,
                description=agent_description,
            )
        return None

    def adapt_one(self, source: AdaptingSpan) -> AdaptingSpan:
        """Process a single span to identify and attach annotation data.

        Checks the span name against known annotation types and parses the
        corresponding annotation model. Falls back to agent detection for
        unrecognized span names.

        Args:
            source: The span to process.

        Returns:
            The span with annotation data attached if identified, otherwise
            the original span unchanged.
        """
        annotation: Optional[Annotation] = None
        if source.name == AGL_ANNOTATION:
            annotation = self.identify_general(source)
        elif source.name == AGL_MESSAGE:
            annotation = self.identify_message(source)
        elif source.name == AGL_OBJECT:
            annotation = self.identify_object(source)
        elif source.name == AGL_EXCEPTION:
            annotation = self.identify_exception(source)
        elif source.name == AGL_OPERATION:
            annotation = self.identify_operation(source)
        else:
            # Fallback to agent annotation detection
            annotation = self.detect_agent_annotation(source)
        if annotation is not None:
            return source.with_data(annotation)
        else:
            return source


class SelectByAnnotation(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
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

    def _filter_linked_spans(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Iterable[AdaptingSpan]:
        annotation_spans = [span for span in source if isinstance(span.data, Annotation)]
        annotation_span_ids = set(annotation_span.span_id for annotation_span in annotation_spans)
        annotation_links = [cast(Annotation, span.data).links for span in annotation_spans]
        for span in source:
            if span.span_id in annotation_span_ids:
                yield span
            # Only check non-empty link lists; empty links means the annotation applies only to itself
            # (check_linked_span returns True for empty links, which would incorrectly match all spans)
            elif any(links and check_linked_span(span, links) for links in annotation_links):
                yield span
            # ignore the current span for now

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> BaseAdaptingSequence[AdaptingSpan]:
        """Filter spans based on annotation membership and links.

        Args:
            source: The span sequence to filter.

        Returns:
            A filtered sequence containing only annotated spans and their linked
            spans (include mode), or all spans except those (exclude mode).
        """
        linked_spans = list(self._filter_linked_spans(source))
        if self.mode == "include":
            return source.retain(lambda span: span in linked_spans)
        else:
            # prune removes items where predicate is True, so we remove linked spans
            return source.prune(lambda span: span in linked_spans)


class RepairMissingLinks(SequenceAdapter[AdaptingSpan, AdaptingSpan]):
    """Populate missing annotation links by searching nearby spans.

    This adapter scans annotations and, for any annotation that has no linked spans, attempts
    to infer and attach link targets using a configurable search strategy.

    Typical use case: upstream extraction produced annotations (e.g., entities, citations)
    but failed to attach their target spans; this adapter backfills those links based on
    proximity and eligibility rules.

    Args:
        candidate_predicate:
            A predicate to filter the candidate spans. If None, all spans within the candidate scope are considered.

        candidate_scope:
            Controls which spans are eligible as link targets:

            - "siblings": search only among sibling spans of the annotation span. Only applicable when input span sequence is a tree.
            - "all": search among all spans provided to the adapter.

            The intersection of the candidate scope and predicate forms the candidate span set.

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
        candidate_predicate: Optional[Callable[[AdaptingSpan], bool]] = None,
        candidate_scope: Literal["siblings", "all"] = "all",
        scan_direction: Literal["backward", "forward"] = "backward",
        allow_reuse_linked_spans: bool = False,
    ) -> None:
        if candidate_predicate is not None:
            self.candidate_predicate = candidate_predicate
        else:
            self.candidate_predicate: Callable[[AdaptingSpan], bool] = lambda _: True
        self.candidate_scope = candidate_scope
        self.scan_direction = scan_direction
        self.allow_reuse_linked_spans = allow_reuse_linked_spans

    def _search_groups(self, source: BaseAdaptingSequence[AdaptingSpan]) -> Iterable[Sequence[AdaptingSpan]]:
        if self.candidate_scope == "siblings":
            if not isinstance(source, Tree):
                raise ValueError("Candidate scope 'siblings' is only applicable to tree sequences")

            def visit(node: Tree[AdaptingSpan]) -> Iterable[Sequence[AdaptingSpan]]:
                # Each group must be siblings
                yield [child.item for child in node.children]  # yield siblings first
                for child in node.children:  # then yield children recursively
                    yield from visit(child)

            yield [source.item]  # yield root first
            yield from visit(source)

        elif self.candidate_scope == "all":
            # Return as a single group containing all spans sorted by default order
            yield sorted(list(source), key=lambda span: default_span_order(span))

        else:
            raise ValueError(f"Invalid candidate scope: {self.candidate_scope}")

    def adapt(self, source: BaseAdaptingSequence[AdaptingSpan]) -> BaseAdaptingSequence[AdaptingSpan]:
        """Repair annotations that have no links by inferring targets from nearby spans.

        Scans the span sequence according to the configured direction and scope,
        linking annotations without targets to the nearest eligible candidate spans.

        Args:
            source: The span sequence containing annotations to repair.

        Returns:
            A new sequence with repaired annotations containing inferred links.

        Raises:
            ValueError: If `candidate_scope` is "siblings" but source is not a tree.
        """
        groups = list(self._search_groups(source))
        span_id_to_link: Dict[str, LinkPydanticModel] = {}
        for group in groups:
            if self.scan_direction == "backward":
                group_to_scan = reversed(group)
            else:
                group_to_scan = group

            annotations_to_fill: List[AdaptingSpan] = []
            for span in group_to_scan:
                if isinstance(span.data, Annotation):
                    if not span.data.links:
                        annotations_to_fill.append(span)
                    # The span is an annotation, skip it from being a candidate
                else:
                    # The span is a candidate
                    if self.candidate_predicate(span):
                        while len(annotations_to_fill) > 0:
                            # Fill the link with the earliest-encountered annotation first (FIFO order)
                            # This ensures each annotation links to its nearest candidate in the scan direction
                            annotation_span = annotations_to_fill.pop(0)
                            span_id_to_link[annotation_span.span_id] = LinkPydanticModel(
                                key_match="span_id", value_match=span.span_id
                            )

                            if not self.allow_reuse_linked_spans:
                                # Once used, the candidate span cannot be reused
                                break
                        # If no annotations to fill, the candidate is wasted
                    # Otherwise, the span is not a candidate, skip it

        def _update_links(span: AdaptingSpan) -> AdaptingSpan:
            if span.span_id in span_id_to_link and isinstance(span.data, Annotation):
                new_annotation = span.data.model_copy(update={"links": [span_id_to_link[span.span_id]]})
                return span.model_copy(update={"data": new_annotation})
            else:
                return span

        return source.map(_update_links)
