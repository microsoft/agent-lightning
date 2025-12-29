# Copyright (c) Microsoft. All rights reserved.

"""Find and repair the annotations from spans."""

from __future__ import annotations

from typing import Literal, Sequence, Tuple, TypeVar

from agentlightning.types.adapter import Annotation
from agentlightning.types.tracer import Span

from .base import Adapter

T_SpanSequence = TypeVar("T_SpanSequence", bound=Sequence[Span])


class CurateAnnotations(Adapter[Sequence[Span], Sequence[Annotation]]):
    """Curate the annotations from the spans."""

    def adapt(self, source: Sequence[Span]) -> Sequence[Annotation]: ...


class SelectByAnnotation(Adapter[Tuple[T_SpanSequence, Sequence[Annotation]], T_SpanSequence]):
    """Select the corresponding spans within the annotation sequence, as well as their linked spans
    (and subtree spans if applicable).

    The effective radius of an annotation is as follows:

    - If the annotation has links, it applies to the linked spans only.
    - If the annotation is on a tree node, it applies to all spans in the subtree.
    - If the annotation has neither links nor tree nodes, it applies to only itself.

    Args:
        mode: "include" to select spans within the annotations; "exclude" to exclude them.
    """

    def __init__(self, mode: Literal["include", "exclude"]) -> None:
        self.mode = mode

    def adapt(self, source: Tuple[T_SpanSequence, Sequence[Annotation]]) -> T_SpanSequence: ...


class FillMissingLinks(Adapter[Tuple[Sequence[Span], Sequence[Annotation]], Sequence[Span]]):
    """Populate missing annotation links by searching nearby spans.

    This adapter scans annotations and, for any annotation that has no linked spans, attempts
    to infer and attach link targets using a configurable search strategy.

    Typical use case: upstream extraction produced annotations (e.g., entities, citations)
    but failed to attach their target spans; this adapter backfills those links based on
    proximity and eligibility rules.

    Args:
        require_annotation_span_child:
            If True, only attempt to fill links for annotations whose *own* span is present
            as a child span in the candidate span set. If False, annotations are considered
            regardless of whether their span is a child.

        candidate_scope:
            Controls which spans are eligible as link targets:

            - "siblings": search only among sibling spans of the annotation span.
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
