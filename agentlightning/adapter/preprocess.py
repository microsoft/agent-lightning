# Copyright (c) Microsoft. All rights reserved.

"""Span re-organization adapters."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Literal, Sequence, Set, Tuple, TypeVar

from agentlightning.semconv import AGL_VIRTUAL
from agentlightning.types.adapter import AdaptingSequence, AdaptingSpan, Tree
from agentlightning.types.tracer import Span, SpanLike
from agentlightning.utils.id import generate_id

from .base import Adapter

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")
T_span = TypeVar("T_span", bound=Span, covariant=True)

logger = logging.getLogger(__name__)


def default_span_order(span: Span) -> tuple[int, float, float]:
    """Return a tuple for sorting spans by sequence ID, then start time, then end time.

    Args:
        span: The span to extract ordering keys from.

    Returns:
        A tuple of (sequence_id, start_time, end_time) for use as a sort key.

    Raises:
        ValueError: If the span has no start or end time set.
    """
    return (span.sequence_id, span.ensure_start_time(), span.ensure_end_time())


class _TreeLikeGraph:
    """A simple directed graph implementation for span hierarchy.

    In preparation before forming a tree.
    """

    def __init__(self) -> None:
        self.forward_graph: Dict[str, List[str]] = defaultdict(list)
        self.parent_map: Dict[str, str] = {}
        self.root_ids: Set[str] = set()
        self.spans_dict: Dict[str, Span] = {}

    def add_edge(self, from_node: str, to_node: str) -> None:
        self.forward_graph[from_node].append(to_node)

    def move_subtree(self, node_id: str, new_parent_id: str) -> None:
        old_parent_id = self.parent_map.get(node_id, None)
        if old_parent_id is not None:
            self.forward_graph[old_parent_id].remove(node_id)
        self.add_edge(new_parent_id, node_id)
        self.parent_map[node_id] = new_parent_id
        if node_id in self.root_ids:
            self.root_ids.remove(node_id)

    def validate_no_cycles(self) -> None:
        visited = set[str]()

        def visit(node_id: str) -> None:
            if node_id in visited:
                raise ValueError(f"Cycle detected in the graph at node {node_id}")
            visited.add(node_id)
            for child_id in self.forward_graph[node_id]:
                visit(child_id)

        for root_id in self.root_ids:
            visit(root_id)

        if len(visited) != len(self.forward_graph):
            raise ValueError("Some nodes are not reachable from the roots")

    def compute_depths(self) -> Dict[str, int]:
        depths = {root: 0 for root in self.root_ids}

        def visit(node_id: str) -> None:
            for child_id in self.forward_graph[node_id]:
                depths[child_id] = depths[node_id] + 1
                visit(child_id)

        for root_id in self.root_ids:
            visit(root_id)

        return depths

    def compute_ancestors(self) -> Dict[str, Set[str]]:
        ancestors = {root: set[str]() for root in self.root_ids}

        def visit(node_id: str) -> None:
            for child_id in self.forward_graph[node_id]:
                ancestors[child_id] = ancestors[node_id] | {node_id}
                visit(child_id)

        for root_id in self.root_ids:
            visit(root_id)

        return ancestors

    def to_tree(self) -> Tree[AdaptingSpan]:
        def build_subtree(node_id: str) -> Tree[AdaptingSpan]:
            children = [build_subtree(child_id) for child_id in self.forward_graph.get(node_id, [])]
            item = AdaptingSpan.from_span(self.spans_dict[node_id], None)
            return Tree(item, sorted(children, key=lambda child: default_span_order(child.item)))

        if len(self.root_ids) != 1:
            raise ValueError(
                "Cannot convert to tree: multiple or no roots found; enable repair options of ToTree to fix this."
            )
        root_id = next(iter(self.root_ids))
        return build_subtree(root_id)

    @staticmethod
    def from_spans(spans: Sequence[Span], logs_invalid_parent: bool = True) -> _TreeLikeGraph:
        graph = _TreeLikeGraph()

        valid_span_ids = set(span.span_id for span in spans)
        graph.root_ids = set(span.span_id for span in spans if span.parent_id is None)
        graph.spans_dict = {span.span_id: span for span in spans}
        for span in spans:
            if span.parent_id is not None:
                if span.parent_id in valid_span_ids:
                    graph.forward_graph[span.parent_id].append(span.span_id)
                    graph.root_ids.discard(span.span_id)
                    graph.parent_map[span.span_id] = span.parent_id
                    # We don't care about the wrong parent id in the spans dict
                    # This fix here is only for graph construction
                else:
                    # Span has invalid parent, treat as root
                    graph.root_ids.add(span.span_id)
                    if logs_invalid_parent:
                        logger.debug(
                            f'Span {span.span_id} has an invalid parent ID "{span.parent_id}". '
                            "The parent will be ignored and the span will be treated as a root."
                        )

        graph.validate_no_cycles()

        return graph


class ToSpans(Adapter[Sequence[SpanLike], Sequence[Span]]):
    """Normalize span-like objects (e.g., OpenTelemetry `ReadableSpan`) to [Span][agentlightning.Span].

    This adapter handles conversion from various span formats to the internal Span type.
    Native Span objects pass through unchanged, while OpenTelemetry spans are converted
    using the provided default values for rollout, attempt, and sequence identifiers.

    Args:
        default_rollout_id: Default rollout ID for converted OpenTelemetry spans.
        default_attempt_id: Default attempt ID for converted OpenTelemetry spans.
        default_sequence_id: Default sequence ID for converted OpenTelemetry spans.
    """

    def __init__(
        self,
        default_rollout_id: str = "rollout-dummy",
        default_attempt_id: str = "attempt-dummy",
        default_sequence_id: int = 0,
    ):
        self.default_rollout_id = default_rollout_id
        self.default_attempt_id = default_attempt_id
        self.default_sequence_id = default_sequence_id

    def adapt_one(self, source: SpanLike) -> Span:
        """Convert a single span-like object to a Span.

        Args:
            source: A Span or OpenTelemetry ReadableSpan to convert.

        Returns:
            The converted Span object. Native Spans pass through unchanged.
        """
        if isinstance(source, Span):
            return source
        return Span.from_opentelemetry(
            source,
            rollout_id=self.default_rollout_id,
            attempt_id=self.default_attempt_id,
            sequence_id=self.default_sequence_id,
        )

    def adapt(self, source: Sequence[SpanLike]) -> Sequence[Span]:
        """Convert a sequence of span-like objects to Spans.

        Args:
            source: A sequence of Span or OpenTelemetry ReadableSpan objects.
        """
        return [self.adapt_one(span) for span in source]


class ToTree(Adapter[Sequence[Span], Tree[AdaptingSpan]]):
    """Convert a sequence of spans into a tree structure.

    This adapter organizes flat span sequences into a hierarchical tree based on parent-child
    relationships. It can repair various structural issues in the span data:

    - **Bad hierarchy**: Spans that are incorrectly positioned (e.g., dangling spans without
      proper parents despite being contained within other spans' time ranges).
    - **Multiple roots**: Cases where more than one span has no parent.

    Note:
        Spans with invalid parent IDs (referencing non-existent spans) will cause a ValueError.
        Use [RepairMalformedSpans][agentlightning.adapter.preprocess.RepairMalformedSpans] with
        `ensure_valid_parent_ids=True` before calling this adapter if you need to handle
        invalid parent references.

    Args:
        repair_bad_hierarchy: Controls hierarchy repair. `"dangling"` repairs only orphaned spans,
            `"all"` re-evaluates all span placements, `"none"` skips hierarchy repair.
        repair_multiple_roots: If True, creates a virtual root when multiple root spans exist.

    Raises:
        TypeError: If the input is not a sequence.
        ValueError: If no spans are provided, if any span has an invalid parent ID,
            or if the tree cannot be constructed.
    """

    def __init__(
        self,
        repair_bad_hierarchy: Literal["dangling", "all", "none"] = "dangling",
        repair_multiple_roots: bool = True,
    ):
        self.repair_bad_hierarchy = repair_bad_hierarchy
        self.repair_multiple_roots = repair_multiple_roots

    def _find_eligible_parents(
        self,
        all_spans: Sequence[Span],
        current: Span,
        graph: _TreeLikeGraph,
        cache_depths: Dict[str, int],
    ) -> List[Span]:
        """We wish to find a good place to insert the span, which is ideally it's sibling or sibling's child.

        Filter the candidates by: (1) must not in current's ancestors; (2) must not be in current's subtree;
        (3) must have an ancestor that is the parent of current span; (4) have start and end time covering the current span.
        The third condition can be optional if current span has no parent.

        Then sort the candidates by: (1) shortest to longest, (2) deep to shallow.
        """
        spans_to_consider: List[Span] = []
        # This needs to be re-computed every time.
        # It will be too troublesome to maintain the dynamic cache of ancestors.
        ancestors = graph.compute_ancestors()

        for candidate_parent in all_spans:
            if candidate_parent.span_id == current.span_id:
                continue
            if (
                candidate_parent.ensure_start_time() > current.ensure_start_time()
                or candidate_parent.ensure_end_time() < current.ensure_end_time()
            ):
                # If the span is not covering the current span, it cannot be a parent.
                continue
            if candidate_parent.span_id in ancestors[current.span_id]:
                # If the span is in the current's ancestors, it cannot be a parent.
                continue
            if current.span_id in ancestors[candidate_parent.span_id]:
                # If the span is in the current's subtree, it cannot be a parent.
                continue
            if current.span_id in graph.parent_map:
                # If the current span has a parent, the eligible parent must live in the parent's subtree.
                if graph.parent_map[current.span_id] not in ancestors[candidate_parent.span_id]:
                    continue
            spans_to_consider.append(candidate_parent)

        # Sort the spans: (1) shortest to longest duration, (2) deeper to shallower (prefer more specific ancestors)
        return sorted(
            spans_to_consider,
            key=lambda span: (span.ensure_end_time() - span.ensure_start_time(), -cache_depths[span.span_id]),
        )

    def _repair_bad_hierarchy(self, source: Sequence[Span]) -> Sequence[Span]:
        """Repair bad hierarchy by re-attaching dangling spans or all spans.

        This is based on the chronological relationships between start time and end time of spans.
        """
        if self.repair_bad_hierarchy == "none":
            return source

        graph = _TreeLikeGraph.from_spans(source)
        depths = graph.compute_depths()

        # Scan all the spans by: (1) shallow to deep, (2) longest to shortest, (3) earliest to latest.
        scan_order = sorted(
            source,
            key=lambda span: (
                depths[span.span_id],
                -(span.ensure_end_time() - span.ensure_start_time()),
                span.ensure_start_time(),
            ),
        )
        for i, span in enumerate(scan_order):
            # Check whether we should repair this span.
            # It must be a dangling span, or the user wants to repair all the spans.
            if (
                self.repair_bad_hierarchy == "dangling" and span.span_id not in graph.parent_map
            ) or self.repair_bad_hierarchy == "all":
                # We wish to find a good place to insert the span.
                eligible_parents = self._find_eligible_parents(source, span, graph, depths)
                if eligible_parents:
                    new_parent_id = eligible_parents[0].span_id
                    scan_order[i] = span.model_copy(update={"parent_id": new_parent_id})

                    # Maintain/update the cache
                    graph.move_subtree(span.span_id, new_parent_id)

        return scan_order

    def _validate_parent_ids(self, source: Sequence[Span]) -> None:
        """Validate that all parent IDs reference existing spans.

        Raises:
            ValueError: If any span references a non-existent parent.
        """
        valid_span_ids: Set[str] = set(span.span_id for span in source)
        invalid_refs: List[str] = []

        for span in source:
            if span.parent_id is not None and span.parent_id not in valid_span_ids:
                invalid_refs.append(f"{span.span_id} -> {span.parent_id}")

        if invalid_refs:
            raise ValueError(
                f"Spans reference non-existent parent IDs: {', '.join(invalid_refs)}. "
                "Use RepairMalformedSpans with ensure_valid_parent_ids=True to fix this."
            )

    def _repair_multiple_roots(self, source: Sequence[Span]) -> Sequence[Span]:
        root_spans = [span for span in source if span.parent_id is None]

        if len(root_spans) <= 1:
            return source

        # Create a new root span
        new_root_span = Span.from_attributes(
            rollout_id=root_spans[0].rollout_id,
            attempt_id=root_spans[0].attempt_id,
            sequence_id=root_spans[0].sequence_id,
            trace_id=root_spans[0].trace_id,
            span_id="span-" + generate_id(12),
            parent_id=None,
            name=AGL_VIRTUAL,
            attributes={},
            start_time=min(span.ensure_start_time() for span in root_spans),
            end_time=max(span.ensure_end_time() for span in root_spans),
        )

        updated_spans = [
            span.model_copy(update={"parent_id": new_root_span.span_id}) if span in root_spans else span
            for span in source
        ]
        return [new_root_span, *updated_spans]

    def adapt(self, source: Sequence[Span]) -> Tree[AdaptingSpan]:
        """Convert a sequence of spans into a tree of AdaptingSpan objects.

        Args:
            source: A sequence of Span objects to organize into a tree.

        Returns:
            A Tree with AdaptingSpan items representing the hierarchical structure.

        Raises:
            TypeError: If source is not a sequence.
            ValueError: If source is empty, has invalid parent IDs, or cannot form a valid tree.
        """
        if not isinstance(source, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected a sequence of spans, but got {type(source)}")
        if not source:
            raise ValueError("No spans provided to create Tree")

        # Validate parent IDs before any processing
        self._validate_parent_ids(source)

        source = self._repair_bad_hierarchy(source)

        if self.repair_multiple_roots:
            source = self._repair_multiple_roots(source)

        graph = _TreeLikeGraph.from_spans(source)
        return graph.to_tree()


class ToAdaptingSpans(Adapter[Sequence[Span], AdaptingSequence[AdaptingSpan]]):
    """Convert spans to a sorted AdaptingSequence.

    Sorts spans by sequence ID (primary), start time (secondary), and end time (tertiary),
    then wraps each in an AdaptingSpan for use in adaptation pipelines.
    """

    def adapt(self, source: Sequence[Span]) -> AdaptingSequence[AdaptingSpan]:
        """Sort spans and convert to AdaptingSequence.

        Args:
            source: A sequence of Span objects to sort and convert.

        Returns:
            An AdaptingSequence containing sorted AdaptingSpan objects.
        """
        sorted_spans = sorted(source, key=default_span_order)
        return AdaptingSequence([AdaptingSpan.from_span(span, None) for span in sorted_spans])


class RepairMalformedSpans(Adapter[Sequence[Span], Sequence[Span]]):
    """Repair common structural issues in span data.

    This adapter fixes several types of malformed span data:

    - **Missing times**: Fills in missing start/end times using the maximum known time.
    - **Negative duration**: Adjusts end times that are earlier than start times.
    - **Improper nesting**: Expands parent time ranges to contain all children.
    - **Invalid parent IDs**: Removes references to non-existent parent spans.

    Spans that don't require repair pass through unchanged (same object reference).

    Args:
        ensure_positive_duration: If True, sets end_time = start_time when end < start.
        ensure_proper_nesting: If True, expands parent spans to contain children's time ranges.
        ensure_valid_parent_ids: If True, sets parent_id to None for orphaned spans.
    """

    def __init__(
        self,
        ensure_positive_duration: bool = True,
        ensure_proper_nesting: bool = True,
        ensure_valid_parent_ids: bool = True,
    ) -> None:
        self.ensure_positive_duration = ensure_positive_duration
        self.ensure_proper_nesting = ensure_proper_nesting
        self.ensure_valid_parent_ids = ensure_valid_parent_ids

    def _repair_start_end_time(self, source: Sequence[Span]) -> List[Span]:
        times_set = set[float]()
        for span in source:
            if span.start_time is not None:
                times_set.add(span.start_time)
            if span.end_time is not None:
                times_set.add(span.end_time)

        if not times_set:
            logger.debug("No times set in the spans. Setting all the time to current time.")
            current_time = time.time()
        else:
            current_time = max(times_set)

        new_spans: List[Span] = []

        for span in source:
            update_fields: Dict[str, float] = {}
            if span.start_time is None:
                update_fields["start_time"] = current_time
            if span.end_time is None:
                update_fields["end_time"] = current_time
            if (
                self.ensure_positive_duration
                and span.start_time is not None
                and span.end_time is not None
                and span.end_time < span.start_time
            ):
                update_fields["end_time"] = span.start_time
            if update_fields:
                new_spans.append(span.model_copy(update=update_fields))
            else:
                new_spans.append(span)

        return new_spans

    def _repair_nesting(self, source: Sequence[Span]) -> List[Span]:
        graph = _TreeLikeGraph.from_spans(source, logs_invalid_parent=False)
        spans = {span.span_id: span for span in source}

        def visit(node_id: str) -> Tuple[float, float]:
            child_start_end_times: List[Tuple[float, float]] = []
            cur_start_time = spans[node_id].ensure_start_time()
            cur_end_time = spans[node_id].ensure_end_time()
            if graph.forward_graph.get(node_id):
                for child_id in graph.forward_graph[node_id]:
                    child_start_end_times.append(visit(child_id))
                start_times, end_times = zip(*child_start_end_times)
                start_time = min(cur_start_time, *start_times)
                end_time = max(cur_end_time, *end_times)
                if start_time != cur_start_time or end_time != cur_end_time:
                    spans[node_id] = spans[node_id].model_copy(update={"start_time": start_time, "end_time": end_time})

            return spans[node_id].ensure_start_time(), spans[node_id].ensure_end_time()

        for root_id in graph.root_ids:
            visit(root_id)

        return [spans[span.span_id] for span in source]

    def _repair_invalid_parent_ids(self, source: Sequence[Span]) -> List[Span]:
        valid_span_ids: Set[str] = set(span.span_id for span in source)
        new_spans: List[Span] = []

        for span in source:
            if span.parent_id is not None and span.parent_id not in valid_span_ids:
                new_spans.append(span.model_copy(update={"parent_id": None, "parent": None}))
            else:
                new_spans.append(span)

        return new_spans

    def adapt(self, source: Sequence[Span]) -> Sequence[Span]:
        """Repair malformed spans according to the configured repair options.

        Args:
            source: A sequence of Span objects to repair.

        Returns:
            A sequence of repaired Span objects. Unmodified spans retain their original
            object reference.
        """
        # This step is always performed first no matter whether the flags are set.
        new_spans = self._repair_start_end_time(source)
        if self.ensure_proper_nesting:
            new_spans = self._repair_nesting(new_spans)
        if self.ensure_valid_parent_ids:
            new_spans = self._repair_invalid_parent_ids(new_spans)
        return new_spans
