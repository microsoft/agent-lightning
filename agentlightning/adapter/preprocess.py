# Copyright (c) Microsoft. All rights reserved.

"""Span re-organization adapters."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List, Literal, Sequence, Set, TypeVar

from agentlightning.types.adapter import Tree
from agentlightning.types.tracer import Span, SpanLike

from .base import Adapter, SequenceAdapter, Sort

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")

logger = logging.getLogger(__name__)


class ToSpans(SequenceAdapter[SpanLike, Span]):
    """Normalize the span-like objects (e.g., OpenTelemetry `ReadableSpan`) to [spans][agentlightning.Span]."""

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
        if isinstance(source, Span):
            return source
        return Span.from_opentelemetry(
            source,
            rollout_id=self.default_rollout_id,
            attempt_id=self.default_attempt_id,
            sequence_id=self.default_sequence_id,
        )


class ToTree(Adapter[Sequence[Span], Tree[Span]]):

    def __init__(
        self,
        repair_bad_hierarchy: Literal["dangling", "all", "none"] = "dangling",
        repair_missing_parents: bool = True,
    ):
        self.repair_bad_hierarchy = repair_bad_hierarchy
        self.repair_missing_parents = repair_missing_parents

    def _validate_tree(self, graph: Dict[str, List[str]], root_ids: Set[str]) -> None:
        visited = set[str]()

        def visit(node_id: str) -> None:
            if node_id in visited:
                raise ValueError(f"Cycle detected in the tree: {node_id}")
            visited.add(node_id)
            for child_id in graph[node_id]:
                visit(child_id)

        for root_id in root_ids:
            visit(root_id)

        if len(visited) != len(graph):
            raise ValueError(f"Some spans are not reachable from the roots: {set(graph.keys()) - visited}")

    def _compute_depths(self, graph: Dict[str, List[str]], root_ids: Set[str]) -> Dict[str, int]:
        depths = {root: 0 for root in root_ids}

        def visit(node_id: str) -> None:
            for child_id in graph[node_id]:
                depths[child_id] = depths[node_id] + 1
                visit(child_id)

        for root_id in root_ids:
            visit(root_id)

        return depths

    def _compute_ancestors(self, graph: Dict[str, List[str]], root_ids: Set[str]) -> Dict[str, Set[str]]:
        ancestors = {root: set[str]() for root in root_ids}

        def visit(node_id: str) -> None:
            for child_id in graph[node_id]:
                ancestors[child_id] = ancestors[node_id] | {node_id}
                visit(child_id)

        for root_id in root_ids:
            visit(root_id)

        return ancestors

    def _find_eligible_parents(
        self,
        all_spans: Sequence[Span],
        current: Span,
        forward_graph: Dict[str, List[str]],
        root_ids: Set[str],
        depths: Dict[str, int],
        parent_ids: Dict[str, str],
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
        ancestors = self._compute_ancestors(forward_graph, root_ids)

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
            if current.span_id in parent_ids:
                # If the current span has a parent, the eligible parent must live in the parent's subtree.
                if parent_ids[current.span_id] not in ancestors[candidate_parent.span_id]:
                    continue
            spans_to_consider.append(candidate_parent)

        # Sort the spans
        return sorted(
            spans_to_consider,
            key=lambda span: (span.ensure_end_time() - span.ensure_start_time(), depths[span.span_id]),
        )

    def _repair_bad_hierarchy(
        self,
        source: Sequence[Span],
        forward_graph: Dict[str, List[str]],
        root_ids: Set[str],
        parent_ids: Dict[str, str],
    ) -> Sequence[Span]:
        depths = self._compute_depths(forward_graph, root_ids)

        # Scan all the spans by: (1) shallow to deep, (2) longest to shortest, (3) earliest to latest.
        scan_order = sorted(
            source,
            key=lambda span: (
                depths[span.span_id],
                span.ensure_end_time() - span.ensure_start_time(),
                span.ensure_start_time(),
            ),
        )
        for i, span in enumerate(scan_order):
            # Check whether we should repair this span.
            # It must be a dangling span, or the user wants to repair all the spans.
            if (
                self.repair_bad_hierarchy == "dangling" and span.span_id not in parent_ids
            ) or self.repair_bad_hierarchy == "all":
                # We wish to find a good place to insert the span.
                eligible_parents = self._find_eligible_parents(
                    source, span, forward_graph, root_ids, depths, parent_ids
                )
                if eligible_parents:
                    original_parent_id = parent_ids.get(span.span_id, None)
                    new_parent_id = eligible_parents[0].span_id
                    scan_order[i] = span.model_copy(update={"parent_id": new_parent_id})
                    # Maintain the cache
                    parent_ids[span.span_id] = new_parent_id
                    if original_parent_id is not None:
                        forward_graph[original_parent_id].remove(span.span_id)
                        forward_graph[new_parent_id].append(span.span_id)

        return scan_order

    def adapt(self, source: Sequence[Span]) -> Tree[Span]:
        if not isinstance(source, Sequence):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError(f"Expected a sequence of spans, but got {type(source)}")
        if not source:
            raise ValueError("No spans provided to create Tree")

        valid_span_ids = set(span.span_id for span in source)
        root_ids = set(span.span_id for span in source if span.parent_id is None)
        forward_graph: Dict[str, List[str]] = defaultdict(list)
        parent_ids: Dict[str, str] = {}
        for span in source:
            if span.parent_id is not None:
                if span.parent_id in valid_span_ids:
                    forward_graph[span.parent_id].append(span.span_id)
                    root_ids.discard(span.span_id)
                    parent_ids[span.span_id] = span.parent_id
                else:
                    logger.debug(
                        f'Span {span.span_id} has an invalid parent ID "{span.parent_id}". The parent will be ignored.'
                    )

        self._validate_tree(forward_graph, root_ids)

        source = self._repair_bad_hierarchy(source, forward_graph, root_ids, parent_ids)

        return Tree(source[0], [Tree(span, []) for span in source if span.parent_id is not None])


class ToSortedSpans(Sort[Span]):
    """Sort the spans with sequence ID as the primary key and start time as the secondary key."""

    def __init__(self) -> None:
        super().__init__(key=lambda span: (span.sequence_id, span.start_time))


class RepairTime(Adapter[Sequence[Span], Sequence[Span]]):
    """Repair the end time of the spans by:

    1. Ensuring the end time is greater than the start time.
    2. Fill the spans with no end time to be the maximum start/end time of all spans.
    """

    def adapt(self, source: Sequence[Span]) -> Sequence[Span]:
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
            if span.start_time is not None and span.end_time is not None and span.end_time < span.start_time:
                update_fields["end_time"] = span.start_time
            if update_fields:
                new_spans.append(span.model_copy(update=update_fields))
            else:
                new_spans.append(span)
        return new_spans


class RepairTreeHierarchy(Adapter[Tree[Span], Tree[Span]]):
    """Repair the tree hierarchy by ensuring that parent-child relationships are consistent
    with span start and end times. Adding missing parent-child relationships as needed.
    """

    def adapt(self, source: Tree[Span]) -> Tree[Span]: ...
