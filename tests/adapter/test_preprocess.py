# Copyright (c) Microsoft. All rights reserved.

"""Tests for the preprocess module adapters."""

import itertools
from typing import Any, Dict, List, Optional

import pytest

from agentlightning.adapter.preprocess import _TreeLikeGraph  # pyright: ignore[reportPrivateUsage]
from agentlightning.adapter.preprocess import (
    RepairMalformedSpans,
    ToAdaptingSpans,
    ToSpans,
    ToTree,
    default_span_order,
)
from agentlightning.semconv import AGL_VIRTUAL
from agentlightning.types import Span
from agentlightning.types.adapter import AdaptingSpan

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


# Tests for default_span_order


def test_default_span_order_by_sequence_id():
    span1 = make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0, sequence_id=2)
    span2 = make_span("s2", "span", parent_id=None, start_time=0.0, end_time=1.0, sequence_id=1)
    spans = [span1, span2]
    sorted_spans = sorted(spans, key=default_span_order)
    assert [s.span_id for s in sorted_spans] == ["s2", "s1"]


def test_default_span_order_by_start_time():
    span1 = make_span("s1", "span", parent_id=None, start_time=2.0, end_time=3.0, sequence_id=0)
    span2 = make_span("s2", "span", parent_id=None, start_time=1.0, end_time=3.0, sequence_id=0)
    spans = [span1, span2]
    sorted_spans = sorted(spans, key=default_span_order)
    assert [s.span_id for s in sorted_spans] == ["s2", "s1"]


def test_default_span_order_by_end_time():
    span1 = make_span("s1", "span", parent_id=None, start_time=1.0, end_time=3.0, sequence_id=0)
    span2 = make_span("s2", "span", parent_id=None, start_time=1.0, end_time=2.0, sequence_id=0)
    spans = [span1, span2]
    sorted_spans = sorted(spans, key=default_span_order)
    assert [s.span_id for s in sorted_spans] == ["s2", "s1"]


# Tests for _TreeLikeGraph


def test_tree_like_graph_from_spans_creates_correct_graph():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child1 = make_span("child1", "child", parent_id="root", start_time=1.0, end_time=5.0)
    child2 = make_span("child2", "child", parent_id="root", start_time=5.0, end_time=9.0)
    grandchild = make_span("grandchild", "grandchild", parent_id="child1", start_time=2.0, end_time=4.0)

    graph = _TreeLikeGraph.from_spans([root, child1, child2, grandchild])

    assert graph.root_ids == {"root"}
    assert set(graph.forward_graph["root"]) == {"child1", "child2"}
    assert graph.forward_graph["child1"] == ["grandchild"]
    assert graph.parent_map["child1"] == "root"
    assert graph.parent_map["child2"] == "root"
    assert graph.parent_map["grandchild"] == "child1"


def test_tree_like_graph_from_spans_handles_invalid_parent():
    """Spans with invalid parent IDs should be treated as roots."""
    orphan = make_span("orphan", "orphan", parent_id="nonexistent", start_time=0.0, end_time=1.0)

    graph = _TreeLikeGraph.from_spans([orphan])

    assert graph.root_ids == {"orphan"}
    assert "orphan" not in graph.parent_map


def test_tree_like_graph_from_spans_multiple_roots():
    root1 = make_span("root1", "root", parent_id=None, start_time=0.0, end_time=5.0)
    root2 = make_span("root2", "root", parent_id=None, start_time=5.0, end_time=10.0)

    graph = _TreeLikeGraph.from_spans([root1, root2])

    assert graph.root_ids == {"root1", "root2"}


def test_tree_like_graph_compute_depths():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child = make_span("child", "child", parent_id="root", start_time=1.0, end_time=9.0)
    grandchild = make_span("grandchild", "grandchild", parent_id="child", start_time=2.0, end_time=8.0)

    graph = _TreeLikeGraph.from_spans([root, child, grandchild])
    depths = graph.compute_depths()

    assert depths["root"] == 0
    assert depths["child"] == 1
    assert depths["grandchild"] == 2


def test_tree_like_graph_compute_ancestors():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child = make_span("child", "child", parent_id="root", start_time=1.0, end_time=9.0)
    grandchild = make_span("grandchild", "grandchild", parent_id="child", start_time=2.0, end_time=8.0)

    graph = _TreeLikeGraph.from_spans([root, child, grandchild])
    ancestors = graph.compute_ancestors()

    assert ancestors["root"] == set()
    assert ancestors["child"] == {"root"}
    assert ancestors["grandchild"] == {"root", "child"}


def test_tree_like_graph_move_subtree():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child1 = make_span("child1", "child", parent_id="root", start_time=1.0, end_time=5.0)
    child2 = make_span("child2", "child", parent_id="root", start_time=5.0, end_time=9.0)
    grandchild = make_span("grandchild", "grandchild", parent_id="child1", start_time=2.0, end_time=4.0)

    graph = _TreeLikeGraph.from_spans([root, child1, child2, grandchild])
    graph.move_subtree("grandchild", "child2")

    assert "grandchild" not in graph.forward_graph["child1"]
    assert "grandchild" in graph.forward_graph["child2"]
    assert graph.parent_map["grandchild"] == "child2"


def test_tree_like_graph_move_subtree_from_root():
    """Moving a root node should remove it from root_ids."""
    root1 = make_span("root1", "root", parent_id=None, start_time=0.0, end_time=10.0)
    root2 = make_span("root2", "root", parent_id=None, start_time=0.0, end_time=10.0)

    graph = _TreeLikeGraph.from_spans([root1, root2])
    assert "root2" in graph.root_ids

    graph.move_subtree("root2", "root1")

    assert "root2" not in graph.root_ids
    assert graph.parent_map["root2"] == "root1"


def test_tree_like_graph_to_tree_single_root():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child1 = make_span("child1", "child", parent_id="root", start_time=1.0, end_time=5.0)
    child2 = make_span("child2", "child", parent_id="root", start_time=5.0, end_time=9.0)

    graph = _TreeLikeGraph.from_spans([root, child1, child2])
    tree = graph.to_tree()

    assert tree.item.span_id == "root"
    assert len(tree.children) == 2
    child_ids = {child.item.span_id for child in tree.children}
    assert child_ids == {"child1", "child2"}


def test_tree_like_graph_to_tree_multiple_roots_raises():
    root1 = make_span("root1", "root", parent_id=None, start_time=0.0, end_time=5.0)
    root2 = make_span("root2", "root", parent_id=None, start_time=5.0, end_time=10.0)

    graph = _TreeLikeGraph.from_spans([root1, root2])

    with pytest.raises(ValueError, match="multiple or no roots"):
        graph.to_tree()


# Tests for ToSpans


def test_to_spans_pass_through_span():
    """Span objects should pass through unchanged."""
    span = make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0)
    adapter = ToSpans()

    result = adapter.adapt_one(span)

    assert result is span


def test_to_spans_default_values():
    """Adapter should use default values for rollout_id, attempt_id, and sequence_id."""
    adapter = ToSpans(
        default_rollout_id="my-rollout",
        default_attempt_id="my-attempt",
        default_sequence_id=42,
    )
    span = make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0)

    result = adapter.adapt_one(span)

    # Span passes through unchanged, defaults only apply to OpenTelemetry spans
    assert result is span


# Tests for ToTree


def test_to_tree_basic_creation():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child = make_span("child", "child", parent_id="root", start_time=1.0, end_time=9.0)
    spans = [root, child]

    adapter = ToTree()
    tree = adapter.adapt(spans)

    assert tree.item.span_id == "root"
    assert len(tree.children) == 1
    assert tree.children[0].item.span_id == "child"


def test_to_tree_empty_spans_raises():
    adapter = ToTree()

    with pytest.raises(ValueError, match="No spans provided"):
        adapter.adapt([])


def test_to_tree_non_sequence_raises():
    adapter = ToTree()

    # String is technically a sequence but will fail when trying to access span attributes
    with pytest.raises(AttributeError):
        adapter.adapt("not a sequence")  # type: ignore


def test_to_tree_repair_multiple_roots():
    root1 = make_span("root1", "root", parent_id=None, start_time=0.0, end_time=5.0)
    root2 = make_span("root2", "root", parent_id=None, start_time=5.0, end_time=10.0)
    spans = [root1, root2]

    adapter = ToTree(repair_multiple_roots=True)
    tree = adapter.adapt(spans)

    assert tree.item.name == AGL_VIRTUAL
    assert len(tree.children) == 2
    child_ids = {child.item.span_id for child in tree.children}
    assert child_ids == {"root1", "root2"}


def test_to_tree_repair_multiple_roots_disabled():
    root1 = make_span("root1", "root", parent_id=None, start_time=0.0, end_time=5.0)
    root2 = make_span("root2", "root", parent_id=None, start_time=5.0, end_time=10.0)
    spans = [root1, root2]

    adapter = ToTree(repair_multiple_roots=False)

    with pytest.raises(ValueError, match="multiple or no roots"):
        adapter.adapt(spans)


def test_to_tree_invalid_parent_raises_error():
    """Spans with invalid parent IDs should raise ValueError."""
    orphan = make_span("orphan", "orphan", parent_id="missing-parent", start_time=0.0, end_time=1.0)
    spans = [orphan]

    adapter = ToTree()

    with pytest.raises(ValueError, match="non-existent parent IDs"):
        adapter.adapt(spans)


def test_to_tree_with_repaired_invalid_parents():
    """Using RepairMalformedSpans before ToTree should handle invalid parent IDs."""
    orphan = make_span("orphan", "orphan", parent_id="missing-parent", start_time=0.0, end_time=1.0)
    spans = [orphan]

    # First repair invalid parent IDs
    repair_adapter = RepairMalformedSpans(ensure_valid_parent_ids=True)
    repaired = repair_adapter.adapt(spans)

    # Now ToTree should work
    tree_adapter = ToTree()
    tree = tree_adapter.adapt(repaired)

    # Orphan becomes root since parent_id was set to None
    assert tree.item.span_id == "orphan"


def test_to_tree_repair_bad_hierarchy_dangling():
    """Dangling spans should be re-attached based on time containment."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    container = make_span("container", "container", parent_id="root", start_time=1.0, end_time=9.0)
    # Dangling span (no parent) that should fit inside container
    dangling = make_span("dangling", "dangling", parent_id=None, start_time=2.0, end_time=8.0)
    spans = [root, container, dangling]

    adapter = ToTree(repair_bad_hierarchy="dangling")
    tree = adapter.adapt(spans)

    # Dangling should be moved under container (best fit by time)
    container_node = next(c for c in tree.children if c.item.span_id == "container")
    dangling_in_container = any(c.item.span_id == "dangling" for c in container_node.children)
    assert dangling_in_container


def test_to_tree_repair_bad_hierarchy_none():
    """When repair_bad_hierarchy is 'none', hierarchy is not repaired."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    dangling = make_span("dangling", "dangling", parent_id=None, start_time=2.0, end_time=8.0)
    spans = [root, dangling]

    adapter = ToTree(repair_bad_hierarchy="none", repair_multiple_roots=True)
    tree = adapter.adapt(spans)

    # Both should be roots under virtual root
    assert tree.item.name == AGL_VIRTUAL
    child_ids = {c.item.span_id for c in tree.children}
    assert child_ids == {"root", "dangling"}


def test_to_tree_children_sorted_by_time():
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child_late = make_span("child-late", "child", parent_id="root", start_time=5.0, end_time=9.0, sequence_id=0)
    child_early = make_span("child-early", "child", parent_id="root", start_time=1.0, end_time=4.0, sequence_id=0)
    spans = [root, child_late, child_early]

    adapter = ToTree()
    tree = adapter.adapt(spans)

    child_ids = [c.item.span_id for c in tree.children]
    assert child_ids == ["child-early", "child-late"]


def test_to_tree_adapting_span_properties():
    """Tree should contain AdaptingSpan instances with proper container references."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child = make_span("child", "child", parent_id="root", start_time=1.0, end_time=9.0)
    spans = [root, child]

    adapter = ToTree()
    tree = adapter.adapt(spans)

    assert isinstance(tree.item, AdaptingSpan)
    assert isinstance(tree.children[0].item, AdaptingSpan)


# Tests for ToAdaptingSpans


def test_to_adapting_spans_sorts_by_default_order():
    span1 = make_span("s1", "span", parent_id=None, start_time=2.0, end_time=3.0, sequence_id=0)
    span2 = make_span("s2", "span", parent_id=None, start_time=1.0, end_time=3.0, sequence_id=0)
    spans = [span1, span2]

    adapter = ToAdaptingSpans()
    result = adapter.adapt(spans)

    span_ids = [s.span_id for s in result]
    assert span_ids == ["s2", "s1"]


def test_to_adapting_spans_returns_adapting_sequence():
    span = make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0)
    spans = [span]

    adapter = ToAdaptingSpans()
    result = adapter.adapt(spans)

    assert len(result) == 1
    assert isinstance(result[0], AdaptingSpan)


# Tests for RepairMalformedSpans


def test_repair_malformed_spans_missing_start_time():
    span = Span.from_attributes(
        rollout_id="r1",
        attempt_id="a1",
        sequence_id=0,
        trace_id="t1",
        span_id="s1",
        parent_id=None,
        name="span",
        attributes={},
        start_time=None,
        end_time=5.0,
    )
    spans = [span]

    adapter = RepairMalformedSpans()
    result = adapter.adapt(spans)

    # Start time should be set to max of all times (5.0)
    assert result[0].start_time == 5.0


def test_repair_malformed_spans_missing_end_time():
    span = Span.from_attributes(
        rollout_id="r1",
        attempt_id="a1",
        sequence_id=0,
        trace_id="t1",
        span_id="s1",
        parent_id=None,
        name="span",
        attributes={},
        start_time=1.0,
        end_time=None,
    )
    spans = [span]

    adapter = RepairMalformedSpans()
    result = adapter.adapt(spans)

    # End time should be set to max of all times (1.0)
    assert result[0].end_time == 1.0


def test_repair_malformed_spans_both_missing_times():
    span = Span.from_attributes(
        rollout_id="r1",
        attempt_id="a1",
        sequence_id=0,
        trace_id="t1",
        span_id="s1",
        parent_id=None,
        name="span",
        attributes={},
        start_time=None,
        end_time=None,
    )
    spans = [span]

    adapter = RepairMalformedSpans()
    result = adapter.adapt(spans)

    # Both should be set to current time (they should be equal and non-None)
    assert result[0].start_time is not None
    assert result[0].end_time is not None
    assert result[0].start_time == result[0].end_time


def test_repair_malformed_spans_negative_duration():
    """When end_time < start_time, end_time should be set to start_time."""
    span = make_span("s1", "span", parent_id=None, start_time=5.0, end_time=3.0)
    spans = [span]

    adapter = RepairMalformedSpans(ensure_positive_duration=True)
    result = adapter.adapt(spans)

    assert result[0].end_time == 5.0


def test_repair_malformed_spans_no_repair_negative_duration_when_disabled():
    span = make_span("s1", "span", parent_id=None, start_time=5.0, end_time=3.0)
    spans = [span]

    adapter = RepairMalformedSpans(ensure_positive_duration=False)
    result = adapter.adapt(spans)

    assert result[0].end_time == 3.0


def test_repair_malformed_spans_invalid_parent_ids():
    span = make_span("s1", "span", parent_id="nonexistent", start_time=0.0, end_time=1.0)
    spans = [span]

    adapter = RepairMalformedSpans(ensure_valid_parent_ids=True)
    result = adapter.adapt(spans)

    assert result[0].parent_id is None


def test_repair_malformed_spans_no_repair_invalid_parent_ids_when_disabled():
    span = make_span("s1", "span", parent_id="nonexistent", start_time=0.0, end_time=1.0)
    spans = [span]

    adapter = RepairMalformedSpans(ensure_valid_parent_ids=False)
    result = adapter.adapt(spans)

    assert result[0].parent_id == "nonexistent"


def test_repair_malformed_spans_proper_nesting():
    """Parent span's time range should be expanded to contain all children."""
    parent = make_span("parent", "parent", parent_id=None, start_time=2.0, end_time=8.0)
    child = make_span("child", "child", parent_id="parent", start_time=1.0, end_time=9.0)
    spans = [parent, child]

    adapter = RepairMalformedSpans(ensure_proper_nesting=True)
    result = adapter.adapt(spans)

    parent_result = next(s for s in result if s.span_id == "parent")
    # Parent's time should be expanded to contain child
    assert parent_result.start_time == 1.0
    assert parent_result.end_time == 9.0


def test_repair_malformed_spans_no_repair_proper_nesting_when_disabled():
    parent = make_span("parent", "parent", parent_id=None, start_time=2.0, end_time=8.0)
    child = make_span("child", "child", parent_id="parent", start_time=1.0, end_time=9.0)
    spans = [parent, child]

    adapter = RepairMalformedSpans(ensure_proper_nesting=False)
    result = adapter.adapt(spans)

    parent_result = next(s for s in result if s.span_id == "parent")
    assert parent_result.start_time == 2.0
    assert parent_result.end_time == 8.0


def test_repair_malformed_spans_unchanged_pass_through():
    """Spans that don't need repair should not be modified."""
    span = make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0)
    spans = [span]

    adapter = RepairMalformedSpans()
    result = adapter.adapt(spans)

    # The span object should be the same (not copied)
    assert result[0] is span


# Integration tests


def test_integration_complex_tree_with_repairs():
    """Test a complex scenario with multiple issues that need repair."""
    # Root span
    root = make_span("root", "session", parent_id=None, start_time=0.0, end_time=100.0)

    # Agent span with misaligned timing
    agent = make_span("agent", "agent.node", parent_id="root", start_time=5.0, end_time=90.0)

    # LLM span that's a sibling of agent but should be under it (dangling repair)
    llm = make_span("llm", "openai.chat", parent_id="root", start_time=10.0, end_time=20.0)

    # Orphan span with missing parent
    orphan = make_span("orphan", "tool.call", parent_id="missing", start_time=30.0, end_time=40.0)

    spans = [root, agent, llm, orphan]

    # First repair invalid parent IDs
    repair_adapter = RepairMalformedSpans(ensure_valid_parent_ids=True)
    repaired = repair_adapter.adapt(spans)

    # Apply hierarchy repairs and convert to tree
    tree_adapter = ToTree(
        repair_bad_hierarchy="dangling",
        repair_multiple_roots=True,
    )
    tree = tree_adapter.adapt(repaired)

    # The tree should be properly structured
    assert tree.size() >= 4  # At least the original spans


def test_integration_adapting_spans_after_tree():
    """AdaptingSpans should work correctly on tree output."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child = make_span("child", "child", parent_id="root", start_time=1.0, end_time=9.0)
    spans = [root, child]

    tree_adapter = ToTree()
    tree = tree_adapter.adapt(spans)

    # Traverse and verify all items are AdaptingSpans
    for span in tree.traverse():
        assert isinstance(span, AdaptingSpan)


def test_integration_repair_then_tree():
    """RepairMalformedSpans followed by ToTree should work correctly."""
    parent = make_span("parent", "parent", parent_id=None, start_time=5.0, end_time=3.0)  # Invalid: end < start
    child = make_span("child", "child", parent_id="parent", start_time=1.0, end_time=2.0)

    # First repair the spans
    repair_adapter = RepairMalformedSpans()
    repaired = repair_adapter.adapt([parent, child])

    # Then create tree
    tree_adapter = ToTree()
    tree = tree_adapter.adapt(repaired)

    # Parent should have repaired time and contain child
    assert tree.item.start_time is not None
    assert tree.item.end_time is not None
    assert tree.item.end_time >= tree.item.start_time


# Edge case tests


def test_edge_case_single_span_tree():
    span = make_span("only", "only", parent_id=None, start_time=0.0, end_time=1.0)

    adapter = ToTree()
    tree = adapter.adapt([span])

    assert tree.item.span_id == "only"
    assert len(tree.children) == 0


def test_edge_case_deep_tree():
    """Test a deeply nested tree."""
    spans: List[Span] = []
    for i in range(10):
        parent_id = f"span-{i-1}" if i > 0 else None
        spans.append(
            make_span(
                f"span-{i}",
                f"level-{i}",
                parent_id=parent_id,
                start_time=float(i),
                end_time=float(20 - i),
            )
        )

    adapter = ToTree()
    tree = adapter.adapt(spans)

    assert tree.size() == 10

    # Verify depth
    current = tree
    depth = 0
    while current.children:
        depth += 1
        current = current.children[0]
    assert depth == 9


def test_edge_case_wide_tree():
    """Test a tree with many siblings."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=100.0)
    children = [
        make_span(f"child-{i}", "child", parent_id="root", start_time=float(i), end_time=float(i + 1))
        for i in range(20)
    ]

    adapter = ToTree()
    tree = adapter.adapt([root] + children)

    assert tree.item.span_id == "root"
    assert len(tree.children) == 20


def test_edge_case_spans_with_same_times():
    """Test handling of spans with identical timestamps.

    When repair_bad_hierarchy is disabled, spans with same times become multiple roots.
    """
    spans = [
        make_span(f"span-{i}", "span", parent_id=None, start_time=0.0, end_time=1.0, sequence_id=i) for i in range(5)
    ]

    # With repair_bad_hierarchy="none", no hierarchy repair happens
    adapter = ToTree(repair_bad_hierarchy="none", repair_multiple_roots=True)
    tree = adapter.adapt(spans)

    # All should become children of virtual root
    assert tree.item.name == AGL_VIRTUAL
    assert len(tree.children) == 5


def test_edge_case_repair_preserves_order():
    """Repaired spans should maintain their original order where possible."""
    spans = [
        make_span("s1", "span", parent_id=None, start_time=0.0, end_time=1.0, sequence_id=0),
        make_span("s2", "span", parent_id=None, start_time=1.0, end_time=2.0, sequence_id=1),
        make_span("s3", "span", parent_id=None, start_time=2.0, end_time=3.0, sequence_id=2),
    ]

    adapter = RepairMalformedSpans()
    result = adapter.adapt(spans)

    # Order should be preserved
    assert [s.span_id for s in result] == ["s1", "s2", "s3"]


# Additional corner case tests


def test_tree_like_graph_full_cycle_no_roots():
    """Graph where all nodes form a cycle (no roots) should raise ValueError."""
    # Create spans with a full cycle: A -> B -> C -> A (no roots)
    span_a = make_span("A", "span", parent_id="C", start_time=0.0, end_time=10.0)
    span_b = make_span("B", "span", parent_id="A", start_time=1.0, end_time=9.0)
    span_c = make_span("C", "span", parent_id="B", start_time=2.0, end_time=8.0)

    with pytest.raises(ValueError, match="not reachable from the roots"):
        _TreeLikeGraph.from_spans([span_a, span_b, span_c])


def test_tree_like_graph_cycle_in_subtree():
    """Graph with a cycle in a subtree should raise ValueError."""
    # Create a graph with cycle: root -> A -> B -> A
    # This tests the "Cycle detected" error path in validate_no_cycles
    graph = _TreeLikeGraph()
    graph.root_ids.add("root")
    graph.add_edge("root", "A")
    graph.add_edge("A", "B")
    graph.add_edge("B", "A")  # Creates cycle: A -> B -> A

    with pytest.raises(ValueError, match="Cycle detected"):
        graph.validate_no_cycles()


def test_tree_like_graph_add_edge_direct():
    """Test add_edge method directly."""
    graph = _TreeLikeGraph()
    graph.root_ids.add("root")
    graph.add_edge("root", "child1")
    graph.add_edge("root", "child2")
    graph.add_edge("child1", "grandchild")

    assert "child1" in graph.forward_graph["root"]
    assert "child2" in graph.forward_graph["root"]
    assert "grandchild" in graph.forward_graph["child1"]


def test_to_tree_repair_bad_hierarchy_all():
    """Test repair_bad_hierarchy='all' mode re-evaluates all span placements."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=100.0)
    # child1 is correctly placed under root
    child1 = make_span("child1", "child", parent_id="root", start_time=10.0, end_time=90.0)
    # child2 is incorrectly a sibling of child1 but should be inside child1 based on time
    child2 = make_span("child2", "child", parent_id="root", start_time=20.0, end_time=80.0)
    spans = [root, child1, child2]

    adapter = ToTree(repair_bad_hierarchy="all")
    tree = adapter.adapt(spans)

    # With "all" mode, child2 should be moved under child1 (tighter fit)
    child1_node = next(c for c in tree.children if c.item.span_id == "child1")
    child2_in_child1 = any(c.item.span_id == "child2" for c in child1_node.children)
    assert child2_in_child1


def test_repair_malformed_spans_empty_sequence():
    """Empty sequence should pass through unchanged."""
    adapter = RepairMalformedSpans()
    result = adapter.adapt([])
    assert result == []


def test_to_tree_invalid_parent_error_message():
    """Test that invalid parent error message includes helpful guidance."""
    grandchild = make_span("grandchild", "gc", parent_id="missing-parent", start_time=2.0, end_time=3.0)
    spans = [grandchild]

    adapter = ToTree()

    with pytest.raises(ValueError, match="RepairMalformedSpans"):
        adapter.adapt(spans)


def test_default_span_order_with_tied_values():
    """Spans with identical ordering keys should maintain stable sort."""
    span1 = make_span("s1", "span", parent_id=None, start_time=1.0, end_time=2.0, sequence_id=0)
    span2 = make_span("s2", "span", parent_id=None, start_time=1.0, end_time=2.0, sequence_id=0)
    span3 = make_span("s3", "span", parent_id=None, start_time=1.0, end_time=2.0, sequence_id=0)
    spans = [span1, span2, span3]

    # Default order should be deterministic
    order1 = [default_span_order(s) for s in spans]
    order2 = [default_span_order(s) for s in spans]
    assert order1 == order2


def test_to_tree_depth_based_parent_selection():
    """Deeper eligible parents should be preferred over shallower ones.

    This verifies the fix for the depth sorting in _find_eligible_parents.
    """
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=100.0)
    # Two nested containers with identical durations
    container1 = make_span("container1", "c1", parent_id="root", start_time=10.0, end_time=90.0)
    container2 = make_span("container2", "c2", parent_id="container1", start_time=10.0, end_time=90.0)
    # Dangling span that fits in both containers
    dangling = make_span("dangling", "d", parent_id=None, start_time=20.0, end_time=80.0)
    spans = [root, container1, container2, dangling]

    adapter = ToTree(repair_bad_hierarchy="dangling")
    tree = adapter.adapt(spans)

    # Dangling should be placed under container2 (deeper) rather than container1
    container1_node = next(c for c in tree.children if c.item.span_id == "container1")
    container2_node = next(c for c in container1_node.children if c.item.span_id == "container2")
    dangling_in_container2 = any(c.item.span_id == "dangling" for c in container2_node.children)
    assert dangling_in_container2, "Dangling span should be placed under the deeper container"


def test_to_tree_repair_bad_hierarchy_prefers_shorter_duration():
    """When multiple parents are eligible, prefer shorter duration (tighter fit)."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=100.0)
    # Wide container (duration 80)
    wide = make_span("wide", "wide", parent_id="root", start_time=10.0, end_time=90.0)
    # Narrow container (duration 40) - should be preferred
    narrow = make_span("narrow", "narrow", parent_id="root", start_time=25.0, end_time=65.0)
    # Dangling span that fits in both
    dangling = make_span("dangling", "d", parent_id=None, start_time=30.0, end_time=60.0)
    spans = [root, wide, narrow, dangling]

    adapter = ToTree(repair_bad_hierarchy="dangling")
    tree = adapter.adapt(spans)

    # Dangling should be placed under narrow (shorter duration = tighter fit)
    narrow_node = next(c for c in tree.children if c.item.span_id == "narrow")
    dangling_in_narrow = any(c.item.span_id == "dangling" for c in narrow_node.children)
    assert dangling_in_narrow, "Dangling span should be placed under the tighter-fitting container"


def test_repair_nesting_deep_hierarchy():
    """Test nesting repair with deeply nested hierarchy."""
    # Create hierarchy where child times exceed parent times at multiple levels
    root = make_span("root", "root", parent_id=None, start_time=5.0, end_time=15.0)
    child = make_span("child", "child", parent_id="root", start_time=4.0, end_time=16.0)
    grandchild = make_span("grandchild", "gc", parent_id="child", start_time=3.0, end_time=17.0)
    great_grandchild = make_span("great-gc", "ggc", parent_id="grandchild", start_time=2.0, end_time=18.0)
    spans = [root, child, grandchild, great_grandchild]

    adapter = RepairMalformedSpans(ensure_proper_nesting=True)
    result = adapter.adapt(spans)

    # All ancestors should be expanded to contain descendants
    root_result = next(s for s in result if s.span_id == "root")
    assert root_result.start_time == 2.0
    assert root_result.end_time == 18.0


def test_to_adapting_spans_empty_sequence():
    """Empty sequence should return empty AdaptingSequence."""
    adapter = ToAdaptingSpans()
    result = adapter.adapt([])
    assert len(result) == 0


def test_tree_like_graph_move_subtree_to_same_parent():
    """Moving subtree to the same parent should be a no-op."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child = make_span("child", "child", parent_id="root", start_time=1.0, end_time=9.0)

    graph = _TreeLikeGraph.from_spans([root, child])

    # Move child to root (its current parent)
    graph.move_subtree("child", "root")

    # Should still have the same structure
    assert graph.parent_map["child"] == "root"
    assert "child" in graph.forward_graph["root"]


def test_repair_malformed_spans_uses_max_time_for_missing():
    """Missing times should be filled with max time from other spans."""
    span_with_times = make_span("s1", "span", parent_id=None, start_time=5.0, end_time=10.0)
    span_missing_start = Span.from_attributes(
        rollout_id="r1",
        attempt_id="a1",
        sequence_id=0,
        trace_id="t1",
        span_id="s2",
        parent_id=None,
        name="span",
        attributes={},
        start_time=None,
        end_time=7.0,
    )
    span_missing_end = Span.from_attributes(
        rollout_id="r1",
        attempt_id="a1",
        sequence_id=0,
        trace_id="t1",
        span_id="s3",
        parent_id=None,
        name="span",
        attributes={},
        start_time=3.0,
        end_time=None,
    )
    spans = [span_with_times, span_missing_start, span_missing_end]

    adapter = RepairMalformedSpans()
    result = adapter.adapt(spans)

    # Max time across all spans is 10.0
    s2_result = next(s for s in result if s.span_id == "s2")
    s3_result = next(s for s in result if s.span_id == "s3")
    assert s2_result.start_time == 10.0  # Filled with max
    assert s3_result.end_time == 10.0  # Filled with max


def test_to_tree_no_repair_needed():
    """Test that properly structured spans don't get modified."""
    root = make_span("root", "root", parent_id=None, start_time=0.0, end_time=10.0)
    child1 = make_span("child1", "child", parent_id="root", start_time=1.0, end_time=4.0)
    child2 = make_span("child2", "child", parent_id="root", start_time=5.0, end_time=9.0)
    grandchild = make_span("grandchild", "gc", parent_id="child1", start_time=2.0, end_time=3.0)
    spans = [root, child1, child2, grandchild]

    adapter = ToTree()
    tree = adapter.adapt(spans)

    # Verify structure is preserved
    assert tree.item.span_id == "root"
    assert len(tree.children) == 2
    child1_node = next(c for c in tree.children if c.item.span_id == "child1")
    assert len(child1_node.children) == 1
    assert child1_node.children[0].item.span_id == "grandchild"
