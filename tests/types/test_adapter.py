# Copyright (c) Microsoft. All rights reserved.

"""Tests for Tree, AdaptingSequence, and AdaptingSpan data structures."""

import logging
from collections import UserString
from typing import Any, Dict, List, Optional

import pytest

from agentlightning.types import OtelResource, Span, TraceStatus
from agentlightning.types.adapter import AdaptingSequence, AdaptingSpan, Tree


class SequenceTestString(UserString):
    """Simple string wrapper that implements with_container for BaseAdaptingSequenceItem."""

    def with_container(self, container: Any) -> "SequenceTestString":
        return type(self)(self.data)


class SequenceTestInt(int):
    """Simple int wrapper that implements with_container for BaseAdaptingSequenceItem."""

    def __new__(cls, value: int) -> "SequenceTestInt":
        return int.__new__(cls, value)

    def with_container(self, container: Any) -> "SequenceTestInt":
        return type(self)(int(self))


def s(value: str) -> SequenceTestString:
    """Helper to create SequenceTestString instances."""
    return SequenceTestString(value)


def i(value: int) -> SequenceTestInt:
    """Helper to create SequenceTestInt instances."""
    return SequenceTestInt(value)


def strs(*values: str) -> list[SequenceTestString]:
    """Helper to create lists of SequenceTestString instances."""
    return [s(value) for value in values]


def ints(*values: int) -> list[SequenceTestInt]:
    """Helper to create lists of SequenceTestInt instances."""
    return [i(value) for value in values]


def make_span(
    name: str,
    attributes: Dict[str, Any],
    sequence_id: int,
    *,
    parent_id: Optional[str] = None,
) -> Span:
    """Create a test span with minimal required fields."""
    return Span(
        rollout_id="rollout-id",
        attempt_id="attempt-id",
        sequence_id=sequence_id,
        trace_id=f"trace-{sequence_id}",
        span_id=f"span-{sequence_id}",
        parent_id=parent_id,
        name=name,
        status=TraceStatus(status_code="OK"),
        attributes=attributes,
        events=[],
        links=[],
        start_time=None,
        end_time=None,
        context=None,
        parent=None,
        resource=OtelResource(attributes={}, schema_url=""),
    )


# ============================================================================
# Tree Tests
# ============================================================================


def test_tree_single_node():
    tree = Tree(s("root"), [])
    assert tree.item == "root"
    assert tree.children == []
    assert tree.parent is None


def test_tree_with_children():
    child1 = Tree(s("child1"), [])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])

    assert root.item == "root"
    assert len(root.children) == 2
    assert root.children[0].item == "child1"
    assert root.children[1].item == "child2"


def test_tree_parent_reference():
    grandchild = Tree(s("grandchild"), [])
    child = Tree(s("child"), [grandchild])
    root = Tree(s("root"), [child])

    assert root.parent is None
    assert child.parent is root
    assert grandchild.parent is child


def test_tree_len():
    child1 = Tree(s("child1"), [])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])
    assert len(root) == 3


def test_tree_len_deep():
    grandchild = Tree(s("grandchild"), [])
    child = Tree(s("child"), [grandchild])
    root = Tree(s("root"), [child])
    assert len(root) == 3


def test_tree_getitem_single():
    root = Tree(s("root"), [])
    assert root[0] == "root"


def test_tree_getitem_with_children():
    child1 = Tree(s("child1"), [])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])
    # DFS order: root, child1, child2
    assert root[0] == "root"
    assert root[1] == "child1"
    assert root[2] == "child2"


def test_tree_getitem_slice():
    child1 = Tree(s("child1"), [])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])
    assert root[1:] == ["child1", "child2"]


def test_tree_iter():
    child1 = Tree(s("child1"), [])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])
    assert list(root) == ["root", "child1", "child2"]


def test_tree_traverse_single_node():
    tree = Tree(s("root"), [])
    assert list(tree.traverse()) == ["root"]


def test_tree_traverse_dfs_order():
    #       root
    #      /    \
    #   child1  child2
    #     |
    #  grandchild
    grandchild = Tree(s("grandchild"), [])
    child1 = Tree(s("child1"), [grandchild])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])

    # DFS order: root, child1, grandchild, child2
    assert list(root.traverse()) == ["root", "child1", "grandchild", "child2"]


def test_tree_size_single_node():
    tree = Tree(s("root"), [])
    assert tree.size() == 1


def test_tree_size_with_children():
    grandchild = Tree(s("grandchild"), [])
    child1 = Tree(s("child1"), [grandchild])
    child2 = Tree(s("child2"), [])
    root = Tree(s("root"), [child1, child2])
    assert root.size() == 4


def test_tree_map_single_node():
    tree = Tree(i(1), [])
    mapped = tree.map(lambda x: type(x)(x * 2))
    assert mapped.item == 2
    assert mapped.children == []


def test_tree_map_with_children():
    child1 = Tree(i(2), [])
    child2 = Tree(i(3), [])
    root = Tree(i(1), [child1, child2])

    mapped = root.map(lambda x: type(x)(x * 10))
    assert mapped.item == 10
    assert mapped.children[0].item == 20
    assert mapped.children[1].item == 30


def test_tree_map_preserves_structure():
    grandchild = Tree(s("gc"), [])
    child = Tree(s("c"), [grandchild])
    root = Tree(s("r"), [child])

    mapped = root.map(lambda x: type(x)(x.upper()))
    assert mapped.item == "R"
    assert mapped.children[0].item == "C"
    assert mapped.children[0].children[0].item == "GC"


def test_tree_retain_keeps_root_always():
    tree = Tree(s("root"), [])
    retained = tree.retain(lambda x: False)
    assert retained.item == "root"
    assert retained.children == []


def test_tree_retain_keeps_matching_subtrees():
    #       root
    #      /    \
    #   keep1   drop1
    #     |
    #    drop2
    drop2 = Tree(s("drop2"), [])
    keep1 = Tree(s("keep1"), [drop2])
    drop1 = Tree(s("drop1"), [])
    root = Tree(s("root"), [keep1, drop1])

    # Retain subtrees rooted at nodes containing "keep"
    retained = root.retain(lambda x: "keep" in x)

    assert retained.item == "root"
    assert len(retained.children) == 1
    # keep1 is retained along with its entire subtree
    assert retained.children[0].item == "keep1"


def test_tree_retain_removes_branches_without_matches():
    #         root
    #        /    \
    #     drop1  drop2
    #       |
    #     keep
    keep = Tree(s("keep"), [])
    drop1 = Tree(s("drop1"), [keep])
    drop2 = Tree(s("drop2"), [])
    root = Tree(s("root"), [drop1, drop2])

    retained = root.retain(lambda x: x == "keep")

    assert retained.item == "root"
    # drop1 branch is kept because it leads to "keep"
    assert len(retained.children) == 1
    assert retained.children[0].item == "drop1"


def test_tree_retain_deep_tree():
    #       root
    #         |
    #        a
    #         |
    #        b (keep)
    #         |
    #        c
    c = Tree(s("c"), [])
    b = Tree(s("b"), [c])
    a = Tree(s("a"), [b])
    root = Tree(s("root"), [a])

    retained = root.retain(lambda x: x == "b")
    # When b matches, the entire subtree rooted at b (including c) is retained
    assert list(retained.traverse()) == ["root", "a", "b", "c"]


def test_tree_prune_does_not_remove_root():
    tree = Tree(s("root"), [])
    pruned = tree.prune(lambda x: x == "root")
    assert pruned.item == "root"


def test_tree_prune_removes_matching_children():
    child1 = Tree(s("remove_me"), [])
    child2 = Tree(s("keep_me"), [])
    root = Tree(s("root"), [child1, child2])

    pruned = root.prune(lambda x: x == "remove_me")

    assert pruned.item == "root"
    assert len(pruned.children) == 1
    assert pruned.children[0].item == "keep_me"


def test_tree_prune_removes_subtrees():
    #       root
    #      /    \
    #   remove  keep
    #     |
    #   child_of_remove
    child_of_remove = Tree(s("child_of_remove"), [])
    remove = Tree(s("remove"), [child_of_remove])
    keep = Tree(s("keep"), [])
    root = Tree(s("root"), [remove, keep])

    pruned = root.prune(lambda x: x == "remove")

    assert pruned.item == "root"
    assert len(pruned.children) == 1
    assert pruned.children[0].item == "keep"


def test_tree_prune_recursive():
    #       root
    #         |
    #        keep
    #       /    \
    #    remove  keep2
    keep2 = Tree(s("keep2"), [])
    remove = Tree(s("remove"), [])
    keep = Tree(s("keep"), [remove, keep2])
    root = Tree(s("root"), [keep])

    pruned = root.prune(lambda x: x == "remove")

    assert pruned.item == "root"
    assert pruned.children[0].item == "keep"
    assert len(pruned.children[0].children) == 1
    assert pruned.children[0].children[0].item == "keep2"


# ============================================================================
# AdaptingSequence Tests
# ============================================================================


def test_adapting_sequence_empty():
    seq = AdaptingSequence[Any]([])
    assert len(seq) == 0
    assert list(seq) == []


def test_adapting_sequence_with_items():
    seq = AdaptingSequence(ints(1, 2, 3))
    assert len(seq) == 3
    assert list(seq) == [1, 2, 3]


def test_adapting_sequence_getitem_single():
    seq = AdaptingSequence(strs("a", "b", "c"))
    assert seq[0] == "a"
    assert seq[1] == "b"
    assert seq[2] == "c"


def test_adapting_sequence_getitem_negative_index():
    seq = AdaptingSequence(strs("a", "b", "c"))
    assert seq[-1] == "c"


def test_adapting_sequence_getitem_slice():
    seq = AdaptingSequence(strs("a", "b", "c", "d"))
    assert seq[1:3] == ["b", "c"]


def test_adapting_sequence_iter():
    seq = AdaptingSequence(ints(1, 2, 3))
    result: List[Any] = []
    for item in seq:
        result.append(item)
    assert result == [1, 2, 3]


def test_adapting_sequence_traverse():
    seq = AdaptingSequence(ints(1, 2, 3))
    assert list(seq.traverse()) == [1, 2, 3]


def test_adapting_sequence_size():
    seq = AdaptingSequence(ints(1, 2, 3, 4))
    assert seq.size() == 4


def test_adapting_sequence_get():
    seq = AdaptingSequence(strs("x", "y", "z"))
    assert seq.get(0) == "x"
    assert seq.get(1) == "y"


def test_adapting_sequence_map_empty():
    seq = AdaptingSequence[Any]([])
    mapped = seq.map(lambda x: type(x)(x * 2))
    assert list(mapped) == []


def test_adapting_sequence_map_integers():
    seq = AdaptingSequence(ints(1, 2, 3))
    mapped = seq.map(lambda x: type(x)(x * 2))
    assert list(mapped) == [2, 4, 6]


def test_adapting_sequence_map_strings():
    seq = AdaptingSequence(strs("a", "b", "c"))
    mapped = seq.map(lambda x: type(x)(x.upper()))
    assert list(mapped) == ["A", "B", "C"]


def test_adapting_sequence_map_returns_adapting_sequence():
    seq = AdaptingSequence(ints(1, 2, 3))
    mapped = seq.map(lambda x: x)
    assert isinstance(mapped, AdaptingSequence)


def test_adapting_sequence_retain_all():
    seq = AdaptingSequence(ints(1, 2, 3))
    retained = seq.retain(lambda x: True)
    assert list(retained) == [1, 2, 3]


def test_adapting_sequence_retain_none():
    seq = AdaptingSequence(ints(1, 2, 3))
    retained = seq.retain(lambda x: False)
    assert list(retained) == []


def test_adapting_sequence_retain_some():
    seq = AdaptingSequence(ints(1, 2, 3, 4, 5))
    retained = seq.retain(lambda x: x % 2 == 0)
    assert list(retained) == [2, 4]


def test_adapting_sequence_retain_returns_adapting_sequence():
    seq = AdaptingSequence(ints(1, 2, 3))
    retained = seq.retain(lambda x: True)
    assert isinstance(retained, AdaptingSequence)


def test_adapting_sequence_prune_none():
    seq = AdaptingSequence(ints(1, 2, 3))
    pruned = seq.prune(lambda x: False)
    assert list(pruned) == [1, 2, 3]


def test_adapting_sequence_prune_all():
    seq = AdaptingSequence(ints(1, 2, 3))
    pruned = seq.prune(lambda x: True)
    assert list(pruned) == []


def test_adapting_sequence_prune_some():
    seq = AdaptingSequence(ints(1, 2, 3, 4, 5))
    pruned = seq.prune(lambda x: x % 2 == 0)
    assert list(pruned) == [1, 3, 5]


def test_adapting_sequence_prune_returns_adapting_sequence():
    seq = AdaptingSequence(ints(1, 2, 3))
    pruned = seq.prune(lambda x: False)
    assert isinstance(pruned, AdaptingSequence)


# ============================================================================
# AdaptingSpan Tests
# ============================================================================


def test_adapting_span_from_span_creates_adapting_span():
    span = make_span("test-span", {"key": "value"}, 0)
    adapting_span = AdaptingSpan.from_span(span, data="test-data")

    assert adapting_span.name == "test-span"
    assert adapting_span.attributes == {"key": "value"}
    assert adapting_span.data == "test-data"


def test_adapting_span_from_span_preserves_fields():
    span = make_span("my-span", {"attr": 123}, 5, parent_id="parent-span")
    adapting_span = AdaptingSpan.from_span(span, data={"nested": "data"})

    assert adapting_span.rollout_id == "rollout-id"
    assert adapting_span.attempt_id == "attempt-id"
    assert adapting_span.sequence_id == 5
    assert adapting_span.parent_id == "parent-span"
    assert adapting_span.data == {"nested": "data"}


def test_adapting_span_from_adapting_span_updates_data():
    span = make_span("test-span", {}, 0)
    adapting_span1 = AdaptingSpan.from_span(span, data="original")
    adapting_span2 = AdaptingSpan.from_span(adapting_span1, data="updated")

    assert adapting_span2.data == "updated"


def test_adapting_span_with_data_creates_copy():
    span = make_span("test-span", {"key": "value"}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)
    new_span = adapting_span.with_data("new-data", override="silent")

    assert new_span.data == "new-data"
    assert new_span is not adapting_span


def test_adapting_span_with_data_preserves_other_fields():
    span = make_span("test-span", {"attr": 42}, 3)
    adapting_span = AdaptingSpan.from_span(span, data=None)
    new_span = adapting_span.with_data("new-data", override="silent")

    assert new_span.name == "test-span"
    assert new_span.attributes == {"attr": 42}
    assert new_span.sequence_id == 3


def test_adapting_span_with_data_silent_override():
    span = make_span("test-span", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data="original")
    new_span = adapting_span.with_data("updated", override="silent")

    assert new_span.data == "updated"


def test_adapting_span_with_data_warning_override(caplog: pytest.LogCaptureFixture) -> None:
    span = make_span("test-span", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data="original")

    with caplog.at_level(logging.WARNING):
        new_span = adapting_span.with_data("updated", override="warning")

    assert new_span.data == "updated"
    assert "overwriting" in caplog.text.lower()


def test_adapting_span_with_data_forbidden_override():
    span = make_span("test-span", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data="original")

    with pytest.raises(ValueError, match="forbidden"):
        adapting_span.with_data("updated", override="forbidden")


def test_adapting_span_with_data_none_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    span = make_span("test-span", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)

    with caplog.at_level(logging.WARNING):
        new_span = adapting_span.with_data("updated", override="warning")

    assert new_span.data == "updated"
    assert "overwriting" not in caplog.text.lower()


def test_adapting_span_container_default_none():
    span = make_span("test", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)
    assert adapting_span.container is None


def test_adapting_span_container_can_be_set():
    span = make_span("test", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)
    seq = AdaptingSequence([adapting_span])
    adapting_span = adapting_span.model_copy(update={"container": seq})

    assert adapting_span.container is seq


@pytest.fixture
def tree_of_adapting_spans():
    """Create a tree structure of AdaptingSpans for testing.

    Structure:
        root
       /    \\
    child1  child2
       |
    grandchild
    """
    root_span = make_span("root", {}, 0)
    child1_span = make_span("child1", {}, 1, parent_id="span-0")
    child2_span = make_span("child2", {}, 2, parent_id="span-0")
    grandchild_span = make_span("grandchild", {}, 3, parent_id="span-1")

    grandchild_tree: Tree[AdaptingSpan] = Tree(
        AdaptingSpan.from_span(grandchild_span, data="gc-data"),
        [],
    )
    child1_tree: Tree[AdaptingSpan] = Tree(
        AdaptingSpan.from_span(child1_span, data="c1-data"),
        [grandchild_tree],
    )
    child2_tree: Tree[AdaptingSpan] = Tree(
        AdaptingSpan.from_span(child2_span, data="c2-data"),
        [],
    )
    root_tree: Tree[AdaptingSpan] = Tree(
        AdaptingSpan.from_span(root_span, data="root-data"),
        [child1_tree, child2_tree],
    )

    # Set container references
    root_adapting = root_tree.item.model_copy(update={"container": root_tree})
    child1_adapting = child1_tree.item.model_copy(update={"container": child1_tree})
    child2_adapting = child2_tree.item.model_copy(update={"container": child2_tree})
    grandchild_adapting = grandchild_tree.item.model_copy(update={"container": grandchild_tree})

    return root_adapting, child1_adapting, child2_adapting, grandchild_adapting


def test_adapting_span_children_returns_child_spans(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore
    children = root.children()

    assert len(children) == 2
    assert children[0].name == "child1"
    assert children[1].name == "child2"


def test_adapting_span_children_leaf_node_empty(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore
    assert child2.children() == []


def test_adapting_span_children_raises_without_tree_container():
    span = make_span("test", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)

    with pytest.raises(ValueError, match="container"):
        adapting_span.children()


def test_adapting_span_children_raises_with_non_tree_container():
    span = make_span("test", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)
    adapting_span = adapting_span.model_copy(update={"container": AdaptingSequence([])})

    with pytest.raises(ValueError, match="Tree"):
        adapting_span.children()


def test_adapting_span_parent_span_returns_parent(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore

    parent = child1.parent_span()
    assert parent is not None
    assert parent.name == "root"


def test_adapting_span_parent_span_root_returns_none(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore
    assert root.parent_span() is None


def test_adapting_span_parent_span_raises_without_tree_container():
    span = make_span("test", {}, 0)
    adapting_span = AdaptingSpan.from_span(span, data=None)

    with pytest.raises(ValueError, match="container"):
        adapting_span.parent_span()


def test_adapting_span_siblings_returns_sibling_spans(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore

    siblings = child1.siblings()
    assert len(siblings) == 1
    assert siblings[0].name == "child2"


def test_adapting_span_siblings_only_child_returns_empty(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore
    assert grandchild.siblings() == []


def test_adapting_span_siblings_root_returns_empty(tree_of_adapting_spans: Tree[AdaptingSpan]):
    root, child1, child2, grandchild = tree_of_adapting_spans  # type: ignore
    assert root.siblings() == []
