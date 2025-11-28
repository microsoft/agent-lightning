# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Mapping, Sequence, Tuple, Union
from uuid import uuid4

import pytest
from pydantic import BaseModel

import agentlightning.store.collection.memory as memory_module
from agentlightning.store.collection import DequeBasedQueue, DictBasedKeyValue, ListBasedCollection
from agentlightning.store.collection.base import Collection
from agentlightning.store.collection.memory import _item_matches_filters  # pyright: ignore[reportPrivateUsage]
from agentlightning.types import Rollout
from tests.store.conftest import QueueItem, SampleItem

if TYPE_CHECKING:
    from pymongo.asynchronous.database import AsyncDatabase


def _build_collection(items: Iterable[SampleItem] = ()) -> ListBasedCollection[SampleItem]:
    return ListBasedCollection(list(items), SampleItem, ("partition", "index"))


BASE_KEY_ORDER: List[Tuple[str, int]] = [
    ("alpha", 1),
    ("alpha", 2),
    ("alpha", 3),
    ("beta", 1),
    ("beta", 2),
    ("gamma", 1),
    ("gamma", 2),
    ("delta", 1),
]


def _key_pairs(items: Sequence[SampleItem]) -> List[Tuple[str, int]]:
    return [(item.partition, item.index) for item in items]


def _sorted_pairs(items: Sequence[SampleItem]) -> List[Tuple[str, int]]:
    return sorted(_key_pairs(items))


def test_list_collection_requires_primary_keys(sample_items: Sequence[SampleItem]) -> None:
    with pytest.raises(ValueError):
        ListBasedCollection(list(sample_items), SampleItem, ())


@pytest.mark.asyncio()
async def test_list_collection_primary_keys(sample_collection: Collection[SampleItem]) -> None:
    assert tuple(sample_collection.primary_keys()) == ("partition", "index")


@pytest.mark.asyncio()
async def test_list_collection_item_type(sample_collection: Collection[SampleItem]) -> None:
    assert sample_collection.item_type() is SampleItem


@pytest.mark.asyncio()
async def test_list_collection_initial_size(
    sample_collection: Collection[SampleItem], sample_items: Sequence[SampleItem]
) -> None:
    assert (await sample_collection.size()) == len(sample_items)


@pytest.mark.asyncio()
async def test_list_collection_repr_contains_model_info(sample_collection: Collection[SampleItem]) -> None:
    result = repr(sample_collection)
    assert sample_collection.__class__.__name__ in result
    assert "SampleItem" in result
    if isinstance(sample_collection, ListBasedCollection):
        assert str(await sample_collection.size()) in result


@pytest.mark.asyncio()
async def test_list_collection_insert_adds_item(sample_collection: Collection[SampleItem]) -> None:
    new_item = SampleItem(partition="omega", index=1, name="omega", status="new")
    await sample_collection.insert([new_item])
    assert (await sample_collection.size()) == 9
    result = await sample_collection.get({"partition": {"exact": "omega"}, "index": {"exact": 1}})
    assert result == new_item


@pytest.mark.asyncio()
async def test_list_collection_insert_duplicate_raises(sample_collection: Collection[SampleItem]) -> None:
    duplicate = SampleItem(partition="alpha", index=1, name="dup", status="new")
    with pytest.raises(ValueError):
        await sample_collection.insert([duplicate])


@pytest.mark.asyncio()
async def test_list_collection_insert_wrong_type(sample_collection: Collection[SampleItem]) -> None:
    class Another(BaseModel):
        partition: str
        index: int

    wrong = Another(partition="omega", index=5)
    with pytest.raises(TypeError):
        await sample_collection.insert([wrong])  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_list_collection_update_existing(sample_collection: Collection[SampleItem]) -> None:
    updated = SampleItem(partition="alpha", index=1, name="updated", status="new")
    await sample_collection.update([updated])
    result = await sample_collection.get({"partition": {"exact": "alpha"}, "index": {"exact": 1}})
    assert result == updated


@pytest.mark.asyncio()
async def test_list_collection_update_missing_raises(sample_collection: Collection[SampleItem]) -> None:
    missing = SampleItem(partition="omega", index=99, name="missing", status="lost")
    with pytest.raises(ValueError):
        await sample_collection.update([missing])


@pytest.mark.asyncio()
async def test_list_collection_delete_existing(sample_collection: Collection[SampleItem]) -> None:
    target = SampleItem(partition="alpha", index=1, name="ignored", status="new")
    await sample_collection.delete([target])
    assert (await sample_collection.size()) == 7
    result = await sample_collection.get({"partition": {"exact": "alpha"}, "index": {"exact": 1}})
    assert result is None


@pytest.mark.asyncio()
async def test_list_collection_delete_missing_raises(sample_collection: Collection[SampleItem]) -> None:
    missing = SampleItem(partition="omega", index=3, name="x", status="y")
    with pytest.raises(ValueError):
        await sample_collection.delete([missing])


@pytest.mark.asyncio()
async def test_list_collection_upsert_inserts_when_missing(sample_collection: Collection[SampleItem]) -> None:
    created = SampleItem(partition="omega", index=4, name="new", status="queued")
    await sample_collection.upsert([created])
    assert (await sample_collection.size()) == 9
    fetched = await sample_collection.get({"partition": {"exact": "omega"}, "index": {"exact": 4}})
    assert fetched == created


@pytest.mark.asyncio()
async def test_list_collection_upsert_updates_when_existing(sample_collection: Collection[SampleItem]) -> None:
    replacement = SampleItem(partition="beta", index=2, name="replacement", status="done")
    await sample_collection.upsert([replacement])
    assert (await sample_collection.size()) == 8
    fetched = await sample_collection.get({"partition": {"exact": "beta"}, "index": {"exact": 2}})
    assert fetched == replacement


@pytest.mark.asyncio()
async def test_list_collection_delete_multiple_items(sample_collection: Collection[SampleItem]) -> None:
    await sample_collection.delete(
        [
            SampleItem(partition="alpha", index=1, name="", status=""),
            SampleItem(partition="beta", index=1, name="", status=""),
        ]
    )
    assert (await sample_collection.size()) == 6


@pytest.mark.asyncio()
async def test_list_collection_insert_accepts_tuple_sequence(
    sample_collection: Collection[SampleItem],
) -> None:
    extra = (
        SampleItem(partition="tuple", index=1, name="a", status="pending"),
        SampleItem(partition="tuple", index=2, name="b", status="pending"),
    )
    await sample_collection.insert(extra)
    assert (await sample_collection.size()) == 10
    fetched = await sample_collection.query(filter={"partition": {"exact": "tuple"}})
    assert _sorted_pairs(fetched.items) == [("tuple", 1), ("tuple", 2)]


@pytest.mark.asyncio()
async def test_list_collection_query_without_filters_returns_all(
    sample_collection: Collection[SampleItem],
) -> None:
    result = await sample_collection.query()
    assert result.total == 8
    assert len(result.items) == 8


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        pytest.param({"status": {"exact": "new"}}, [("alpha", 1), ("beta", 1)], id="exact-single-field"),
        pytest.param(
            {"partition": {"exact": "alpha"}, "index": {"exact": 2}},
            [("alpha", 2)],
            id="exact-multiple-fields",
        ),
        pytest.param(
            {"status": {"within": {"running", "blocked"}}},
            [("alpha", 2), ("alpha", 3), ("gamma", 1)],
            id="within-set",
        ),
        pytest.param(
            {"partition": {"within": ["gamma", "delta"]}},
            [("gamma", 1), ("gamma", 2), ("delta", 1)],
            id="within-list",
        ),
        pytest.param(
            {"name": {"contains": "phase"}},
            [("alpha", 1), ("alpha", 2), ("alpha", 3), ("gamma", 1)],
            id="contains-substring",
        ),
        pytest.param(
            {"tags": {"contains": "urgent"}},
            [("alpha", 1), ("beta", 1)],
            id="contains-list",
        ),
        pytest.param({"metadata": {"contains": "phase"}}, [("alpha", 3), ("gamma", 1)], id="contains-with-none-values"),
        pytest.param({"tags": {"contains": "missing"}}, [], id="contains-no-match"),
        pytest.param({"partition": {"exact": "delta"}}, [("delta", 1)], id="single-exact-match"),
        pytest.param({"missing": {"exact": "value"}}, [], id="exact-missing-field"),
        pytest.param({"score": {"contains": "phase"}}, [], id="contains-typeerror"),
        pytest.param({"name": {"contains": None}}, list(BASE_KEY_ORDER), id="contains-null-check"),
        pytest.param({"status": {"exact": None}}, list(BASE_KEY_ORDER), id="exact-null-no-filter"),
        pytest.param({"status": {"within": 1}}, [], id="within-non-iterable"),
    ],
)
async def test_list_collection_query_filters(
    sample_collection: Collection[SampleItem],
    filters: Dict[str, Dict[str, object]],
    expected: Sequence[Tuple[str, int]],
    request: pytest.FixtureRequest,
) -> None:
    # Mongo implementation raises ValueError for non-iterable values in within filter
    if request.node.callspec.id == "mongo-within-non-iterable":  # type: ignore
        with pytest.raises(ValueError):
            await sample_collection.query(filter=filters)  # type: ignore[arg-type]
        return

    result = await sample_collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == sorted(expected)
    assert result.total == len(expected)


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("filters", "filter_logic", "expected"),
    [
        (
            {"status": {"exact": "new"}, "tags": {"contains": "beta"}},
            "and",
            [("beta", 1)],
        ),
        (
            {"status": {"exact": "new"}, "tags": {"contains": "beta"}},
            "or",
            [("alpha", 1), ("beta", 1), ("beta", 2)],
        ),
        (
            {"status": {"exact": "done"}, "tags": {"contains": "core"}},
            "and",
            [("gamma", 2)],
        ),
        (
            {"status": {"exact": "done"}, "tags": {"contains": "core"}},
            "or",
            [("alpha", 1), ("alpha", 2), ("beta", 2), ("gamma", 2)],
        ),
    ],
)
async def test_list_collection_filter_logic(
    sample_collection: Collection[SampleItem],
    filters: Dict[str, Dict[str, object]],
    filter_logic: Literal["and", "or"],
    expected: Sequence[Tuple[str, int]],
) -> None:
    filter_payload = dict(filters)
    filter_payload["_aggregate"] = filter_logic  # type: ignore[index]
    result = await sample_collection.query(filter=filter_payload)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == sorted(expected)


@pytest.mark.asyncio()
async def test_list_collection_must_filters_respected_with_or(
    sample_collection: Collection[SampleItem],
) -> None:
    filters = {
        "_aggregate": "or",
        "_must": {"partition": {"exact": "alpha"}},
        "status": {"exact": "done"},
        "tags": {"contains": "urgent"},
    }
    result = await sample_collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == [("alpha", 1)]


@pytest.mark.asyncio()
async def test_list_collection_must_filters_accept_sequence(
    sample_collection: Collection[SampleItem],
) -> None:
    filters = {
        "_aggregate": "or",
        "_must": [
            {"partition": {"exact": "beta"}},
            {"index": {"exact": 2}},
        ],
        "status": {"exact": "new"},
        "tags": {"contains": "beta"},
    }
    result = await sample_collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == [("beta", 2)]


@pytest.mark.asyncio()
async def test_list_collection_must_filters_limit_tree_scan_even_with_or(
    sample_collection: Collection[SampleItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not isinstance(sample_collection, ListBasedCollection):
        pytest.skip("This test is only valid for pure-memory collections")
    seen: List[Tuple[str, int]] = []
    original = _item_matches_filters

    def tracking(
        item: SampleItem,
        filters: object,
        filter_logic: str,
        must_filters: object | None = None,
    ) -> bool:
        seen.append((item.partition, item.index))
        return original(item, filters, filter_logic, must_filters)  # type: ignore[arg-type]

    monkeypatch.setattr(memory_module, "_item_matches_filters", tracking)

    filters = {
        "_aggregate": "or",
        "_must": {"partition": {"exact": "gamma"}},
        "status": {"exact": "done"},
        "tags": {"contains": "urgent"},
    }
    result = await sample_collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == [("gamma", 2)]
    assert set(seen) == {("gamma", 1), ("gamma", 2)}


@pytest.mark.asyncio()
async def test_list_collection_primary_key_prefix_limits_filter_checks(
    sample_items: Sequence[SampleItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = _build_collection(sample_items)
    seen: List[Tuple[str, int]] = []
    original = _item_matches_filters

    def tracking(
        item: SampleItem,
        filters: object,
        filter_logic: str,
        must_filters: object | None = None,
    ) -> bool:
        seen.append((item.partition, item.index))
        return original(item, filters, filter_logic, must_filters)  # type: ignore[arg-type]

    monkeypatch.setattr(memory_module, "_item_matches_filters", tracking)

    filters = {"partition": {"exact": "alpha"}, "index": {"within": {1, 2}}}
    result = await collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == [("alpha", 1), ("alpha", 2)]
    assert set(seen) == {("alpha", 1), ("alpha", 2), ("alpha", 3)}


@pytest.mark.asyncio()
async def test_list_collection_full_primary_key_avoids_tree_scan(
    sample_collection: Collection[SampleItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not isinstance(sample_collection, ListBasedCollection):
        pytest.skip("This test is only valid for pure-memory collections")
    call_count = 0
    original_iter_items = (  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownVariableType]
        ListBasedCollection._iter_items  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType]
    )

    def tracking(
        self: ListBasedCollection[SampleItem],
        root: Mapping[str, object] | None = None,
        filters: object | None = None,
        must_filters: object | None = None,
        filter_logic: str = "and",
    ) -> Iterable[SampleItem]:
        nonlocal call_count
        call_count += 1
        return original_iter_items(self, root, filters, must_filters, filter_logic)  # type: ignore[arg-type]

    monkeypatch.setattr(ListBasedCollection, "_iter_items", tracking)

    filters = {"partition": {"exact": "beta"}, "index": {"exact": 2}}
    result = await sample_collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == [("beta", 2)]
    assert call_count == 0


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("sort_by", "sort_order", "limit", "expected"),
    [
        ("name", "asc", 4, [("beta", 2), ("beta", 1), ("alpha", 3), ("delta", 1)]),
        ("name", "desc", 4, [("alpha", 1), ("alpha", 2), ("gamma", 1), ("gamma", 2)]),
        ("rank", "asc", 4, [("beta", 2), ("beta", 1), ("alpha", 2), ("alpha", 1)]),
        ("rank", "desc", 4, [("delta", 1), ("gamma", 2), ("alpha", 3), ("gamma", 1)]),
        ("score", "asc", 4, [("alpha", 3), ("gamma", 2), ("beta", 2), ("delta", 1)]),
        ("score", "desc", 4, [("alpha", 1), ("gamma", 1), ("beta", 1), ("alpha", 2)]),
        (
            "updated_time",
            "asc",
            4,
            (
                [("alpha", 2), ("gamma", 1), ("beta", 2), ("beta", 1)],
                [("beta", 2), ("beta", 1), ("delta", 1), ("alpha", 1)],
            ),
        ),
        (
            "updated_time",
            "desc",
            4,
            (
                [("gamma", 2), ("alpha", 3), ("alpha", 1), ("delta", 1)],
                [("gamma", 1), ("alpha", 2), ("gamma", 2), ("alpha", 3)],
            ),
        ),
    ],
)
async def test_list_collection_sorting(
    sample_collection: Collection[SampleItem],
    sort_by: str,
    sort_order: str,
    limit: int,
    expected: Union[Sequence[Tuple[str, int]], Tuple[Sequence[Tuple[str, int]], ...]],
) -> None:
    result = await sample_collection.query(sort={"name": sort_by, "order": sort_order}, limit=limit)  # type: ignore[arg-type]
    if isinstance(expected, tuple):
        assert any(_key_pairs(result.items) == list(expected) for expected in expected)
    else:
        assert _key_pairs(result.items) == list(expected)


@pytest.mark.asyncio()
async def test_list_collection_sort_by_missing_field_raises(sample_collection: Collection[SampleItem]) -> None:
    with pytest.raises(ValueError):
        await sample_collection.query(sort={"name": "does_not_exist", "order": "asc"})


@pytest.mark.asyncio()
@pytest.mark.parametrize(
    ("limit", "offset", "expected", "total"),
    [
        (1, 0, [("alpha", 1)], 3),
        (2, 1, [("alpha", 2), ("alpha", 3)], 3),
        (-1, 1, [("alpha", 2), ("alpha", 3)], 3),
        (10, 0, [("alpha", 1), ("alpha", 2), ("alpha", 3)], 3),
        (0, 0, [], 3),
        (1, 10, [], 3),
    ],
)
async def test_list_collection_pagination_without_sort(
    sample_collection: Collection[SampleItem],
    limit: int,
    offset: int,
    expected: Sequence[Tuple[str, int]],
    total: int,
) -> None:
    result = await sample_collection.query(filter={"partition": {"exact": "alpha"}}, limit=limit, offset=offset)
    assert _key_pairs(result.items) == list(expected)
    assert result.total == total


@pytest.mark.asyncio()
async def test_list_collection_pagination_with_sort(sample_collection: Collection[SampleItem]) -> None:
    result = await sample_collection.query(sort={"name": "name", "order": "asc"}, limit=2, offset=3)
    assert _key_pairs(result.items) == [("delta", 1), ("gamma", 2)]
    assert result.total == 8


@pytest.mark.asyncio()
async def test_list_collection_limit_unbounded_with_sort(sample_collection: Collection[SampleItem]) -> None:
    result = await sample_collection.query(sort={"name": "name", "order": "asc"}, limit=-1, offset=6)
    assert _key_pairs(result.items) == [("alpha", 2), ("alpha", 1)]
    assert result.total == 8


@pytest.mark.asyncio()
async def test_list_collection_limit_zero_reports_total(sample_collection: Collection[SampleItem]) -> None:
    result = await sample_collection.query(filter={"status": {"exact": "done"}}, limit=0)
    assert result.items == []
    assert result.total == 2


@pytest.mark.asyncio()
async def test_list_collection_offset_beyond_total_returns_empty(
    sample_collection: Collection[SampleItem],
) -> None:
    result = await sample_collection.query(filter={"status": {"exact": "done"}}, offset=10)
    assert result.items == []
    assert result.total == 2


@pytest.mark.asyncio()
async def test_list_collection_query_reports_total_with_limit(
    sample_collection: Collection[SampleItem],
) -> None:
    result = await sample_collection.query(filter={"partition": {"exact": "alpha"}}, limit=1)
    assert result.total == 3
    assert len(result.items) == 1


@pytest.mark.asyncio()
async def test_list_collection_get_returns_first_match(sample_collection: Collection[SampleItem]) -> None:
    item = await sample_collection.get({"status": {"exact": "new"}})
    assert item is not None
    assert (item.partition, item.index) in [("beta", 1), ("alpha", 1)]


@pytest.mark.asyncio()
async def test_list_collection_get_returns_none(sample_collection: Collection[SampleItem]) -> None:
    result = await sample_collection.get({"partition": {"exact": "missing"}})
    assert result is None


@pytest.mark.asyncio()
async def test_list_collection_get_respects_filter_logic(sample_collection: Collection[SampleItem]) -> None:
    filters = {"status": {"exact": "done"}, "tags": {"contains": "urgent"}, "_aggregate": "or"}
    item = await sample_collection.get(filters)  # type: ignore[arg-type]
    assert item is not None
    assert (item.partition, item.index) in [("gamma", 2), ("alpha", 1)]


@pytest.mark.asyncio()
async def test_list_collection_get_honors_sort_by(sample_collection: Collection[SampleItem]) -> None:
    filters = {"partition": {"exact": "alpha"}}
    item = await sample_collection.get(filters, sort={"name": "rank", "order": "asc"})  # type: ignore[arg-type]
    assert item is not None
    assert (item.partition, item.index) == ("alpha", 2)


@pytest.mark.asyncio()
async def test_list_collection_get_honors_sort_order(sample_collection: Collection[SampleItem]) -> None:
    filters = {"partition": {"exact": "alpha"}}
    item = await sample_collection.get(filters, sort={"name": "rank", "order": "desc"})  # type: ignore[arg-type]
    assert item is not None
    assert (item.partition, item.index) == ("alpha", 3)


@pytest.mark.asyncio()
async def test_list_collection_query_handles_large_dataset() -> None:
    bulk_items = [
        SampleItem(
            partition=f"partition-{i % 5}",
            index=i,
            name=f"bulk-{i}",
            status="bulk",
            score=float(i),
            rank=i,
            updated_time=float(i),
        )
        for i in range(1500)
    ]
    collection = _build_collection(bulk_items)
    result = await collection.query(sort={"name": "index", "order": "asc"}, limit=50, offset=100)
    assert result.total == 1500
    assert len(result.items) == 50
    assert result.items[0].index == 100
    assert result.items[-1].index == 149


@pytest.mark.asyncio()
async def test_list_collection_bulk_delete_and_size() -> None:
    items = [SampleItem(partition="bulk", index=i, name=f"item-{i}", status="bulk") for i in range(40)]
    collection = _build_collection(items)
    await collection.delete(items[:20])
    assert (await collection.size()) == 20
    await collection.delete(items[20:])
    assert (await collection.size()) == 0


@pytest.mark.asyncio()
async def test_list_collection_query_rejects_unknown_operator(
    sample_collection: Collection[SampleItem],
) -> None:
    with pytest.raises(ValueError):
        await sample_collection.query(filter={"status": {"invalid": "x"}})  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_list_collection_query_result_type() -> None:
    collection = _build_collection([])
    result = await collection.query(filter=None)
    assert result.items == []
    assert result.offset == 0


@pytest.mark.asyncio()
async def test_deque_queue_initial_size(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert (await deque_queue.size()) == 3


def test_deque_queue_item_type(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert deque_queue.item_type() is QueueItem


@pytest.mark.asyncio()
async def test_deque_queue_has_detects_members(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert await deque_queue.has(QueueItem(idx=1))
    assert not await deque_queue.has(QueueItem(idx=99))


@pytest.mark.asyncio()
async def test_deque_queue_enqueue_appends_items(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    items = [QueueItem(idx=3), QueueItem(idx=4)]
    returned = await deque_queue.enqueue(items)
    assert returned == items
    assert (await deque_queue.size()) == 5


@pytest.mark.asyncio()
async def test_deque_queue_enqueue_rejects_wrong_type(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    class Wrong(BaseModel):
        idx: int

    with pytest.raises(TypeError):
        await deque_queue.enqueue([Wrong(idx=9)])  # type: ignore[arg-type]


@pytest.mark.asyncio()
@pytest.mark.parametrize("limit", [1, 2, 5])
async def test_deque_queue_dequeue_respects_limit(deque_queue: DequeBasedQueue[QueueItem], limit: int) -> None:
    result = await deque_queue.dequeue(limit)
    assert len(result) == min(limit, 3)
    assert (await deque_queue.size()) == 3 - min(limit, 3)


@pytest.mark.asyncio()
async def test_deque_queue_dequeue_zero_returns_empty(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert await deque_queue.dequeue(0) == []


@pytest.mark.asyncio()
async def test_deque_queue_dequeue_more_than_available(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    result = await deque_queue.dequeue(10)
    assert len(result) == 3
    assert (await deque_queue.size()) == 0


@pytest.mark.asyncio()
async def test_deque_queue_peek_preserves_items(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    snapshot = await deque_queue.peek(2)
    assert [item.idx for item in snapshot] == [0, 1]
    assert (await deque_queue.size()) == 3


@pytest.mark.asyncio()
async def test_deque_queue_peek_zero_returns_empty(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert await deque_queue.peek(0) == []


@pytest.mark.asyncio()
async def test_deque_queue_peek_after_partial_dequeue(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    await deque_queue.dequeue(1)
    snapshot = await deque_queue.peek(2)
    assert [item.idx for item in snapshot] == [1, 2]


@pytest.mark.asyncio()
async def test_deque_queue_handles_large_volume() -> None:
    queue = DequeBasedQueue(QueueItem)
    items = [QueueItem(idx=i) for i in range(2000)]
    await queue.enqueue(items)
    assert (await queue.size()) == 2000
    drained = await queue.dequeue(1500)
    assert len(drained) == 1500
    assert (await queue.size()) == 500


@pytest.mark.asyncio()
async def test_dict_key_value_initial_state(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    assert await dict_key_value.size() == 2
    assert await dict_key_value.get("alpha") == 1
    assert await dict_key_value.get("missing") is None


@pytest.mark.asyncio()
async def test_dict_key_value_has_handles_presence(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    assert await dict_key_value.has("alpha")
    assert not await dict_key_value.has("gamma")


@pytest.mark.asyncio()
async def test_dict_key_value_set_updates_and_expands(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    await dict_key_value.set("gamma", 3)
    assert await dict_key_value.size() == 3
    await dict_key_value.set("alpha", 99)
    assert await dict_key_value.get("alpha") == 99
    assert await dict_key_value.size() == 3


@pytest.mark.asyncio()
async def test_dict_key_value_pop_returns_default(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    result = await dict_key_value.pop("beta")
    assert result == 2
    assert await dict_key_value.size() == 1
    result = await dict_key_value.pop("missing", 42)
    assert result == 42
    assert await dict_key_value.size() == 1


@pytest.mark.asyncio()
async def test_dict_key_value_does_not_mutate_input_mapping(dict_key_value_data: Dict[str, int]) -> None:
    key_value = DictBasedKeyValue(dict_key_value_data)
    await key_value.set("gamma", 3)  # type: ignore[arg-type]
    await key_value.pop("alpha")  # type: ignore[arg-type]
    assert dict_key_value_data == {"alpha": 1, "beta": 2}


@pytest.mark.mongo
@pytest.mark.asyncio()
async def test_mongo_based_sanity_check(temporary_mongo_database: AsyncDatabase[Any]) -> None:
    from agentlightning.store.collection.mongo import (
        MongoBasedCollection,
        MongoBasedKeyValue,
        MongoBasedQueue,
        MongoClientPool,
    )

    async with MongoClientPool(temporary_mongo_database.client) as client_pool:
        collection = MongoBasedCollection[Any](
            client_pool, temporary_mongo_database.name, "test", "test-123", ["rollout_id"], Rollout
        )
        await collection.ensure_collection()

        start_time = time.time()
        await collection.insert(
            [Rollout(rollout_id="test-123", input="test-123", start_time=start_time, status="running")]
        )

        result = await collection.query(filter={"status": {"exact": "running"}})
        assert result.items == [
            Rollout(rollout_id="test-123", input="test-123", start_time=start_time, status="running")
        ]

        rollout_queue = MongoBasedQueue[str](
            client_pool, temporary_mongo_database.name, "rollout_queue", "partition-1", str
        )
        await rollout_queue.ensure_collection()

        await rollout_queue.enqueue(["r1", "r2", "r3"])
        assert await rollout_queue.size() == 3
        assert await rollout_queue.peek(2) == ["r1", "r2"]
        assert await rollout_queue.dequeue(2) == ["r1", "r2"]
        assert await rollout_queue.size() == 1

        span_kv = MongoBasedKeyValue[str, int](
            client_pool, temporary_mongo_database.name, "span_sequence_ids", "partition-1", str, int
        )
        await span_kv.ensure_collection()

        await span_kv.set("span-123", 1)
        assert await span_kv.has("span-123")
        assert await span_kv.get("span-123") == 1
        assert await span_kv.pop("span-123") == 1
        assert not await span_kv.has("span-123")


@pytest.mark.mongo
@pytest.mark.asyncio()
async def test_mongo_ensure_collection_creates_partition_scoped_index(
    temporary_mongo_database: AsyncDatabase[Any],
) -> None:
    from agentlightning.store.collection.mongo import MongoBasedCollection, MongoClientPool

    collection_name = f"ensure-{uuid4().hex}"
    async with MongoClientPool(temporary_mongo_database.client) as client_pool:
        collection = MongoBasedCollection[Any](
            client_pool,
            temporary_mongo_database.name,
            collection_name,
            "partition-ensure",
            ["name", "index"],
            SampleItem,
        )
        await collection.ensure_collection()

        unique_index = None
        async for index in await temporary_mongo_database[collection_name].list_indexes():  # type: ignore
            if index["name"] == "uniq_partition_name_index" and index.get("unique"):  # type: ignore
                unique_index = index  # type: ignore
                break

        assert unique_index is not None, "expected unique partition/index key"
        key_pairs = list(unique_index["key"].items())  # type: ignore
        assert key_pairs == [("partition_id", 1), ("name", 1), ("index", 1)]


@pytest.mark.mongo
@pytest.mark.asyncio()
async def test_mongo_ensure_collection_survives_concurrent_calls(temporary_mongo_database: AsyncDatabase[Any]) -> None:
    from agentlightning.store.collection.mongo import MongoBasedCollection, MongoClientPool

    collection_name = f"ensure-{uuid4().hex}"

    async def ensure_once() -> None:
        async with MongoClientPool(temporary_mongo_database.client) as client_pool:
            collection = MongoBasedCollection(
                client_pool,
                temporary_mongo_database.name,
                collection_name,
                "partition-concurrent",
                ["index"],
                SampleItem,
            )
            await collection.ensure_collection()

    await asyncio.gather(*(ensure_once() for _ in range(20)))

    names = await temporary_mongo_database.list_collection_names()
    assert names.count(collection_name) == 1

    unique_indexes = []
    async for index in await temporary_mongo_database[collection_name].list_indexes():  # type: ignore
        if index["name"].startswith("uniq_partition_"):  # type: ignore
            unique_indexes.append(index["name"])  # type: ignore
    assert unique_indexes == ["uniq_partition_index"]


@pytest.mark.mongo
@pytest.mark.asyncio()
async def test_mongo_ensure_collection_repeats_without_altering_indexes(
    temporary_mongo_database: AsyncDatabase[Any],
) -> None:
    from agentlightning.store.collection.mongo import MongoBasedCollection, MongoClientPool

    collection_name = f"ensure-{uuid4().hex}"
    async with MongoClientPool(temporary_mongo_database.client) as client_pool:
        collection = MongoBasedCollection(
            client_pool, temporary_mongo_database.name, collection_name, "partition-repeat", ["index"], SampleItem
        )
        await collection.ensure_collection()
        await collection.ensure_collection()

        unique_indexes = []
        async for index in await temporary_mongo_database[collection_name].list_indexes():  # type: ignore
            if index["name"].startswith("uniq_partition_"):  # type: ignore
                unique_indexes.append((index["name"], list(index["key"].items())))  # type: ignore

        assert unique_indexes == [("uniq_partition_index", [("partition_id", 1), ("index", 1)])]
