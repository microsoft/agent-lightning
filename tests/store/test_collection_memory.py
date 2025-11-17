# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Dict, Iterable, List, Literal, Mapping, Sequence, Tuple

import pytest
from pydantic import BaseModel, Field

from agentlightning.store.collection import DequeBasedQueue, DictBasedKeyValue, ListBasedCollection


class SampleItem(BaseModel):
    partition: str
    index: int
    name: str
    status: str
    tags: List[str] = Field(default_factory=list)
    score: float | None = None
    rank: int | None = None
    updated_time: float | None = None
    payload: Dict[str, int] = Field(default_factory=dict)
    metadata: str | None = None


def _build_collection(items: Iterable[SampleItem] = ()) -> ListBasedCollection[SampleItem]:
    return ListBasedCollection(list(items), SampleItem, ("partition", "index"))


@pytest.fixture()
def sample_items() -> List[SampleItem]:
    return [
        SampleItem(
            partition="alpha",
            index=1,
            name="urgent-phase-one",
            status="new",
            tags=["core", "urgent"],
            score=10.5,
            rank=3,
            updated_time=12.0,
            payload={"priority": 10},
            metadata="alpha-start",
        ),
        SampleItem(
            partition="alpha",
            index=2,
            name="phase-two",
            status="running",
            tags=["core"],
            score=5.0,
            rank=2,
            updated_time=None,
            payload={"priority": 5},
            metadata=None,
        ),
        SampleItem(
            partition="alpha",
            index=3,
            name="delayed-phase",
            status="blocked",
            tags=["delayed"],
            score=None,
            rank=5,
            updated_time=15.1,
            payload={"priority": 8},
            metadata="delayed-phase",
        ),
        SampleItem(
            partition="beta",
            index=1,
            name="beta-critical",
            status="new",
            tags=["beta", "urgent"],
            score=8.0,
            rank=1,
            updated_time=7.0,
            payload={"priority": 7},
            metadata="beta critical",
        ),
        SampleItem(
            partition="beta",
            index=2,
            name="beta optional",
            status="done",
            tags=["beta"],
            score=3.0,
            rank=None,
            updated_time=2.0,
            payload={"priority": 1},
            metadata="optional path",
        ),
        SampleItem(
            partition="gamma",
            index=1,
            name="gamma-phase",
            status="running",
            tags=[],
            score=9.5,
            rank=4,
            updated_time=None,
            payload={"priority": 9},
            metadata="gamma-phase data",
        ),
        SampleItem(
            partition="gamma",
            index=2,
            name="gamma-late",
            status="done",
            tags=["late", "core"],
            score=1.0,
            rank=6,
            updated_time=20.0,
            payload={"priority": 2},
            metadata="gamma late entry",
        ),
        SampleItem(
            partition="delta",
            index=1,
            name="delta misc",
            status="archived",
            tags=["misc"],
            score=4.2,
            rank=7,
            updated_time=11.0,
            payload={"priority": 3},
            metadata="delta misc block",
        ),
    ]


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


@pytest.fixture()
def sample_collection(sample_items: Sequence[SampleItem]) -> ListBasedCollection[SampleItem]:
    return _build_collection(sample_items)


def _key_pairs(items: Sequence[SampleItem]) -> List[Tuple[str, int]]:
    return [(item.partition, item.index) for item in items]


def _sorted_pairs(items: Sequence[SampleItem]) -> List[Tuple[str, int]]:
    return sorted(_key_pairs(items))


def test_list_collection_requires_primary_keys(sample_items: Sequence[SampleItem]) -> None:
    with pytest.raises(ValueError):
        ListBasedCollection(list(sample_items), SampleItem, ())


def test_list_collection_primary_keys(sample_collection: ListBasedCollection[SampleItem]) -> None:
    assert tuple(sample_collection.primary_keys()) == ("partition", "index")


def test_list_collection_item_type(sample_collection: ListBasedCollection[SampleItem]) -> None:
    assert sample_collection.item_type() is SampleItem


def test_list_collection_initial_size(
    sample_collection: ListBasedCollection[SampleItem], sample_items: Sequence[SampleItem]
) -> None:
    assert sample_collection.size() == len(sample_items)


def test_list_collection_repr_contains_model_info(sample_collection: ListBasedCollection[SampleItem]) -> None:
    result = repr(sample_collection)
    assert "ListBasedCollection" in result and "SampleItem" in result and str(sample_collection.size()) in result


@pytest.mark.asyncio()
async def test_list_collection_insert_adds_item(sample_collection: ListBasedCollection[SampleItem]) -> None:
    new_item = SampleItem(partition="omega", index=1, name="omega", status="new")
    await sample_collection.insert([new_item])
    assert sample_collection.size() == 9
    result = await sample_collection.get({"partition": {"exact": "omega"}, "index": {"exact": 1}})
    assert result == new_item


@pytest.mark.asyncio()
async def test_list_collection_insert_duplicate_raises(sample_collection: ListBasedCollection[SampleItem]) -> None:
    duplicate = SampleItem(partition="alpha", index=1, name="dup", status="new")
    with pytest.raises(ValueError):
        await sample_collection.insert([duplicate])


@pytest.mark.asyncio()
async def test_list_collection_insert_wrong_type(sample_collection: ListBasedCollection[SampleItem]) -> None:
    class Another(BaseModel):
        partition: str
        index: int

    wrong = Another(partition="omega", index=5)
    with pytest.raises(TypeError):
        await sample_collection.insert([wrong])  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_list_collection_update_existing(sample_collection: ListBasedCollection[SampleItem]) -> None:
    updated = SampleItem(partition="alpha", index=1, name="updated", status="new")
    await sample_collection.update([updated])
    result = await sample_collection.get({"partition": {"exact": "alpha"}, "index": {"exact": 1}})
    assert result == updated


@pytest.mark.asyncio()
async def test_list_collection_update_missing_raises(sample_collection: ListBasedCollection[SampleItem]) -> None:
    missing = SampleItem(partition="omega", index=99, name="missing", status="lost")
    with pytest.raises(ValueError):
        await sample_collection.update([missing])


@pytest.mark.asyncio()
async def test_list_collection_delete_existing(sample_collection: ListBasedCollection[SampleItem]) -> None:
    target = SampleItem(partition="alpha", index=1, name="ignored", status="new")
    await sample_collection.delete([target])
    assert sample_collection.size() == 7
    result = await sample_collection.get({"partition": {"exact": "alpha"}, "index": {"exact": 1}})
    assert result is None


@pytest.mark.asyncio()
async def test_list_collection_delete_missing_raises(sample_collection: ListBasedCollection[SampleItem]) -> None:
    missing = SampleItem(partition="omega", index=3, name="x", status="y")
    with pytest.raises(ValueError):
        await sample_collection.delete([missing])


@pytest.mark.asyncio()
async def test_list_collection_upsert_inserts_when_missing(sample_collection: ListBasedCollection[SampleItem]) -> None:
    created = SampleItem(partition="omega", index=4, name="new", status="queued")
    await sample_collection.upsert([created])
    assert sample_collection.size() == 9
    fetched = await sample_collection.get({"partition": {"exact": "omega"}, "index": {"exact": 4}})
    assert fetched == created


@pytest.mark.asyncio()
async def test_list_collection_upsert_updates_when_existing(sample_collection: ListBasedCollection[SampleItem]) -> None:
    replacement = SampleItem(partition="beta", index=2, name="replacement", status="done")
    await sample_collection.upsert([replacement])
    assert sample_collection.size() == 8
    fetched = await sample_collection.get({"partition": {"exact": "beta"}, "index": {"exact": 2}})
    assert fetched == replacement


@pytest.mark.asyncio()
async def test_list_collection_delete_multiple_items(sample_collection: ListBasedCollection[SampleItem]) -> None:
    await sample_collection.delete(
        [
            SampleItem(partition="alpha", index=1, name="", status=""),
            SampleItem(partition="beta", index=1, name="", status=""),
        ]
    )
    assert sample_collection.size() == 6


@pytest.mark.asyncio()
async def test_list_collection_insert_accepts_tuple_sequence(
    sample_collection: ListBasedCollection[SampleItem],
) -> None:
    extra = (
        SampleItem(partition="tuple", index=1, name="a", status="pending"),
        SampleItem(partition="tuple", index=2, name="b", status="pending"),
    )
    await sample_collection.insert(extra)
    assert sample_collection.size() == 10
    fetched = await sample_collection.query(filter={"partition": {"exact": "tuple"}})
    assert _sorted_pairs(fetched.items) == [("tuple", 1), ("tuple", 2)]


@pytest.mark.asyncio()
async def test_list_collection_query_without_filters_returns_all(
    sample_collection: ListBasedCollection[SampleItem],
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
    sample_collection: ListBasedCollection[SampleItem],
    filters: Dict[str, Dict[str, object]],
    expected: Sequence[Tuple[str, int]],
) -> None:
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
    sample_collection: ListBasedCollection[SampleItem],
    filters: Dict[str, Dict[str, object]],
    filter_logic: Literal["and", "or"],
    expected: Sequence[Tuple[str, int]],
) -> None:
    filter_payload = dict(filters)
    filter_payload["_aggregate"] = filter_logic  # type: ignore[index]
    result = await sample_collection.query(filter=filter_payload)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == sorted(expected)


@pytest.mark.asyncio()
async def test_list_collection_primary_key_prefix_limits_filter_checks(
    sample_items: Sequence[SampleItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = _build_collection(sample_items)
    seen: List[Tuple[str, int]] = []
    original = (  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownVariableType]
        ListBasedCollection._item_matches_filters  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType]
    )

    def tracking(item: SampleItem, filters: object, filter_logic: str) -> bool:
        seen.append((item.partition, item.index))
        return original(item, filters, filter_logic)  # type: ignore[arg-type]

    monkeypatch.setattr(ListBasedCollection, "_item_matches_filters", staticmethod(tracking))  # type: ignore[arg-type]

    filters = {"partition": {"exact": "alpha"}, "index": {"within": {1, 2}}}
    result = await collection.query(filter=filters)  # type: ignore[arg-type]
    assert _sorted_pairs(result.items) == [("alpha", 1), ("alpha", 2)]
    assert set(seen) == {("alpha", 1), ("alpha", 2), ("alpha", 3)}


@pytest.mark.asyncio()
async def test_list_collection_full_primary_key_avoids_tree_scan(
    sample_collection: ListBasedCollection[SampleItem],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0
    original_iter_items = (  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType,reportUnknownVariableType]
        ListBasedCollection._iter_items  # pyright: ignore[reportPrivateUsage,reportUnknownMemberType]
    )

    def tracking(
        self: ListBasedCollection[SampleItem],
        root: Mapping[str, object] | None = None,
        filters: object | None = None,
        filter_logic: str = "and",
    ) -> Iterable[SampleItem]:
        nonlocal call_count
        call_count += 1
        return original_iter_items(self, root, filters, filter_logic)  # type: ignore[arg-type]

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
        ("updated_time", "asc", 4, [("beta", 2), ("beta", 1), ("delta", 1), ("alpha", 1)]),
        ("updated_time", "desc", 4, [("gamma", 1), ("alpha", 2), ("gamma", 2), ("alpha", 3)]),
    ],
)
async def test_list_collection_sorting(
    sample_collection: ListBasedCollection[SampleItem],
    sort_by: str,
    sort_order: str,
    limit: int,
    expected: Sequence[Tuple[str, int]],
) -> None:
    result = await sample_collection.query(sort={"name": sort_by, "order": sort_order}, limit=limit)  # type: ignore[arg-type]
    assert _key_pairs(result.items) == list(expected)


@pytest.mark.asyncio()
async def test_list_collection_sort_by_missing_field_raises(sample_collection: ListBasedCollection[SampleItem]) -> None:
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
    sample_collection: ListBasedCollection[SampleItem],
    limit: int,
    offset: int,
    expected: Sequence[Tuple[str, int]],
    total: int,
) -> None:
    result = await sample_collection.query(filter={"partition": {"exact": "alpha"}}, limit=limit, offset=offset)
    assert _key_pairs(result.items) == list(expected)
    assert result.total == total


@pytest.mark.asyncio()
async def test_list_collection_pagination_with_sort(sample_collection: ListBasedCollection[SampleItem]) -> None:
    result = await sample_collection.query(sort={"name": "name", "order": "asc"}, limit=2, offset=3)
    assert _key_pairs(result.items) == [("delta", 1), ("gamma", 2)]
    assert result.total == 8


@pytest.mark.asyncio()
async def test_list_collection_limit_unbounded_with_sort(sample_collection: ListBasedCollection[SampleItem]) -> None:
    result = await sample_collection.query(sort={"name": "name", "order": "asc"}, limit=-1, offset=6)
    assert _key_pairs(result.items) == [("alpha", 2), ("alpha", 1)]
    assert result.total == 8


@pytest.mark.asyncio()
async def test_list_collection_limit_zero_reports_total(sample_collection: ListBasedCollection[SampleItem]) -> None:
    result = await sample_collection.query(filter={"status": {"exact": "done"}}, limit=0)
    assert result.items == []
    assert result.total == 2


@pytest.mark.asyncio()
async def test_list_collection_offset_beyond_total_returns_empty(
    sample_collection: ListBasedCollection[SampleItem],
) -> None:
    result = await sample_collection.query(filter={"status": {"exact": "done"}}, offset=10)
    assert result.items == []
    assert result.total == 2


@pytest.mark.asyncio()
async def test_list_collection_query_reports_total_with_limit(
    sample_collection: ListBasedCollection[SampleItem],
) -> None:
    result = await sample_collection.query(filter={"partition": {"exact": "alpha"}}, limit=1)
    assert result.total == 3
    assert len(result.items) == 1


@pytest.mark.asyncio()
async def test_list_collection_get_returns_first_match(sample_collection: ListBasedCollection[SampleItem]) -> None:
    item = await sample_collection.get({"status": {"exact": "new"}})
    assert item is not None
    assert (item.partition, item.index) == ("beta", 1)


@pytest.mark.asyncio()
async def test_list_collection_get_returns_none(sample_collection: ListBasedCollection[SampleItem]) -> None:
    result = await sample_collection.get({"partition": {"exact": "missing"}})
    assert result is None


@pytest.mark.asyncio()
async def test_list_collection_get_respects_filter_logic(sample_collection: ListBasedCollection[SampleItem]) -> None:
    filters = {"status": {"exact": "done"}, "tags": {"contains": "urgent"}, "_aggregate": "or"}
    item = await sample_collection.get(filters)  # type: ignore[arg-type]
    assert item is not None
    assert (item.partition, item.index) == ("gamma", 2)


@pytest.mark.asyncio()
async def test_list_collection_get_honors_sort_by(sample_collection: ListBasedCollection[SampleItem]) -> None:
    filters = {"partition": {"exact": "alpha"}}
    item = await sample_collection.get(filters, sort={"name": "rank", "order": "asc"})  # type: ignore[arg-type]
    assert item is not None
    assert (item.partition, item.index) == ("alpha", 2)


@pytest.mark.asyncio()
async def test_list_collection_get_honors_sort_order(sample_collection: ListBasedCollection[SampleItem]) -> None:
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
    assert collection.size() == 20
    await collection.delete(items[20:])
    assert collection.size() == 0


@pytest.mark.asyncio()
async def test_list_collection_query_rejects_unknown_operator(
    sample_collection: ListBasedCollection[SampleItem],
) -> None:
    with pytest.raises(ValueError):
        await sample_collection.query(filter={"status": {"invalid": "x"}})  # type: ignore[arg-type]


@pytest.mark.asyncio()
async def test_list_collection_query_result_type() -> None:
    collection = _build_collection([])
    result = await collection.query(filter=None)
    assert result.items == []
    assert result.offset == 0


class QueueItem(BaseModel):
    idx: int


@pytest.fixture()
def deque_queue() -> DequeBasedQueue[QueueItem]:
    return DequeBasedQueue(QueueItem, [QueueItem(idx=i) for i in range(3)])


def test_deque_queue_initial_size(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert deque_queue.size() == 3


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
    assert deque_queue.size() == 5


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
    assert deque_queue.size() == 3 - min(limit, 3)


@pytest.mark.asyncio()
async def test_deque_queue_dequeue_zero_returns_empty(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    assert await deque_queue.dequeue(0) == []


@pytest.mark.asyncio()
async def test_deque_queue_dequeue_more_than_available(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    result = await deque_queue.dequeue(10)
    assert len(result) == 3
    assert deque_queue.size() == 0


@pytest.mark.asyncio()
async def test_deque_queue_peek_preserves_items(deque_queue: DequeBasedQueue[QueueItem]) -> None:
    snapshot = await deque_queue.peek(2)
    assert [item.idx for item in snapshot] == [0, 1]
    assert deque_queue.size() == 3


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
    assert queue.size() == 2000
    drained = await queue.dequeue(1500)
    assert len(drained) == 1500
    assert queue.size() == 500


@pytest.fixture()
def dict_key_value_data() -> Dict[str, int]:
    return {"alpha": 1, "beta": 2}


@pytest.fixture()
def dict_key_value(dict_key_value_data: Dict[str, int]) -> DictBasedKeyValue[str, int]:
    return DictBasedKeyValue(dict_key_value_data)


@pytest.mark.asyncio()
async def test_dict_key_value_initial_state(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    assert dict_key_value.size() == 2
    assert await dict_key_value.get("alpha") == 1
    assert await dict_key_value.get("missing") is None


@pytest.mark.asyncio()
async def test_dict_key_value_has_handles_presence(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    assert await dict_key_value.has("alpha")
    assert not await dict_key_value.has("gamma")


@pytest.mark.asyncio()
async def test_dict_key_value_set_updates_and_expands(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    await dict_key_value.set("gamma", 3)
    assert dict_key_value.size() == 3
    await dict_key_value.set("alpha", 99)
    assert await dict_key_value.get("alpha") == 99
    assert dict_key_value.size() == 3


@pytest.mark.asyncio()
async def test_dict_key_value_pop_returns_default(dict_key_value: DictBasedKeyValue[str, int]) -> None:
    result = await dict_key_value.pop("beta")
    assert result == 2
    assert dict_key_value.size() == 1
    result = await dict_key_value.pop("missing", 42)
    assert result == 42
    assert dict_key_value.size() == 1


@pytest.mark.asyncio()
async def test_dict_key_value_does_not_mutate_input_mapping(dict_key_value_data: Dict[str, int]) -> None:
    key_value = DictBasedKeyValue(dict_key_value_data)
    await key_value.set("gamma", 3)  # type: ignore[arg-type]
    await key_value.pop("alpha")  # type: ignore[arg-type]
    assert dict_key_value_data == {"alpha": 1, "beta": 2}
