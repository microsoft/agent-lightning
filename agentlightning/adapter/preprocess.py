# Copyright (c) Microsoft. All rights reserved.

"""Span re-organization adapters."""

from __future__ import annotations

from typing import Sequence, TypeVar

from agentlightning.types.adapter import Tree
from agentlightning.types.tracer import Span, SpanLike

from .base import Adapter, SequenceAdapter, Sort

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")


class ToSpans(SequenceAdapter[SpanLike, Span]):

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

    def adapt(self, source: Sequence[Span]) -> Tree[Span]: ...


class ToSortedSpans(Sort[Span]):
    """Sort the spans with sequence ID as the primary key and start time as the secondary key."""

    def __init__(self) -> None:
        super().__init__(key=lambda span: (span.sequence_id, span.start_time))


class RepairTreeHierarchy(Adapter[Tree[Span], Tree[Span]]):
    """Repair the tree hierarchy by ensuring that parent-child relationships are consistent
    with span start and end times. Adding missing parent-child relationships as needed.
    """

    def adapt(self, source: Tree[Span]) -> Tree[Span]: ...
