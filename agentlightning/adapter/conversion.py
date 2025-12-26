# Copyright (c) Microsoft. All rights reserved.

"""Non-opinionated conversion adapters for different data formats, without loss of information."""

from __future__ import annotations

from typing import Generic, Sequence, TypeVar

from agentlightning.types.adapter import (
    AccumulatedTokenSequence,
    AnnotatedChatCompletionCall,
    Annotation,
    ChatCompletionCall,
    TokenInputOutputTriplet,
    Tree,
)
from agentlightning.types.tracer import Span, SpanLike

from .base import Adapter, SequenceAdapter

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

    def __init__(self, repair_hierarchy: bool = True):
        self.repair_hierarchy = repair_hierarchy

    def adapt(self, source: Sequence[Span]) -> Tree[Span]: ...


class ToChatCompletionCalls(Adapter[Sequence[Span], Sequence[ChatCompletionCall]]): ...


class ToAnnotations(Adapter[Sequence[Span], Sequence[Annotation]]): ...


class ToTokenInputOutputTriplet(Adapter[Sequence[AnnotatedChatCompletionCall], Sequence[TokenInputOutputTriplet]]): ...
