# Copyright (c) Microsoft. All rights reserved.

"""Filter/aggregate multiple objects into a single object.

Opinionated towards which objects to keep and how to aggregate them.
"""

from __future__ import annotations

import re
from typing import Callable, Generic, Literal, Sequence, Tuple, TypeVar

from agentlightning.types.adapter import (
    AccumulatedMessages,
    AccumulatedTokenSequence,
    AnnotatedChatCompletionCall,
    Annotation,
    ChatCompletionCall,
    TokenInputOutputTriplet,
    Tree,
)
from agentlightning.types.tracer import Span

from .base import Adapter

T_SpanSequence = TypeVar("T_SpanSequence", bound=Sequence[Span])

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")


class CurateChatCompletionCalls(Adapter[Sequence[Span], Sequence[ChatCompletionCall]]):
    """Curate the chat completion calls from the spans."""

    def adapt(self, source: Sequence[Span]) -> Sequence[ChatCompletionCall]: ...


class CurateAnnotations(Adapter[Sequence[Span], Sequence[Annotation]]):
    """Curate the annotations from the spans."""

    def adapt(self, source: Sequence[Span]) -> Sequence[Annotation]: ...


class Filter(Adapter[Sequence[T_from], Sequence[T_to]]):
    """Filter items of type T_from to items of type T_to based on a predicate."""

    def __init__(self, predicate: Callable[[T_from], bool]) -> None:
        self.predicate = predicate

    def adapt(self, source: Sequence[T_from]) -> Sequence[T_to]: ...


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


class AnnotateChatCompletionCalls(
    Adapter[Tuple[Sequence[ChatCompletionCall], Sequence[Annotation]], Sequence[AnnotatedChatCompletionCall]]
):
    """Annotate chat completion calls with the given annotations.

    The intersection of "effective radius" of annotations and chat completion calls is used to determine
    which annotations apply to which chat completion calls.

    If an annotation is not linked to any span, try to use `FillMissingLinks` first to link it to spans.
    """

    def adapt(
        self,
        source: Tuple[Sequence[ChatCompletionCall], Sequence[Annotation]],
    ) -> Sequence[AnnotatedChatCompletionCall]: ...


class AccumulateTokenSequence(Adapter[Sequence[TokenInputOutputTriplet], Sequence[AccumulatedTokenSequence]]):
    """Assemble multiple token input-output triplets into accumulated token sequences."""

    def adapt(self, source: Sequence[TokenInputOutputTriplet]) -> Sequence[AccumulatedTokenSequence]: ...


class AccumulateMessages(Adapter[Sequence[AnnotatedChatCompletionCall], Sequence[AccumulatedMessages]]):
    """Assemble multiple token input-output triplets into accumulated chat messages."""

    def adapt(self, source: Sequence[AnnotatedChatCompletionCall]) -> Sequence[AccumulatedMessages]: ...
