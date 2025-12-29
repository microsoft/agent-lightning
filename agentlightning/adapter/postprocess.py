# Copyright (c) Microsoft. All rights reserved.

"""Post-process the data to make it more suitable for training."""

from __future__ import annotations

from typing import Sequence

from agentlightning.types.adapter import (
    AccumulatedMessages,
    AccumulatedTokenSequence,
    AnnotatedChatCompletionCall,
    TokenInputOutputTriplet,
)

from .base import Adapter


class AccumulateTokenSequence(Adapter[Sequence[TokenInputOutputTriplet], Sequence[AccumulatedTokenSequence]]):
    """Assemble multiple token input-output triplets into accumulated token sequences."""

    def adapt(self, source: Sequence[TokenInputOutputTriplet]) -> Sequence[AccumulatedTokenSequence]: ...


class AccumulateMessages(Adapter[Sequence[AnnotatedChatCompletionCall], Sequence[AccumulatedMessages]]):
    """Assemble multiple token input-output triplets into accumulated chat messages."""

    def adapt(self, source: Sequence[AnnotatedChatCompletionCall]) -> Sequence[AccumulatedMessages]: ...


class ToTokenInputOutputTriplet(Adapter[Sequence[AnnotatedChatCompletionCall], Sequence[TokenInputOutputTriplet]]):
    """Convert annotated chat completion calls to token input-output triplets."""

    def adapt(self, source: Sequence[AnnotatedChatCompletionCall]) -> Sequence[TokenInputOutputTriplet]: ...
