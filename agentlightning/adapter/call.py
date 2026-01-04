# Copyright (c) Microsoft. All rights reserved.

"""Handles chat completion / response calls. Extracts them from spans and annotates them with annotations."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

from agentlightning.types.adapter import AnnotatedChatCompletionCall, Annotation, ChatCompletionCall, Tree
from agentlightning.types.tracer import Span

from .base import Adapter


class CurateChatCompletionCalls(Adapter[Sequence[Span], Sequence[ChatCompletionCall]]):
    """Curate the chat completion calls from the spans."""

    def _parse_openai_chat_completion_create(self, span: Union[Span, Tree[Span]]) -> ChatCompletionCall: ...

    def _parse_litellm_request(self, span: Union[Span, Tree[Span]]) -> ChatCompletionCall: ...

    def adapt(self, source: Sequence[Span]) -> Sequence[ChatCompletionCall]: ...


class AnnotateChatCompletionCalls(
    Adapter[Tuple[Sequence[ChatCompletionCall], Sequence[Annotation]], Sequence[AnnotatedChatCompletionCall]]
):
    """Annotate chat completion calls with the given annotations.

    The intersection of "effective radius" of annotations and chat completion calls is used to determine
    which annotations apply to which chat completion calls.

    If an annotation is not linked to any span, try to use `RepairMissingLinks` first to link it to spans.
    """

    def adapt(
        self,
        source: Tuple[Sequence[ChatCompletionCall], Sequence[Annotation]],
    ) -> Sequence[AnnotatedChatCompletionCall]: ...
