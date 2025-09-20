# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import warnings
from typing import Any, List

import vllm.entrypoints.openai.protocol
from vllm.entrypoints.openai.protocol import ChatCompletionResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat


class ChatCompletionResponsePatched(ChatCompletionResponse):
    prompt_token_ids: List[int] | None = None
    response_token_ids: List[int] | None = None


original_chat_completion_full_generator = OpenAIServingChat.chat_completion_full_generator


async def chat_completion_full_generator(
    self: Any,
    request: Any,
    result_generator: Any,
    request_id: str,
    model_name: str,
    conversation: Any,
    tokenizer: Any,
    request_metadata: Any,
) -> Any:
    prompt_token_ids: List[int] | None = None
    response_token_ids: List[List[int]] | None = None

    async def _generate_inceptor():
        nonlocal prompt_token_ids, response_token_ids
        async for res in result_generator:
            yield res
            prompt_token_ids = res.prompt_token_ids
            response_token_ids = [output.token_ids for output in res.outputs]

    response = await original_chat_completion_full_generator(
        self,
        request,
        _generate_inceptor(),
        request_id,
        model_name,
        conversation,
        tokenizer,
        request_metadata,
    )
    response = response.model_copy(
        update={
            "prompt_token_ids": prompt_token_ids,
            "response_token_ids": response_token_ids,
        }
    )

    return response


def instrument_vllm():
    if vllm.entrypoints.openai.protocol.ChatCompletionResponse is ChatCompletionResponsePatched:
        warnings.warn("vllm is already instrumented. Skip the instrumentation.")
        return

    vllm.entrypoints.openai.protocol.ChatCompletionResponse = ChatCompletionResponsePatched
    OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator


def uninstrument_vllm():
    OpenAIServingChat.chat_completion_full_generator = original_chat_completion_full_generator
