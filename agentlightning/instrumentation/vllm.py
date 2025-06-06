from __future__ import annotations

from typing import List

from vllm.entrypoints.openai.protocol import ChatCompletionResponse
import vllm.entrypoints.openai.protocol
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat


class ChatCompletionResponsePatched(ChatCompletionResponse):
    prompt_token_ids: List[int] | None = None
    response_token_ids: List[int] | None = None


original_chat_completion_full_generator = (
    OpenAIServingChat.chat_completion_full_generator
)


async def chat_completion_full_generator(
    self,
    request,
    result_generator,
    request_id: str,
    model_name: str,
    conversation,
    tokenizer,
    request_metadata,
):
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
    vllm.entrypoints.openai.protocol.ChatCompletionResponse = (
        ChatCompletionResponsePatched
    )
    OpenAIServingChat.chat_completion_full_generator = chat_completion_full_generator
