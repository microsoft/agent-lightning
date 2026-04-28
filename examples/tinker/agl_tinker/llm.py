# Copyright (c) Microsoft. All rights reserved.

"""LLM proxy utilities for Agent-lightning with Tinker.

This module provides a custom LLM implementation that bridges LiteLLM with Tinker's
sampling client, enabling fine-tuned model serving through Agent-lightning.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Dict, List, Literal, Optional, Type, TypeGuard, TypeVar, cast

import litellm
import tinker
from litellm.llms.custom_llm import CustomLLM
from litellm.types.utils import ChatCompletionMessageToolCall, ChatCompletionTokenLogprob
from litellm.types.utils import ChoiceLogprobs as LitellmChoiceLogprobs
from litellm.types.utils import Choices
from litellm.types.utils import Message as LitellmMessage
from litellm.types.utils import ModelResponse
from litellm.utils import custom_llm_setup
from pydantic import TypeAdapter
from tinker.types import ModelInput, SampleResponse, SamplingParams
from tinker_cookbook.renderers import Message as TinkerMessage
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.renderers import ToolCall as TinkerToolCall
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from transformers import PreTrainedTokenizer

from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store import LightningStore

logger = logging.getLogger(__name__)

T = TypeVar("T")


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix.

    Args:
        prefix: String prefix for the generated ID.

    Returns:
        A unique identifier string.
    """
    return prefix + str(uuid.uuid4())


class TinkerLLM(CustomLLM):
    """LiteLLM provider that proxies Tinker's sampling client.

    The cookbook exposes fine-tuned models through `TinkerTokenCompleter` (a
    lightweight callable). Agent-lightning needs a persistent LiteLLM endpoint,
    so that agent developers can still reuse the same agent code without changes.

    This class rewraps the sampling client to satisfy LiteLLM's `CustomLLM`
    protocol while keeping Tinker's renderer/tokenizer pipeline intact.

    Attributes:
        model_name: The HuggingFace model identifier.
        renderer: Prompt renderer for formatting messages.
        tokenizer: Tokenizer for the model.
        sampling_client: Tinker sampling client for generation.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
        top_p: Nucleus sampling parameter.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        model_name: str,
        renderer: Renderer,
        tokenizer: PreTrainedTokenizer,
        sampling_client: tinker.SamplingClient,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        seed: int = 42,
    ) -> None:
        """Initialize the TinkerLLM."""
        self.model_name = model_name
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.sampling_client = sampling_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.seed = seed

    def update_sampling_client(self, sampling_client: tinker.SamplingClient) -> None:
        """Update the sampling client used for generation.

        Args:
            sampling_client: New Tinker sampling client to use.
        """
        self.sampling_client = sampling_client

    def _validate_messages(self, messages: Any) -> TypeGuard[List[TinkerMessage]]:
        TypeAdapter(List[TinkerMessage]).validate_python(messages)
        # Exception will be raised if validation fails
        return True

    def _validate_role(self, role: str) -> TypeGuard[Literal["assistant", "user", "system", "tool", "function"]]:
        if role not in ["assistant", "user", "system", "tool", "function"]:
            raise ValueError(f"Invalid role: {role}")
        return True

    def _parse_tool_call(self, tool_call: TinkerToolCall) -> ChatCompletionMessageToolCall:
        if set(tool_call.keys()) != {"name", "args"}:
            logger.warning(f"Found unexpected tool call keys: {tool_call.keys()}")
        return ChatCompletionMessageToolCall(
            id=generate_id("tinker-tool-call-"),
            function={
                "name": tool_call["name"],
                "arguments": tool_call["args"],
            },
            type="function",
        )

    def _get_optional_params(
        self,
        kwargs: Dict[str, Any],
        keys: List[str],
        expected_type: Type[T],
        validate_fn: Callable[[T], bool],
        default_value: T,
    ) -> T:
        optional_params = cast(Dict[str, Any], kwargs.get("optional_params", {}))
        if not isinstance(optional_params, dict):  # type: ignore
            raise ValueError(f"Invalid optional params type: {type(optional_params)}")
        for key in keys:
            if key in optional_params:
                value = optional_params[key]
                if not isinstance(value, expected_type):
                    raise ValueError(f"Invalid {key} type: {type(value)}")
                if not validate_fn(value):
                    raise ValueError(f"Invalid {key}. Did not pass validation: {value}")
                return value
        return default_value

    def _prepare_model_input(self, **kwargs: Any) -> ModelInput:
        """LiteLLM messages -> Tinker ModelInput."""
        messages = kwargs.pop("messages", None)
        if self._validate_messages(messages):
            return self.renderer.build_generation_prompt(messages)
        else:
            assert False, "This should never happen"

    def _parse_response(self, model_input: ModelInput, response: SampleResponse) -> ModelResponse:
        """Tinker Response -> LiteLLM Response.

        Extract log probabilities as well.
        """
        choices: List[Choices] = []
        for seq in response.sequences:
            if seq.logprobs is not None:
                token_strings: List[str] = self.tokenizer.batch_decode([token] for token in seq.tokens)  # type: ignore
                # FIXME: This might not be accurate for some corner cases.
                # But it's not actually used in most cases.
                bytes_list: List[List[int]] = [list(token.encode("utf-8")) for token in token_strings]
                logprobs = LitellmChoiceLogprobs(
                    content=[
                        ChatCompletionTokenLogprob(
                            token=token,
                            bytes=bytes,
                            logprob=logprob,
                            top_logprobs=[],
                        )
                        for token, bytes, logprob in zip(token_strings, bytes_list, seq.logprobs)
                    ]
                )
            else:
                logprobs = None

            parsed_response, parse_success = self.renderer.parse_response(seq.tokens)
            if parse_success:
                role = parsed_response["role"]
                if not self._validate_role(role):
                    assert False, "This should never happen"
                content = parsed_response["content"]
                # NOTE(yuge): I thought about adding this to make it more robust to empty responses,
                # but later I found it's a configuration error in my renderer. So I think it's better
                # to just log a warning and go with the default path.
                # if not content:
                #     raise ValueError("Parsed content is empty. Original response: " + str(response))
                if not content:
                    logger.warning("Parsed content is empty. Original response: " + str(response))
                tool_calls = parsed_response.get("tool_calls", None)
                if tool_calls:
                    tool_calls = [self._parse_tool_call(tool_call) for tool_call in tool_calls]
                choices.append(
                    Choices(
                        message=LitellmMessage(role=role, content=content, tool_calls=tool_calls),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
            else:
                logger.warning(f"Failed to parse response: {parsed_response}")
                # Go with the default path
                choices.append(
                    Choices(
                        message=LitellmMessage(role="assistant", content=parsed_response["content"]),
                        finish_reason=seq.stop_reason,
                        logprobs=logprobs,
                        token_ids=seq.tokens,
                    )
                )
        return ModelResponse(
            id=generate_id("tinker-sampling-"), choices=choices, prompt_token_ids=model_input.to_ints()
        )

    async def acompletion(self, **kwargs: Any) -> ModelResponse:  # type: ignore
        """Main entrypoint for LiteLLM to call."""
        max_tokens = self._get_optional_params(
            kwargs, ["max_completion_tokens", "max_tokens"], int, lambda x: x >= 0, self.max_tokens
        )
        temperature = self._get_optional_params(
            kwargs, ["temperature"], float, lambda x: 0.0 <= x <= 2.0, self.temperature
        )
        top_k = self._get_optional_params(kwargs, ["top_k"], int, lambda x: True, self.top_k)
        top_p = self._get_optional_params(kwargs, ["top_p"], float, lambda x: 0.0 <= x <= 1.0, self.top_p)
        seed = self._get_optional_params(kwargs, ["seed"], int, lambda _: True, self.seed)
        model_input = self._prepare_model_input(**kwargs)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            seed=seed,
            stop=self.renderer.get_stop_sequences(),
        )
        result = await self.sampling_client.sample_async(prompt=model_input, sampling_params=params, num_samples=1)
        final_response = self._parse_response(model_input, result)
        return final_response

    def as_model_list(self) -> List[ModelConfig]:
        """Generate model configuration for LiteLLM proxy.

        Returns:
            List containing model configuration dict for LiteLLM.
        """
        return [
            {
                "model_name": self.model_name,
                "litellm_params": {
                    "model": f"agl-tinker/{self.model_name}",
                },
            }
        ]

    def rewrite_litellm_custom_providers(self) -> TinkerLLM:
        """Register this TinkerLLM as a custom provider in LiteLLM.

        !!! warning
            This method modifies the global LiteLLM state, which could interfere with other tests in the
            same process.

        Returns:
            Self for method chaining.
        """
        litellm.custom_provider_map = [{"provider": "agl-tinker", "custom_handler": self}]
        custom_llm_setup()
        return self


def create_llm_proxy(
    model_name: str,
    renderer_name: str,
    port: int = 1899,
    store: Optional[LightningStore] = None,
    add_return_token_ids: bool = True,
) -> LLMProxy:
    """Create an LLMProxy configured for a Tinker-based model.

    The Tinker Cookbook typically hands a `TinkerTokenCompleter` straight to
    the trainer. Here we build the longer chain required by Agent-lightning:
    Tinker sampling client -> `TinkerLLM` custom provider -> LiteLLM -> LLMProxy.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507").
        renderer_name: Renderer type for prompt formatting (e.g., "qwen3", "qwen3_instruct").
        port: Port to expose the LiteLLM proxy. Defaults to 1899.
        store: Optional Lightning store for tracking usage. Defaults to None.
        add_return_token_ids: Whether to add return token ids to the response. Defaults to True.

    Returns:
        Configured LLMProxy instance ready to serve the model.
    """
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(base_model=model_name)

    tokenizer = get_tokenizer(model_name)
    tinker_llm = TinkerLLM(
        model_name=model_name,
        sampling_client=sampling_client,
        renderer=get_renderer(renderer_name, tokenizer),
        tokenizer=tokenizer,
    )
    tinker_llm.rewrite_litellm_custom_providers()
    return LLMProxy(
        port=port,
        store=store,
        model_list=tinker_llm.as_model_list(),
        num_retries=2,
        # Must use thread mode here because otherwise the Tinker sampling client will hang.
        launch_mode="thread",
        # If not adding return token ids, we need to add the opentelemetry callback.
        # Otherwise, we set it to default.
        callbacks=["opentelemetry"] if not add_return_token_ids else None,
        # Lengthened timeout
        litellm_config={
            "router_settings": {
                "timeout": 300,
            }
        },
    )
