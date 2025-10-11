# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import inspect
import logging
from typing import (
    Annotated,
    Any,
    Dict,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, Field

from .core import AttemptedRollout

logger = logging.getLogger(__name__)


__all__ = [
    "Resource",
    "LLM",
    "ProxyLLM",
    "PromptTemplate",
    "ResourceUnion",
    "NamedResources",
    "ResourcesUpdate",
]


class Resource(BaseModel):
    """
    Base class for all tunable resources.
    """

    resource_type: Any


class LLM(Resource):
    """
    Provide an LLM endpoint and model name as a resource.

    Attributes:
        endpoint (str): The URL of the LLM API endpoint.
        model (str): The identifier for the model to be used (e.g., 'gpt-4o').
        sampling_parameters (SamplingParameters): A dictionary of hyperparameters
            for model inference, such as temperature, top_p, etc.
    """

    resource_type: Literal["llm"] = "llm"
    endpoint: str
    model: str
    api_key: Optional[str] = None
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)

    def get_base_url(self, *args: Any, **kwargs: Any) -> str:
        """The base_url to put into openai.OpenAI.

        Users are encouraged to use `base_url` to get the LLM endpoint instead of accessing `endpoint` directly.
        """
        return self.endpoint


class ProxyLLM(LLM):
    """Proxy LLM resource that is tailored by `llm_proxy.LLMProxy`."""

    resource_type: Literal["proxy_llm"] = "proxy_llm"  # type: ignore
    _initialized: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Mark initialization as complete after Pydantic finishes setup."""
        super().model_post_init(__context)
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name: str) -> Any:
        """Override to emit a warning when endpoint is accessed directly."""
        # Check if we're accessing endpoint after initialization and not from base_url
        if name == "endpoint":
            try:
                initialized = object.__getattribute__(self, "_initialized")
            except AttributeError:
                initialized = False

            if initialized:
                # Check the call stack to see if we're being called from base_url
                frame = inspect.currentframe()
                if frame and frame.f_back:
                    caller_name = frame.f_back.f_code.co_name
                    if caller_name != "get_base_url":
                        logger.warning(
                            "Accessing 'endpoint' directly on ProxyLLM is discouraged. "
                            "Use 'get_base_url(rollout_id, attempt_id)' instead to get the properly formatted endpoint."
                        )
        return super().__getattribute__(name)

    def with_attempted_rollout(self, rollout: AttemptedRollout) -> LLM:
        """Bake the rollout and attempt id into the endpoint."""
        return LLM(
            endpoint=self.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            model=self.model,
            sampling_parameters=self.sampling_parameters,
            api_key=self.api_key,
        )

    def get_base_url(self, rollout_id: Optional[str], attempt_id: Optional[str]) -> str:
        if rollout_id is None and attempt_id is None:
            return self.endpoint

        if not (isinstance(rollout_id, str) and isinstance(attempt_id, str)):
            raise ValueError("rollout_id and attempt_id must be strings or all be empty")

        prefix = self.endpoint
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        if prefix.endswith("/v1"):
            prefix = prefix[:-3]
            has_v1 = True
        else:
            has_v1 = False
        # Now the prefix should look like "http://localhost:11434"

        # Append the rollout and attempt id to the prefix
        prefix = prefix + f"/rollout/{rollout_id}/attempt/{attempt_id}"
        if has_v1:
            prefix = prefix + "/v1"
        return prefix


class PromptTemplate(Resource):
    """
    A prompt template as a resource.

    Attributes:
        template (str): The template string. The format depends on the engine.
        engine (Literal['jinja', 'f-string', 'poml']): The templating engine
            to use for rendering the prompt. I imagine users can use their own
            customized engines, but algos can only well operate on a subset of them.
    """

    resource_type: Literal["prompt_template"] = "prompt_template"
    template: str
    engine: Literal["jinja", "f-string", "poml"]

    def format(self, **kwargs: Any) -> str:
        """Format the prompt template with the given kwargs."""
        if self.engine == "f-string":
            return self.template.format(**kwargs)
        else:
            raise NotImplementedError(
                "Formatting prompt templates for non-f-string engines with format() helper is not supported yet."
            )


# Use discriminated union for proper deserialization
# TODO: migrate to use a registry
ResourceUnion = Annotated[Union[LLM, ProxyLLM, PromptTemplate], Field(discriminator="resource_type")]
NamedResources = Dict[str, ResourceUnion]
"""
A dictionary-like class to hold named resources.

Example:
    resources: NamedResources = {
        'main_llm': LLM(
            endpoint="http://localhost:8080",
            model="llama3",
            sampling_parameters={'temperature': 0.7, 'max_tokens': 100}
        ),
        'system_prompt': PromptTemplate(
            template="You are a helpful assistant.",
            engine='f-string'
        )
    }
"""


class ResourcesUpdate(BaseModel):
    """
    A resource update message to be sent from the server to clients.

    This message contains a dictionary of resources that clients should use
    for subsequent tasks. It is used to update the resources available to
    clients dynamically.
    """

    resources_id: str
    resources: NamedResources
