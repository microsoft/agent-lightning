# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

"""Typed representations of tunable resources shared between Agent Lightning components."""

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
    """Base class for tunable resources distributed to executors."""

    resource_type: Any
    """Alias of the resource type."""


class LLM(Resource):
    """Resource that identifies an LLM endpoint and its configuration."""

    resource_type: Literal["llm"] = "llm"
    endpoint: str
    """The URL of the LLM API endpoint."""
    model: str
    """The identifier for the model to be used (e.g., 'gpt-4o')."""
    api_key: Optional[str] = None
    """Optional secret used to authenticate requests."""
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)
    """A dictionary of hyperparameters for model inference, such as temperature, top_p, etc."""

    def get_base_url(self, *args: Any, **kwargs: Any) -> str:
        """Return the base URL consumed by OpenAI-compatible clients.

        Users are encouraged to use `get_base_url(rollout_id, attempt_id)` to get
        the LLM endpoint instead of accessing `.endpoint` directly.
        """
        return self.endpoint


class ProxyLLM(LLM):
    """LLM resource that rewrites endpoints through [`LLMProxy`][agentlightning.LLMProxy].

    The proxy injects rollout- and attempt-specific routing information into the
    endpoint so that downstream services can attribute requests correctly.
    """

    resource_type: Literal["proxy_llm"] = "proxy_llm"  # type: ignore
    _initialized: bool = False

    def model_post_init(self, __context: Any) -> None:
        """Mark initialization as complete after Pydantic finishes setup."""
        super().model_post_init(__context)
        object.__setattr__(self, "_initialized", True)

    def __getattribute__(self, name: str) -> Any:
        """Emit a warning when `endpoint` is accessed directly after initialization."""
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
        """Bake rollout metadata into a concrete [`LLM`][agentlightning.LLM] instance."""
        return LLM(
            endpoint=self.get_base_url(rollout.rollout_id, rollout.attempt.attempt_id),
            model=self.model,
            sampling_parameters=self.sampling_parameters,
            api_key=self.api_key,
        )

    def get_base_url(self, rollout_id: Optional[str], attempt_id: Optional[str]) -> str:
        """Return the routed endpoint for a specific rollout/attempt pair.

        Args:
            rollout_id: Identifier of the rollout making the request.
            attempt_id: Identifier of the attempt within that rollout.

        Returns:
            Fully qualified endpoint including rollout metadata.

        Raises:
            ValueError: If exactly one of ``rollout_id`` or ``attempt_id`` is provided.
        """
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
    """Resource describing a reusable prompt template."""

    resource_type: Literal["prompt_template"] = "prompt_template"
    template: str
    """The template string. The format depends on the engine."""
    engine: Literal["jinja", "f-string", "poml"]
    """The templating engine to use for rendering the prompt."""

    def format(self, **kwargs: Any) -> str:
        """Format the prompt using keyword arguments.

        !!! warning
            Only the `f-string` engine is supported for now.
        """
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
"""Mapping from resource names to their configured instances.

Examples:
    ```python
    resources: NamedResources = {
        "main_llm": LLM(
            endpoint="http://localhost:8080",
            model="llama3",
            sampling_parameters={"temperature": 0.7, "max_tokens": 100},
        ),
        "system_prompt": PromptTemplate(
            template="You are a helpful assistant.",
            engine="f-string",
        ),
    }
    ```
"""


class ResourcesUpdate(BaseModel):
    """Update payload broadcast to clients when resources change."""

    resources_id: str
    """Identifier used to version the resources."""
    create_time: float
    """Timestamp of the creation time of the resources."""
    update_time: float
    """Timestamp of the last update time of the resources."""
    version: int
    """Version of the resources."""
    resources: NamedResources
    """Mapping of resource names to their definitions."""
