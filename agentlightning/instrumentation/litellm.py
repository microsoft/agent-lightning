# Copyright (c) Microsoft. All rights reserved.

"""LiteLLM instrumentations.

It's unclear whether or not this file is useful.
It seems that LiteLLM owns its own telemetry from their own entrance

[Related documentation](https://docs.litellm.ai/docs/observability/agentops_integration).
"""

from typing import Any, Optional

from litellm.integrations.opentelemetry import OpenTelemetry

__all__ = [
    "instrument_litellm",
    "uninstrument_litellm",
]

original_set_attributes = OpenTelemetry.set_attributes  # type: ignore


def patched_set_attributes(self: Any, span: Any, kwargs: Any, response_obj: Optional[Any]):
    original_set_attributes(self, span, kwargs, response_obj)
    # Add custom attributes
    if response_obj is not None and response_obj.get("prompt_token_ids"):
        span.set_attribute("prompt_token_ids", list(response_obj.get("prompt_token_ids")))
    if response_obj is not None and response_obj.get("response_token_ids"):
        span.set_attribute("response_token_ids", list(response_obj.get("response_token_ids")[0]))


def instrument_litellm():
    """Instrument litellm to capture token IDs."""
    OpenTelemetry.set_attributes = patched_set_attributes


def uninstrument_litellm():
    """Uninstrument litellm to stop capturing token IDs."""
    OpenTelemetry.set_attributes = original_set_attributes
