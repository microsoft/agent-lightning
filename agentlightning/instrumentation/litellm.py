from typing import Optional, Any

from litellm.integrations.opentelemetry import OpenTelemetry

# It's unclear whether or not this file is useful
# It seems that LiteLLM owns its own telemetry from their own entrance
# https://docs.litellm.ai/docs/observability/agentops_integration

original_set_attributes = OpenTelemetry.set_attributes


def patched_set_attributes(self, span: Any, kwargs, response_obj: Optional[Any]):
    original_set_attributes(self, span, kwargs, response_obj)
    # Add custom attributes
    if response_obj.get("prompt_token_ids"):
        span.set_attribute(
            "prompt_token_ids", list(response_obj.get("prompt_token_ids"))
        )
    if response_obj.get("response_token_ids"):
        span.set_attribute(
            "response_token_ids", list(response_obj.get("response_token_ids")[0])
        )


def instrument_litellm():
    OpenTelemetry.set_attributes = patched_set_attributes
