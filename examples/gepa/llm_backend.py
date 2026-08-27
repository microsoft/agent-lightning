# Copyright (c) Microsoft. All rights reserved.

"""Centralized LLM provider logic for the room-booking example.

Supports three backends selectable via the ``LLM_PROVIDER`` env var or CLI
``--provider`` flag:

- ``azure_entra`` (default) — Azure OpenAI with Entra ID / ``DefaultAzureCredential``
- ``azure_key`` — Azure OpenAI with a plain API key
- ``openai`` — OpenAI (or any OpenAI-compatible endpoint)

Usage::

    from llm_backend import get_provider, make_client, build_reflection_config, get_model_names

    provider = get_provider()          # reads LLM_PROVIDER env var
    client = make_client(provider)     # returns OpenAI or AzureOpenAI
    model, grader = get_model_names(provider)
    reflection_model, reflection_kwargs = build_reflection_config(provider)
"""

import logging
import os
from typing import Any, Dict, Literal, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

LLMProvider = Literal["azure_entra", "azure_key", "openai"]
VALID_PROVIDERS: Tuple[str, ...] = ("azure_entra", "azure_key", "openai")


def get_provider(override: str | None = None) -> LLMProvider:
    """Resolve the LLM provider from an explicit override or ``LLM_PROVIDER`` env var.

    Defaults to ``azure_entra`` when neither is set.
    """
    raw = override or os.environ.get("LLM_PROVIDER", "azure_entra")
    if raw not in VALID_PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER '{raw}'. Choose from: {', '.join(VALID_PROVIDERS)}")
    return raw  # type: ignore[return-value]


def get_model_names(provider: LLMProvider) -> Tuple[str, str]:
    """Return ``(model, grader_model)`` deployment/model names for the given provider."""
    if provider in ("azure_entra", "azure_key"):
        model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-nano")
        grader = os.environ.get("AZURE_OPENAI_GRADER_DEPLOYMENT", "gpt-4.1-mini")
    else:
        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
        grader = os.environ.get("OPENAI_GRADER_MODEL", "gpt-4.1-mini")
    return model, grader


def make_client(provider: LLMProvider) -> OpenAI:
    """Create an OpenAI-compatible client for the given provider.

    Returns an `OpenAI` instance (the base class of `AzureOpenAI`) so callers
    don't need to branch on the provider.
    """
    if provider == "azure_entra":
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return AzureOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )

    if provider == "azure_key":
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
        )

    # provider == "openai"
    return OpenAI()


def build_reflection_config(provider: LLMProvider) -> Tuple[str, Dict[str, Any]]:
    """Return ``(litellm_model_string, extra_kwargs)`` for GEPA reflection calls.

    The returned values are meant to be passed as ``reflection_model`` and
    ``reflection_model_kwargs`` to [`GEPAConfig`][agentlightning.algorithm.gepa.GEPAConfig].
    """
    _, grader = get_model_names(provider)

    if provider == "azure_entra":
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider

        token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        )
        return f"azure/{grader}", {"azure_ad_token_provider": token_provider}

    if provider == "azure_key":
        return f"azure/{grader}", {"api_key": os.environ["AZURE_OPENAI_API_KEY"]}

    # provider == "openai"
    return grader, {}
