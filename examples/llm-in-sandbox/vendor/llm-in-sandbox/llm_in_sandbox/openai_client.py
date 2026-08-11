# Copyright (c) Microsoft. All rights reserved.

"""OpenAI-compatible client helpers."""

from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

PROVIDER_PREFIXES = ("openai/", "hosted_vllm/", "azure/")


def strip_provider_prefix(model_name: str) -> str:
    for prefix in PROVIDER_PREFIXES:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def openai_timeout(default: int = 1200) -> int:
    return int(os.environ.get("OPENAI_TIMEOUT", str(default)))


def create_openai_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int | float | None = None,
) -> OpenAI:
    return OpenAI(
        api_key=api_key or os.environ.get("OPENAI_API_KEY", "dummy"),
        base_url=base_url,
        timeout=timeout,
        max_retries=0,
    )


def create_chat_completion(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: int | float | None = None,
    tools: list[dict[str, Any]] | None = None,
    extra_body: dict[str, Any] | None = None,
) -> Any:
    client = create_openai_client(api_key=api_key, base_url=base_url, timeout=timeout)
    request: dict[str, Any] = {
        "model": strip_provider_prefix(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if tools:
        request["tools"] = tools
    if extra_body:
        request["extra_body"] = extra_body
    return client.chat.completions.create(**request)
