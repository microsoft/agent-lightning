# Copyright (c) Microsoft. All rights reserved.

"""Prompt identity shared by event deduplication and image alignment."""

from __future__ import annotations

import json
from typing import Any

PromptKey = tuple[tuple[int, ...], tuple[str, ...]]


def extract_image_urls_from_messages(messages: Any) -> list[str]:
    if not isinstance(messages, list):
        return []
    image_urls: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)  # Some clients serialize content parts as JSON text.
            except json.JSONDecodeError:
                continue
        if isinstance(content, list):
            for content_part in content:
                if not isinstance(content_part, dict) or content_part.get("type") != "image_url":
                    continue
                image_url = content_part.get("image_url")
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    image_urls.append(image_url["url"])
    return image_urls


def prompt_key(prompt_token_ids: Any, image_urls: list[str]) -> PromptKey | None:
    """Include ordered image references: placeholder token IDs omit image content."""
    if (
        not isinstance(prompt_token_ids, list)
        or not prompt_token_ids
        or any(type(token_id) is not int for token_id in prompt_token_ids)
    ):
        return None
    return tuple(prompt_token_ids), tuple(image_urls)
