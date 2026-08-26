# Copyright (c) Microsoft. All rights reserved.

"""Minimal multimodal (image) QA agent for local Agent Lightning rollouts."""

# OpenAI is an optional runtime dependency for this example.
# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import os
import re

import httpx


def extract_first_integer(text: str) -> str | None:
    """Extract the first integer from the model output, if any."""
    match = re.search(r"\d+", str(text))
    return match.group(0) if match else None


def post_reward(*, event_url: str, agl_key: str, reward: float) -> None:
    httpx.post(
        event_url,
        json={
            "event_type": "reward",
            "data": {"value": reward},
        },
        headers={"Authorization": f"Bearer {agl_key}"},
        timeout=10.0,
    ).raise_for_status()


class MultimodalQAAgent:
    """Send one image + question to the proxied VLM and score the numeric answer."""

    async def run(self) -> None:
        from openai import AsyncOpenAI

        image_data_url = os.environ["IMAGE"]
        question = os.environ["QUESTION"]
        answer = os.environ["ANSWER"]
        agl_key = os.environ["AGL_KEY"]
        event_url = os.environ["AGL_EVENT_URL"]
        openai_base_url = os.environ["AGL_OPENAI_BASE_URL"]

        client = AsyncOpenAI(
            base_url=openai_base_url,
            api_key=agl_key,
            max_retries=6,
        )
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model="auto",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                temperature=1.0,
                max_tokens=256,
            ),
            timeout=300.0,
        )
        prediction = extract_first_integer(response.choices[0].message.content or "")
        reward = 1.0 if prediction is not None and prediction == answer else 0.0
        post_reward(event_url=event_url, agl_key=agl_key, reward=reward)
