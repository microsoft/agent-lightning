# Copyright (c) Microsoft. All rights reserved.

import asyncio
import os
import re

import httpx
from math_verify import LatexExtractionConfig, parse, verify
from openai import AsyncOpenAI


def extract_final_answer(text):
    text = text.rsplit("</think>", 1)[-1]
    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    return None


def score_answer(completion, expected):
    answer = extract_final_answer(completion)
    if answer is None:
        return 0.0
    config = [LatexExtractionConfig()]
    gold = parse(r"\boxed{" + expected + "}", extraction_config=config, fallback_mode="no_fallback")
    prediction = parse(r"\boxed{" + answer + "}", extraction_config=config, fallback_mode="no_fallback")
    return float(verify(gold, prediction))


class Agent:
    async def run(self):
        key = os.environ["AGL_KEY"]
        question = os.environ["QUESTION"]
        instruction = r"Solve the problem step by step and put your final answer in \boxed{...}."
        async with AsyncOpenAI(
            base_url=os.environ["AGL_OPENAI_BASE_URL"], api_key=key, timeout=3600, max_retries=720
        ) as client:
            response = await client.chat.completions.create(
                model="auto",
                messages=[{"role": "user", "content": f"{question}\n\n{instruction}"}],
                n=1,
                temperature=1.0,
                max_tokens=int(os.environ.get("AGL_MAX_TOKENS", "8192")),
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
        choice = response.choices[0]
        reward = (
            score_answer(choice.message.content or "", os.environ["RESULT"]) if choice.finish_reason == "stop" else 0.0
        )
        async with httpx.AsyncClient() as client:
            result = await client.post(
                os.environ["AGL_EVENT_URL"],
                json={"event_type": "reward", "data": {"value": reward}},
                headers={"Authorization": f"Bearer {key}"},
                timeout=10,
            )
            result.raise_for_status()


if __name__ == "__main__":
    asyncio.run(Agent().run())
