"""Minimal GSM8K agent entrypoint for local agl-lite rollouts."""

# OpenAI is an optional runtime dependency for this example.
# pyright: reportMissingImports=false

from __future__ import annotations

import asyncio
import os
import re

import httpx


def get_tokenizer(model: str) -> object:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def extract_final_answer(text: str) -> str:
    """Extract the final GSM8K answer from model or dataset text."""
    text = str(text).strip()
    boxed_match = re.search(r"###\s*ANSWER:\s*(.+?)(\s*###|$)", text, re.DOTALL | re.IGNORECASE)
    if boxed_match:
        return boxed_match.group(1).strip()
    gsm8k_match = re.search(r"####\s*(.+)$", text, re.DOTALL)
    if gsm8k_match:
        return gsm8k_match.group(1).strip()
    number_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return number_matches[-1].replace(",", "") if number_matches else text


def normalize_answer(text: str) -> str:
    return extract_final_answer(text).replace(",", "").strip()


def encode_prompt(model: str, prompt: str) -> list[int]:
    tokenizer = get_tokenizer(model)
    return tokenizer.encode(prompt, add_special_tokens=False)


def decode_choice_token_ids(choice: object, model: str) -> str:
    token_ids = getattr(choice, "token_ids", None)
    if token_ids is None and hasattr(choice, "model_extra"):
        token_ids = choice.model_extra.get("token_ids")
    if token_ids is None:
        raise ValueError("OpenAI response choice did not include token_ids")
    tokenizer = get_tokenizer(model)
    return tokenizer.decode(token_ids, skip_special_tokens=True)


def build_prompt(question: str) -> str:
    output_format = (
        "Solve the problem step by step. When you are ready, output the final answer surrounded by "
        "three sharps (`###`), in the form of ### ANSWER: <answer> ###."
    )
    return f"{question}\n\n{output_format}"


def post_reward(*, event_url: str, agl_key: str, prediction: str, answer: str) -> None:
    reward = 1.0 if normalize_answer(prediction) == normalize_answer(answer) else 0.0

    httpx.post(
        event_url,
        json={
            "event_type": "reward",
            "data": {"value": reward},
        },
        headers={"Authorization": f"Bearer {agl_key}"},
        timeout=10.0,
    ).raise_for_status()


class ChatAgent:
    async def run(self) -> None:
        from openai import AsyncOpenAI

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
                        "content": build_prompt(question),
                    }
                ],
                temperature=1.0,
                max_tokens=1024,
            ),
            timeout=300.0,
        )
        post_reward(
            event_url=event_url,
            agl_key=agl_key,
            prediction=response.choices[0].message.content or "",
            answer=answer,
        )


class CompletionAgent:
    async def run(self) -> None:
        from openai import AsyncOpenAI

        question = os.environ["QUESTION"]
        answer = os.environ["ANSWER"]
        agl_key = os.environ["AGL_KEY"]
        event_url = os.environ["AGL_EVENT_URL"]
        openai_base_url = os.environ["AGL_OPENAI_BASE_URL"]
        model = os.environ.get("GSM8K_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

        client = AsyncOpenAI(
            base_url=openai_base_url,
            api_key=agl_key,
            max_retries=6,
        )
        response = await asyncio.wait_for(
            client.completions.create(
                model="auto",
                prompt=encode_prompt(model, build_prompt(question)),
                temperature=1.0,
                max_tokens=1024,
            ),
            timeout=300.0,
        )
        post_reward(
            event_url=event_url,
            agl_key=agl_key,
            prediction=decode_choice_token_ids(response.choices[0], model),
            answer=answer,
        )


