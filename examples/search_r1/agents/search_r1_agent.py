# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from collections.abc import Sequence
from typing import Any, TypedDict, cast

import httpx

from examples.search_r1.agents.qa_em import compute_score_em

log = logging.getLogger(__name__)

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/scripts/data_process/nq_search.py
INSTRUCTION_FORMAT = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you "
    "get new information. After reasoning, if you find you lack some knowledge, you can call a search engine "
    "by <search> query </search> and it will return the top searched results between <information> and "
    "</information>. You can search as many times as your want. If you find no further external knowledge "
    "needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: "
)

DEFAULT_RETRIEVAL_URL = "http://127.0.0.1:8000/retrieve"
DEFAULT_MAX_TURNS = 4
DEFAULT_MAX_TOKENS = 500
DEFAULT_SEARCH_TOPK = 3


class Document(TypedDict, total=False):
    title: str
    text: str
    contents: str


class RetrievalItem(TypedDict, total=False):
    document: Document
    score: float


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_golden_answers(raw: str) -> list[str]:
    """Parse env-mapped golden answers from JSON or a plain string."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]

    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return [str(value)]


def postprocess_response(response: str) -> str:
    """Process responses to stop at a search or answer operation."""
    if "</search>" in response:
        return response.split("</search>")[0] + "</search>"
    if "</answer>" in response:
        return response.split("</answer>")[0] + "</answer>"
    return response


def extract_action(response: str) -> tuple[str | None, str]:
    """Extract the first Search-R1 action tag and content from a model response."""
    pattern = r"<(search|answer)>(.*?)</\1>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1), match.group(2).strip()
    return None, ""


def passages2string(retrieval_result: list[RetrievalItem]) -> str:
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        document = doc_item.get("document", {})
        content = document.get("contents")
        if content:
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
        else:
            title = document.get("title", "")
            text = document.get("text", "")
        format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
    return format_reference


async def retrieve_doc(
    query: str,
    *,
    retrieval_url: str = DEFAULT_RETRIEVAL_URL,
    topk: int = DEFAULT_SEARCH_TOPK,
    timeout: float = 30.0,
) -> str:
    payload: dict[str, Any] = {"queries": [query], "topk": topk, "return_scores": True}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(retrieval_url, json=payload)
        response.raise_for_status()
        json_resp = response.json()
    retrieval_result = cast(list[RetrievalItem], json_resp["result"][0])
    return passages2string(retrieval_result)


async def execute_response(
    response: str,
    *,
    retrieval_url: str = DEFAULT_RETRIEVAL_URL,
    topk: int = DEFAULT_SEARCH_TOPK,
    do_search: bool = True,
) -> str:
    """Execute a Search-R1 text action and return environment feedback."""
    action, content = extract_action(response)
    if action == "answer":
        return ""
    if action == "search":
        search_result = await retrieve_doc(content, retrieval_url=retrieval_url, topk=topk) if do_search else ""
        return f"\n\n<information>{search_result}</information>\n\n"
    return (
        "\nMy previous action is invalid. If I want to search, I should put the query between <search> and "
        "</search>. If I want to give the final answer, I should put the answer between <answer> and "
        "</answer>. Let me try again.\n"
    )


async def call_llm(
    client: Any,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    response = await client.chat.completions.create(
        model="auto",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def post_reward(event_url: str, agl_key: str, reward: float, reason: str) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            event_url,
            json={
                "event_type": "reward",
                "data": {"value": reward, "source": "agent", "reason": reason},
            },
            headers={"Authorization": f"Bearer {agl_key}"},
        )
        response.raise_for_status()


class SearchR1Agent:
    """Search-R1 agent for agl-lite local runner mode."""

    async def run(self) -> None:
        setup_logging()

        question = os.environ["QUESTION"]
        answer_list = parse_golden_answers(os.environ["GOLDEN_ANSWERS"])
        agl_key = os.environ["AGL_KEY"]
        event_url = os.environ["AGL_EVENT_URL"]
        openai_base_url = os.environ["AGL_OPENAI_BASE_URL"]

        retrieval_url = os.environ.get("SEARCH_R1_RETRIEVAL_URL", DEFAULT_RETRIEVAL_URL)
        topk = int(os.environ.get("SEARCH_R1_TOPK", str(DEFAULT_SEARCH_TOPK)))
        max_turns = int(os.environ.get("SEARCH_R1_MAX_TURNS", str(DEFAULT_MAX_TURNS)))
        max_tokens = int(os.environ.get("SEARCH_R1_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        train_temperature = float(os.environ.get("SEARCH_R1_TEMPERATURE", "1.0"))

        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=openai_base_url, api_key=agl_key, max_retries=6)
        prompt = INSTRUCTION_FORMAT + question
        messages = [{"role": "user", "content": prompt}]
        rollout_content = ""
        finished = False
        start_time = time.time()

        log.info("Search-R1 question=%r answers=%r max_turns=%d", question, answer_list, max_turns)
        for turn_id in range(1, max_turns + 1):
            turn_response = await call_llm(
                client,
                messages,
                temperature=train_temperature,
                max_tokens=max_tokens,
            )
            valid_turn_response = postprocess_response(turn_response)
            rollout_content += valid_turn_response
            messages.append({"role": "assistant", "content": valid_turn_response})
            turn_env_feedback = await execute_response(valid_turn_response, retrieval_url=retrieval_url, topk=topk)
            if not turn_env_feedback:
                finished = True
                log.info("turn=%d finished response=%r", turn_id, valid_turn_response)
                break
            rollout_content += turn_env_feedback
            messages.append({"role": "user", "content": turn_env_feedback})
            log.info("turn=%d response=%r env_feedback_chars=%d", turn_id, valid_turn_response, len(turn_env_feedback))

        if not finished:
            turn_response = await call_llm(
                client,
                messages,
                temperature=train_temperature,
                max_tokens=max_tokens,
            )
            rollout_content += turn_response
            messages.append({"role": "assistant", "content": turn_response})
            log.info("last_turn response=%r", turn_response)

        reward = float(compute_score_em(rollout_content, answer_list))
        reason = "em_match" if reward > 0 else "em_miss"
        await post_reward(event_url, agl_key, reward, reason)

        log.info("Search-R1 reward=%.3f reason=%s elapsed=%.2fs", reward, reason, time.time() - start_time)


if __name__ == "__main__":
    asyncio.run(SearchR1Agent().run())
