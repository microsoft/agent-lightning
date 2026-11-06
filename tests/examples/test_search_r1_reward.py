# Copyright (c) Microsoft. All rights reserved.

"""Search-R1 rewards must describe the answer action selected by the agent."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import AsyncMock

import openai
import pytest

from examples.search_r1.agents import search_r1_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["chat", "completion"])
@pytest.mark.parametrize(
    ("responses", "max_turns", "retrieval_text", "expected_reward", "expected_reason"),
    [
        (["<answer>London</answer><answer>Paris</answer>"], 2, "", 0.0, "em_miss"),
        (["<answer>Paris</answer><answer>London</answer>"], 2, "", 1.0, "em_match"),
        (["<answer><answer>Paris</answer></answer>"], 2, "", 0.0, "em_miss"),
        (["<search>capital</search>", "I do not know."], 1, "<answer>Paris</answer>", 0.0, "em_miss"),
        (["<search>capital</search>", "<answer>Paris</answer>"], 2, "Paris is the capital.", 1.0, "em_match"),
        (["<search>capital</search>", "<answer>Paris</answer>"], 1, "Paris is the capital.", 1.0, "em_match"),
        (["No action tags."], 2, "", 0.0, "invalid_action"),
        (
            ["<search>capital</search>", "<search>again</search><answer>Paris</answer>"],
            1,
            "Paris is the capital.",
            0.0,
            "em_miss",
        ),
        (
            ["<search>capital</search>", "<answer>London</answer><answer>Paris</answer>"],
            1,
            "Paris is the capital.",
            0.0,
            "em_miss",
        ),
    ],
    ids=[
        "first-answer-miss",
        "first-answer-match",
        "nested-answer-is-literal",
        "retrieval-tags-without-answer",
        "search-then-answer",
        "answer-after-turn-limit",
        "invalid-action",
        "search-before-unused-answer-at-turn-limit",
        "first-answer-at-turn-limit",
    ],
)
async def test_reward_uses_selected_answer(
    monkeypatch: pytest.MonkeyPatch,
    api_type: str,
    responses: list[str],
    max_turns: int,
    retrieval_text: str,
    expected_reward: float,
    expected_reason: str,
) -> None:
    for name, value in {
        "QUESTION": "What is the capital of France?",
        "GOLDEN_ANSWERS": '["Paris"]',
        "AGL_KEY": "test-key",
        "AGL_EVENT_URL": "http://gateway/events",
        "AGL_OPENAI_BASE_URL": "http://gateway/v1",
        "SEARCH_R1_MAX_TURNS": str(max_turns),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: object())
    replies = iter(responses)
    contexts: list[Any] = []

    async def call_llm(_client: object, messages: list[dict[str, str]], **_kwargs: Any) -> str:
        contexts.append(copy.deepcopy(messages))
        return next(replies)

    async def call_completion(_client: object, prompt_ids: list[int], **_kwargs: Any) -> tuple[str, list[int]]:
        contexts.append(list(prompt_ids))
        text = next(replies)
        return text, list(text.encode())

    monkeypatch.setattr(search_r1_agent, "call_llm", call_llm)
    monkeypatch.setattr(search_r1_agent, "call_completion", call_completion)
    monkeypatch.setattr(search_r1_agent, "encode_text", lambda _model, text: list(text.encode()))
    retrieve = AsyncMock(return_value=retrieval_text)
    monkeypatch.setattr(search_r1_agent, "retrieve_doc", retrieve)
    post_reward = AsyncMock()
    monkeypatch.setattr(search_r1_agent, "post_reward", post_reward)

    agent = search_r1_agent.SearchR1Agent() if api_type == "chat" else search_r1_agent.SearchR1CompletionAgent()
    await agent.run()

    post_reward.assert_awaited_once_with("http://gateway/events", "test-key", expected_reward, expected_reason)
    assert len(contexts) == len(responses)
    if len(responses) > 1:
        retrieve.assert_awaited_once()
        feedback = f"\n\n<information>{retrieval_text}</information>\n\n"
        if api_type == "chat":
            assert contexts[1][-2:] == [
                {"role": "assistant", "content": responses[0]},
                {"role": "user", "content": feedback},
            ]
        else:
            assert contexts[1] == contexts[0] + list(responses[0].encode()) + list(feedback.encode())
    else:
        retrieve.assert_not_awaited()
