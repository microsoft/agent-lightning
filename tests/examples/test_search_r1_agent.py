from __future__ import annotations

import asyncio
from typing import Any

from examples.search_r1.agents import search_r1_agent


def test_postprocess_response_stops_at_search() -> None:
    response = "<think>Need info</think><search>capital of France</search> trailing"

    assert (
        search_r1_agent.postprocess_response(response) == "<think>Need info</think><search>capital of France</search>"
    )


def test_postprocess_response_stops_at_answer() -> None:
    response = "<think>Done</think><answer>Paris</answer> trailing"

    assert search_r1_agent.postprocess_response(response) == "<think>Done</think><answer>Paris</answer>"


def test_extract_action_search_and_answer() -> None:
    assert search_r1_agent.extract_action("before <search>  query  </search>") == ("search", "query")
    assert search_r1_agent.extract_action("before <answer> Paris </answer>") == ("answer", "Paris")
    assert search_r1_agent.extract_action("no action") == (None, "")


def test_passages2string_formats_contents_and_title_text() -> None:
    result = search_r1_agent.passages2string(
        [
            {"document": {"contents": "France\nFrance is a country."}, "score": 1.0},
            {"document": {"title": "Paris", "text": "Paris is the capital."}, "score": 0.5},
        ]
    )

    assert "Doc 1(Title: France) France is a country." in result
    assert "Doc 2(Title: Paris) Paris is the capital." in result


def test_execute_response_wraps_retrieval_feedback(monkeypatch) -> None:
    async def fake_retrieve_doc(query: str, *, retrieval_url: str, topk: int, timeout: float = 30.0) -> str:
        assert query == "capital of France"
        assert retrieval_url == "http://retriever/retrieve"
        assert topk == 2
        return "Doc 1(Title: Paris) Paris is the capital.\n"

    monkeypatch.setattr(search_r1_agent, "retrieve_doc", fake_retrieve_doc)

    feedback = asyncio.run(
        search_r1_agent.execute_response(
            "<search>capital of France</search>",
            retrieval_url="http://retriever/retrieve",
            topk=2,
        )
    )

    assert feedback == "\n\n<information>Doc 1(Title: Paris) Paris is the capital.\n</information>\n\n"


def test_search_r1_agent_posts_reward(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    async def fake_call_llm(client: Any, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        calls.setdefault("messages", []).append(list(messages))
        assert temperature == 0.25
        assert max_tokens == 64
        return "<think>Known</think><answer>Paris</answer>"

    async def fake_post_reward(event_url: str, agl_key: str, reward: float, reason: str) -> None:
        calls["reward"] = {
            "event_url": event_url,
            "agl_key": agl_key,
            "reward": reward,
            "reason": reason,
        }

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            calls["openai"] = kwargs

    monkeypatch.setattr(search_r1_agent, "call_llm", fake_call_llm)
    monkeypatch.setattr(search_r1_agent, "post_reward", fake_post_reward)
    monkeypatch.setitem(__import__("sys").modules, "openai", type("OpenAIModule", (), {"AsyncOpenAI": FakeAsyncOpenAI}))
    monkeypatch.setenv("QUESTION", "What is the capital of France?")
    monkeypatch.setenv("GOLDEN_ANSWERS", '["Paris"]')
    monkeypatch.setenv("AGL_KEY", "key")
    monkeypatch.setenv("AGL_EVENT_URL", "http://agl/events")
    monkeypatch.setenv("AGL_OPENAI_BASE_URL", "http://agl/proxy/openai/v1")
    monkeypatch.setenv("SEARCH_R1_TEMPERATURE", "0.25")
    monkeypatch.setenv("SEARCH_R1_MAX_TOKENS", "64")

    asyncio.run(search_r1_agent.SearchR1Agent().run())

    assert calls["openai"] == {
        "base_url": "http://agl/proxy/openai/v1",
        "api_key": "key",
        "max_retries": 6,
    }
    assert calls["reward"] == {
        "event_url": "http://agl/events",
        "agl_key": "key",
        "reward": 1.0,
        "reason": "em_match",
    }
    assert calls["messages"][0][0]["role"] == "user"
