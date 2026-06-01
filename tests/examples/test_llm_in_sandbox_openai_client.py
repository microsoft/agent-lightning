from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

HELPER_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/llm-in-sandbox/vendor/llm-in-sandbox/llm_in_sandbox/openai_client.py"
)
_spec = importlib.util.spec_from_file_location("llm_in_sandbox_openai_client", HELPER_PATH)
assert _spec is not None and _spec.loader is not None
openai_client = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(openai_client)


class _FakeCompletions:
    def __init__(self, calls: dict[str, Any]) -> None:
        self.calls = calls

    def create(self, **kwargs: Any) -> object:
        self.calls["create"] = kwargs
        return object()


class _FakeChat:
    def __init__(self, calls: dict[str, Any]) -> None:
        self.completions = _FakeCompletions(calls)


class _FakeOpenAI:
    def __init__(self, **kwargs: Any) -> None:
        self.calls = _FAKE_CALLS
        self.calls["client"] = kwargs
        self.chat = _FakeChat(self.calls)


_FAKE_CALLS: dict[str, Any] = {}


def test_create_chat_completion_strips_prefix_and_passes_temperature(monkeypatch) -> None:
    _FAKE_CALLS.clear()
    monkeypatch.setattr(openai_client, "OpenAI", _FakeOpenAI)

    response = openai_client.create_chat_completion(
        model="openai/Qwen/Qwen3-4B-Instruct-2507",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.2,
        max_tokens=123,
        base_url="http://gateway/v1",
        api_key="key",
        timeout=900,
        tools=[{"type": "function", "function": {"name": "submit"}}],
        extra_body={"chat_template_kwargs": {"thinking": True}},
    )

    assert response is not None
    assert _FAKE_CALLS["client"] == {
        "api_key": "key",
        "base_url": "http://gateway/v1",
        "timeout": 900,
        "max_retries": 0,
    }
    assert _FAKE_CALLS["create"] == {
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "max_tokens": 123,
        "tools": [{"type": "function", "function": {"name": "submit"}}],
        "extra_body": {"chat_template_kwargs": {"thinking": True}},
    }
