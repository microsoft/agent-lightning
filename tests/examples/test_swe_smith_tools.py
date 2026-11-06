# Copyright (c) Microsoft. All rights reserved.

"""Exercise the harness with the SDK messages returned by native tool parsers."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from openai.types.chat import ChatCompletion

from examples.swe_smith.agents import smith_agent


def _call(command: str, *, call_id: str = "call_1", name: str = "execute_bash") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps({"command": command})},
    }


class _Client:
    def __init__(self, *messages: dict[str, Any], finish_reason: str = "tool_calls") -> None:
        self.messages = iter(messages)
        self.requests: list[dict[str, Any]] = []
        self.finish_reason = finish_reason
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> ChatCompletion:
        # Model the server's JSON decoding and Qwen template's mapping access,
        # so a mocked completion cannot hide an invalid follow-up request.
        for message in kwargs["messages"]:
            for call in message.get("tool_calls", []):
                arguments = json.loads(call["function"]["arguments"])
                assert isinstance(arguments, dict)
        self.requests.append(copy.deepcopy(kwargs))
        return ChatCompletion.model_validate(
            {
                "id": "completion",
                "created": 0,
                "model": "Qwen/Qwen3.5-9B",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": self.finish_reason,
                        "message": {"role": "assistant", "content": None, **next(self.messages)},
                    }
                ],
                "usage": {"prompt_tokens": 100 * len(self.requests), "completion_tokens": 10, "total_tokens": 110},
            }
        )


def _run_loop(client: _Client, **kwargs: Any) -> tuple[bool, int, int]:
    return smith_agent.run_agent_loop(
        client, "fix the bug", max_turns=2, cmd_timeout=1, obs_cap=1000, max_tokens=2048, **kwargs
    )


@pytest.mark.parametrize("reasoning_field", ["reasoning_content", "reasoning"])
def test_native_calls_keep_reasoning_and_link_tool_results(monkeypatch, reasoning_field: str) -> None:
    client = _Client(
        {"tool_calls": [_call("ls -la")], reasoning_field: "Inspect the files first."},
        {"tool_calls": [_call(f"echo {smith_agent.SUBMIT_MARKER}", call_id="call_2")]},
    )
    commands: list[str] = []

    def run(command: str, _timeout: int) -> tuple[str, int]:
        commands.append(command)
        return (smith_agent.SUBMIT_MARKER if command.startswith("echo ") else "file.py\n"), 0

    monkeypatch.setattr(smith_agent, "_run", run)
    assert _run_loop(client, use_tools=True, enable_thinking=True) == (True, 2, 200)
    assert commands == ["ls -la", f"echo {smith_agent.SUBMIT_MARKER}"]
    request = client.requests[1]
    assert request["tools"] == [smith_agent.BASH_TOOL]
    assert request["parallel_tool_calls"] is False
    assert request["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assistant, observation = request["messages"][-2:]
    assert assistant["tool_calls"] == [_call("ls -la")]
    assert assistant["reasoning_content"] == "Inspect the files first."
    assert assistant["reasoning"] == "Inspect the files first."
    assert observation == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "<returncode>0</returncode>\n<output>\nfile.py\n</output>",
    }
    assert "bash code block" not in request["messages"][0]["content"]
    assert "```bash" not in request["messages"][1]["content"]


@pytest.mark.parametrize(
    "arguments",
    [
        '{"command": 4}',
        '{"command": " "}',
        '{"other": "ls"}',
        '{"command": "ls", "extra": true}',
    ],
)
def test_bad_tool_arguments_are_reported_without_running(monkeypatch, arguments: str) -> None:
    call = _call("unused")
    call["function"]["arguments"] = arguments
    client = _Client({"tool_calls": [call]}, {"tool_calls": [call]})
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("malformed command ran"))
    assert _run_loop(client, use_tools=True) == (False, 2, 200)
    observation = client.requests[1]["messages"][-1]
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call_1"
    assert "command" in observation["content"]


@pytest.mark.parametrize("arguments", ["{", "null", "[]", '"ls"'])
@pytest.mark.parametrize("variant", ["single", "parallel", "truncated", "unknown"])
def test_unrenderable_tool_arguments_end_episode(monkeypatch, arguments: str, variant: str) -> None:
    call = _call("unused", name="unknown" if variant == "unknown" else "execute_bash")
    call["function"]["arguments"] = arguments
    calls = [_call("ls", call_id="call_2"), call] if variant == "parallel" else [call]
    client = _Client({"tool_calls": calls}, finish_reason="length" if variant == "truncated" else "tool_calls")
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("malformed command ran"))
    assert _run_loop(client, use_tools=True) == (False, 1, 100)
    assert len(client.requests) == 1


@pytest.mark.parametrize(
    "calls",
    [[_call("ls", name="unknown")], [_call("ls"), _call("pwd", call_id="call_2")]],
)
def test_unknown_or_parallel_tools_reply_to_every_call(monkeypatch, calls: list[dict[str, Any]]) -> None:
    client = _Client({"tool_calls": calls}, {"tool_calls": calls})
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("invalid tool ran"))
    assert _run_loop(client, use_tools=True) == (False, 2, 200)
    observations = client.requests[1]["messages"][-len(calls) :]
    assert [message["tool_call_id"] for message in observations] == [call["id"] for call in calls]
    assert all(message["role"] == "tool" for message in observations)


@pytest.mark.parametrize("command", ["git show HEAD", "curl https://example.com", "pip install example"])
def test_native_tools_keep_action_restrictions(monkeypatch, command: str) -> None:
    client = _Client({"tool_calls": [_call(command)]}, {"tool_calls": [_call(command)]})
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("forbidden command ran"))
    assert _run_loop(client, use_tools=True) == (False, 2, 200)
    observation = client.requests[1]["messages"][-1]
    assert observation["role"] == "tool"
    assert observation["tool_call_id"] == "call_1"
    assert "<returncode>1</returncode>" in observation["content"]


def test_native_mode_does_not_execute_fenced_text(monkeypatch) -> None:
    client = _Client({"content": "```bash\nls\n```"}, {"content": "```bash\nls\n```"}, finish_reason="stop")
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("non-tool command ran"))
    assert _run_loop(client, use_tools=True) == (False, 2, 200)
    assert client.requests[1]["messages"][-1] == {
        "role": "user",
        "content": "Expected exactly one execute_bash call, found 0.",
    }


def test_truncated_native_call_is_not_executed(monkeypatch) -> None:
    client = _Client({"tool_calls": [_call("ls")]}, {"tool_calls": [_call("ls")]}, finish_reason="length")
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("truncated command ran"))
    assert _run_loop(client, use_tools=True) == (False, 2, 200)
    assert "output token limit" in client.requests[1]["messages"][-1]["content"]


def test_default_bash_mode_remains_non_thinking(monkeypatch) -> None:
    client = _Client({"content": "```bash\nls\n```"}, {"content": "```bash\npwd\n```"}, finish_reason="stop")
    commands: list[str] = []
    monkeypatch.setattr(smith_agent, "_run", lambda command, _: (commands.append(command) or "ok", 0))
    assert _run_loop(client) == (False, 2, 200)
    assert commands == ["ls", "pwd"]
    assert "tools" not in client.requests[0]
    assert client.requests[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert client.requests[1]["messages"][-1]["role"] == "user"


@pytest.mark.parametrize("opening_tag", ["<think>\n", ""])
def test_bash_mode_ignores_commands_inside_raw_thinking(monkeypatch, opening_tag: str) -> None:
    content = opening_tag + "Consider ```bash\nwrong\n```\n</think>\n```bash\nls\n```"
    client = _Client({"content": content}, {"content": content}, finish_reason="stop")
    commands: list[str] = []
    monkeypatch.setattr(smith_agent, "_run", lambda command, _: (commands.append(command) or "ok", 0))
    assert _run_loop(client, enable_thinking=True) == (False, 2, 200)
    assert commands == ["ls", "ls"]
    assert client.requests[1]["messages"][-2]["content"] == content


def test_bash_mode_accepts_separate_reasoning(monkeypatch) -> None:
    message = {"content": "```bash\nls\n```", "reasoning": "Consider ```bash\nwrong\n```"}
    client = _Client(message, message, finish_reason="stop")
    commands: list[str] = []
    monkeypatch.setattr(smith_agent, "_run", lambda command, _: (commands.append(command) or "ok", 0))
    assert _run_loop(client, enable_thinking=True) == (False, 2, 200)
    assert commands == ["ls", "ls"]
    assert client.requests[1]["messages"][-2]["reasoning_content"] == message["reasoning"]


@pytest.mark.parametrize("opening_tag", ["<think>\n", ""])
def test_unfinished_raw_thinking_does_not_execute_a_command(monkeypatch, opening_tag: str) -> None:
    message = {"content": opening_tag + "Consider ```bash\nwrong\n```"}
    client = _Client(message, message, finish_reason="length")
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("reasoning command ran"))
    assert _run_loop(client, enable_thinking=True) == (False, 2, 200)
    assert "output token limit" in client.requests[1]["messages"][-1]["content"]


@pytest.mark.parametrize("enabled", [False, True])
def test_main_reads_tool_and_thinking_settings(monkeypatch, enabled: bool) -> None:
    monkeypatch.setenv("AGL_TASK_INPUT", "fix the bug")
    monkeypatch.setenv("AGL_EVAL_META", '{"instance_id": "example"}')
    monkeypatch.setenv("AGL_OPENAI_BASE_URL", "http://example.invalid/v1")
    if enabled:
        monkeypatch.setenv("SMITH_USE_TOOLS", "true")
        monkeypatch.setenv("SMITH_ENABLE_THINKING", "1")
    else:
        monkeypatch.delenv("SMITH_USE_TOOLS", raising=False)
        monkeypatch.delenv("SMITH_ENABLE_THINKING", raising=False)
    settings: dict[str, Any] = {}

    def run_loop(*_args: Any, **kwargs: Any) -> tuple[bool, int, int]:
        settings.update(kwargs)
        return False, 0, 0

    monkeypatch.setattr("openai.OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(smith_agent, "checkout_bug_commit", lambda *_args: None)
    monkeypatch.setattr(smith_agent, "relocate_git", lambda: None)
    monkeypatch.setattr(smith_agent, "run_agent_loop", run_loop)
    monkeypatch.setattr(smith_agent, "capture_patch", lambda: "")
    monkeypatch.setattr(smith_agent, "evaluate", lambda *_args, **_kwargs: (0.0, False, "test", False))
    monkeypatch.setattr(smith_agent, "post_event", lambda *_args, **_kwargs: None)
    assert smith_agent.main() == 0
    assert settings["use_tools"] is enabled
    assert settings["enable_thinking"] is enabled


def test_tool_format_error_can_recover_and_submit(monkeypatch) -> None:
    client = _Client(
        {"tool_calls": [_call("ls", name="unknown")]},
        {"tool_calls": [_call(f"echo {smith_agent.SUBMIT_MARKER}", call_id="call_2")]},
    )
    commands: list[str] = []
    monkeypatch.setattr(
        smith_agent, "_run", lambda command, _: (commands.append(command) or smith_agent.SUBMIT_MARKER, 0)
    )
    assert _run_loop(client, use_tools=True) == (True, 2, 200)
    assert commands == [f"echo {smith_agent.SUBMIT_MARKER}"]


def test_native_format_errors_respect_the_error_limit(monkeypatch) -> None:
    client = _Client({"tool_calls": [_call("ls", name="unknown")]})
    monkeypatch.setattr(smith_agent, "_run", lambda *_args: pytest.fail("invalid tool ran"))
    assert _run_loop(client, use_tools=True, max_format_errors=1) == (False, 1, 100)
    assert len(client.requests) == 1


def test_canonical_reasoning_takes_precedence_over_empty_legacy_field() -> None:
    client = _Client({"reasoning": "Inspect the files.", "reasoning_content": "", "tool_calls": [_call("ls")]})
    message, _, _ = smith_agent._query(client, [], 2048, use_tools=True, enable_thinking=True)
    assert message["reasoning"] == "Inspect the files."
    assert message["reasoning_content"] == "Inspect the files."
