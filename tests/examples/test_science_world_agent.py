# Copyright (c) Microsoft. All rights reserved.

"""ScienceWorld action selection through the OpenAI-compatible API boundary."""

from __future__ import annotations

import json
import sys
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import httpx
import openai
import pytest

from examples.science_world.agents.sw_agent import SWAgent, _parse_action_index


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["", "\n", "\n\n"])
async def test_selected_action_survives_stop_filtering(monkeypatch: pytest.MonkeyPatch, ending: str) -> None:
    generated = "### ACTION: 2 ###" + ending
    requests: list[dict[str, Any]] = []
    returned_text: list[str] = []

    def complete(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        # vLLM excludes a matched stop string unless explicitly asked to retain it.
        text = generated
        for stop in body.get("stop", []):
            if stop in text:
                end = text.index(stop) + (len(stop) if body.get("include_stop_str_in_output", False) else 0)
                text = text[:end]
                break
        returned_text.append(text)
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
            },
        )

    env = Mock()
    env.get_task_description.return_value = "Turn on the furnace."
    env.reset.return_value = ("A room with a furnace.", {"score": 0})
    env.inventory.return_value = "Nothing."
    env.get_valid_action_object_combinations_with_templates.return_value = [
        {"action": "look around"},
        {"action": "inventory"},
        {"action": "turn on furnace"},
    ]
    env.step.side_effect = [("Continue.", 0, False, {"score": 50}), ("Done.", 50, True, {"score": 100})]
    scienceworld = ModuleType("scienceworld")
    monkeypatch.setattr(scienceworld, "ScienceWorldEnv", Mock(return_value=env), raising=False)
    monkeypatch.setitem(sys.modules, "scienceworld", scienceworld)
    for name, value in {
        "TASK_NAME": "test",
        "VARIATION_IDX": "0",
        "SW_MAX_STEPS": "2",
        "SW_MAX_VALID_ACTIONS_SHOWN": "3",
        "AGL_KEY": "test-key",
        "AGL_EVENT_URL": "http://gateway/events",
        "AGL_OPENAI_BASE_URL": "http://gateway/v1",
    }.items():
        monkeypatch.setenv(name, value)
    reward_post = Mock(return_value=httpx.Response(200, request=httpx.Request("POST", "http://gateway/events")))
    monkeypatch.setattr(httpx, "post", reward_post)

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(complete)) as http_client,
        openai.AsyncOpenAI(api_key="test-key", base_url="http://gateway/v1", http_client=http_client) as client,
    ):
        monkeypatch.setattr(openai, "AsyncOpenAI", lambda **_kwargs: client)
        await SWAgent().run()

    assert [call.args[0] for call in env.step.call_args_list] == ["turn on furnace", "turn on furnace"]
    assert len(requests) == 2
    assert requests[1]["messages"][-2] == {"role": "assistant", "content": returned_text[0]}
    assert _parse_action_index(returned_text[0], 3) == 2
    assert reward_post.call_args.kwargs["json"]["data"]["value"] == 1.0


@pytest.mark.parametrize("text", ["### ACTION: 2", "### ACTION: invalid ###", "### ACTION: 3 ###", "unrelated"])
def test_invalid_actions_still_use_the_fallback(text: str) -> None:
    assert _parse_action_index(text, 3) == 0
