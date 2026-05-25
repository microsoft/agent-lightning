from __future__ import annotations

import asyncio
from typing import Any

from examples.async_calc_x.agents import calc_agent


def test_calc_x_agent_run_solves_and_posts_agent_output(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    async def fake_solve(question: str, model: str, temperature: float) -> tuple[str, str]:
        calls["solve"] = {
            "question": question,
            "model": model,
            "temperature": temperature,
        }
        return "4", "### ANSWER: 4 ###"

    def fake_post_event(event_type: str, data: dict[str, Any]) -> None:
        calls["event"] = {"event_type": event_type, "data": data}

    monkeypatch.setenv("AGL_MODEL_NAME", "local-model")
    monkeypatch.setenv("AGL_TEMPERATURE", "0.2")
    monkeypatch.setattr(calc_agent, "solve", fake_solve)
    monkeypatch.setattr(calc_agent, "post_event", fake_post_event)

    result = calc_agent.CalcXAgent().run({"question": "2+2?", "id": "sample-1"})

    if asyncio.iscoroutine(result):
        asyncio.run(result)

    assert calls["solve"] == {
        "question": "2+2?",
        "model": "local-model",
        "temperature": 0.2,
    }
    assert calls["event"] == {
        "event_type": "agent_output",
        "data": {
            "answer": "4",
            "raw_response": "### ANSWER: 4 ###",
            "task_id": "sample-1",
        },
    }
