# Copyright (c) Microsoft. All rights reserved.

from typing import Any

from agentlightning.types import AgentSpanPayload, RolloutResult, SpanWriteResult


def test_agent_span_payload_fields() -> None:
    payload = AgentSpanPayload(
        name="agent.step",
        status={"status_code": "OK"},
        attributes={"reward": 1.0},
        start_time=1.0,
        end_time=2.0,
        events=[{"name": "tool_call", "data": {"tool": "search"}}],
    )

    assert payload.name == "agent.step"
    assert payload.status == {"status_code": "OK"}
    assert payload.attributes == {"reward": 1.0}
    assert payload.start_time == 1.0
    assert payload.end_time == 2.0
    assert payload.events == [{"name": "tool_call", "data": {"tool": "search"}}]


def test_span_write_result_schema_defaults() -> None:
    empty = SpanWriteResult()
    full = SpanWriteResult(inserted=2, duplicates=1, failed=0)

    assert empty.inserted == 0
    assert empty.duplicates == 0
    assert empty.failed == 0
    assert full.inserted == 2
    assert full.duplicates == 1
    assert full.failed == 0


def test_rollout_result_contract_alias() -> None:
    payload = AgentSpanPayload(
        name="agent.run",
        status={"status_code": "OK"},
        attributes={"ok": True},
    )

    null_result: RolloutResult = None
    reward_result: RolloutResult = 1.0
    payload_result: RolloutResult = [payload]

    assert null_result is None
    assert reward_result == 1.0
    assert payload_result == [payload]
    assert isinstance(payload_result, list)
    assert isinstance(payload_result[0], AgentSpanPayload)
