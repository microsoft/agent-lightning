# Copyright (c) Microsoft. All rights reserved.

from pathlib import Path

import pytest
from pydantic import ValidationError

import agentlightning.types.core as core
from agentlightning.types import AgentSpanPayload, RolloutResult, SpanWriteResult


def test_agent_span_payload_fields() -> None:
    payload = AgentSpanPayload(
        name="agent.step",
        status={"status_code": "OK"},
        attributes={"reward": 1.0},
        start_time=1.0,
        end_time=2.0,
    )

    assert payload.name == "agent.step"
    assert payload.status == {"status_code": "OK"}
    assert payload.attributes == {"reward": 1.0}
    assert payload.start_time == 1.0
    assert payload.end_time == 2.0


def test_agent_span_payload_rejects_unsupported_fields() -> None:
    with pytest.raises(ValidationError, match="events"):
        AgentSpanPayload.model_validate(
            {
                "name": "agent.step",
                "status": {"status_code": "OK"},
                "attributes": {},
                "events": [{"name": "tool_call"}],
            }
        )


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


def test_core_does_not_expose_legacy_result_contracts() -> None:
    """Core type module should not expose legacy HTTP payload models."""

    assert not hasattr(core, "RolloutLegacy")
    assert not hasattr(core, "Task")
    assert not hasattr(core, "TaskIfAny")
    assert not hasattr(core, "RolloutRawResultLegacy")
    assert not hasattr(core, "RolloutRawResult")


def test_legacy_types_module_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    assert not (repo_root / "agentlightning" / "types" / "legacy.py").exists()
