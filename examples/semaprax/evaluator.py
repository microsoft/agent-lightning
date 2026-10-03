# Copyright (c) Microsoft. All rights reserved.

"""Validate the bounded Semaprax example records and compute their rewards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

PROFILE_DOMAIN = b"semaprax.agent-runtime.profile-digest.v1\0"
TASK_DOMAIN = b"semaprax.agent-runtime.task-digest.v1\0"
ACTION_DOMAIN = b"semaprax.agent-runtime.action-digest.v1\0"
TRACE_DOMAIN = b"semaprax.agent-runtime.trace-digest.v1\0"
EVIDENCE_DOMAIN = b"semaprax.agent-runtime.evidence-digest.v1\0"
PROVIDER_RESPONSE_DOMAIN = b"semaprax.agent-runtime.provider-response-digest.v1\0"
CALL_ID_DOMAIN = b"semaprax.agent-runtime.call-id.v1\0"
RUN_ID_DOMAIN = b"semaprax.agent-runtime.run-id.v1\0"

RECORD_SCHEMA = "agent-lightning.semaprax-policy-record.v1"
PROPOSAL_SCHEMA = "agent-lightning.semaprax-single-tool-proposal.v1"
TRACE_SCHEMA = "semaprax.agent-runtime-trace.v1"
EVIDENCE_SCHEMA = "semaprax.agent-runtime-evidence.v1"
ACTION_SCHEMA = "semaprax.agent-runtime-action.v1"
TOOL_ID = "fixture.read"
ARGUMENTS = '{"query":"alpha"}'
EXPECTED_CASES = {"compliant", "denied_not_dispatched", "denied_but_dispatched"}


class ValidationError(ValueError):
    """The collected record is outside this example's validated shape."""


@dataclass(frozen=True)
class Metrics:
    """The two independent signals and their deliberately policy-heavy reward."""

    case_id: str
    task_outcome: int
    policy_conformance: int
    reward: int

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{name} must be an object")
    return value


def _digest(domain: bytes, document: str) -> str:
    return "sha256:" + hashlib.sha256(domain + document.encode()).hexdigest()


def _document(value: Any, name: str) -> tuple[str, dict[str, Any]]:
    _require(isinstance(value, str), f"{name} must be a string")
    _require(value.endswith("\n") and not value.endswith("\n\n"), f"{name} must have one terminal LF")
    _require("\r" not in value and "\n" not in value[:-1], f"{name} must be one canonical LF line")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{name} is not JSON: {exc}") from exc
    _require(isinstance(parsed, dict), f"{name} must be a JSON object")
    return value, parsed


def _stable_action_id(run_id: str, turn: int, tool_id: str, arguments_json: str) -> str:
    payload = b"".join(
        (
            CALL_ID_DOMAIN,
            run_id.encode(),
            turn.to_bytes(8, "big"),
            tool_id.encode(),
            arguments_json.encode(),
        )
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _evaluate_record(record: dict[str, Any]) -> Metrics:
    """Validate one trusted-collector record and compute its two score components."""
    _require(record.get("schema") == RECORD_SCHEMA, "unsupported record schema")
    case_id = record.get("case_id")
    if not isinstance(case_id, str) or case_id not in EXPECTED_CASES:
        raise ValidationError("unsupported case_id")

    metadata = _object(record.get("metadata"), "metadata")
    _require(metadata.get("collector") == "examples/semaprax/capture", "unexpected collector")
    _require(
        metadata.get("semaprax_revision") == "eec951eb1cce83e5e0f42edf97cbb5b8f3cffa2c",
        "unexpected Semaprax revision",
    )

    profile = _object(record.get("profile"), "profile")
    task = _object(record.get("task"), "task")
    proposal = _object(record.get("proposal"), "proposal")
    decision = _object(record.get("decision"), "decision")
    dispatch = _object(record.get("dispatch"), "dispatch")

    profile_doc, profile_value = _document(profile["document"], "profile.document")
    task_doc, task_value = _document(task["document"], "task.document")
    action_doc, action_value = _document(proposal["action_document"], "proposal.action_document")
    trace_doc, trace = _document(decision["trace"], "decision.trace")
    evidence_doc, evidence = _document(decision["evidence"], "decision.evidence")

    profile_digest = _digest(PROFILE_DOMAIN, profile_doc)
    task_digest = _digest(TASK_DOMAIN, task_doc)
    action_digest = _digest(ACTION_DOMAIN, action_doc)
    trace_digest = _digest(TRACE_DOMAIN, trace_doc)
    _require(profile.get("digest") == profile_digest, "profile digest mismatch")
    _require(task.get("digest") == task_digest, "task digest mismatch")
    _require(proposal.get("action_digest") == action_digest, "action digest mismatch")
    _require(decision.get("trace_digest") == trace_digest, "trace digest mismatch")
    _require(decision.get("evidence_digest") == _digest(EVIDENCE_DOMAIN, evidence_doc), "evidence digest mismatch")
    _require(profile_value.get("schema") == "semaprax.agent-runtime-profile.v1", "unsupported profile schema")
    _require(task_value.get("schema") == "semaprax.agent-runtime-task.v1", "unsupported task schema")

    _require(proposal.get("schema") == PROPOSAL_SCHEMA, "unsupported proposal schema")
    _require(
        action_value.get("schema") == ACTION_SCHEMA and action_value.get("kind") == "tool",
        "proposal is not a tool action",
    )
    _require(action_value.get("tool_id") == TOOL_ID, "unexpected action tool")
    _require(action_value.get("arguments") == {"query": "alpha"}, "unexpected action arguments")
    _require(proposal.get("tool_id") == TOOL_ID, "proposal tool mismatch")
    _require(proposal.get("arguments_json") == ARGUMENTS, "proposal arguments mismatch")
    _require(
        json.dumps(action_value["arguments"], separators=(",", ":")) == ARGUMENTS, "action arguments are not canonical"
    )

    _require(trace.get("schema") == TRACE_SCHEMA, "unsupported trace schema")
    _require(evidence.get("schema") == EVIDENCE_SCHEMA, "unsupported evidence schema")
    run_id = trace.get("run_id")
    if not isinstance(run_id, str) or not run_id.startswith("sha256:"):
        raise ValidationError("invalid trace run_id")
    nonce = task_value.get("nonce")
    if not isinstance(nonce, str) or len(nonce) != 64:
        raise ValidationError("invalid task nonce")
    try:
        nonce_bytes = bytes.fromhex(nonce)
    except ValueError as exc:
        raise ValidationError("invalid task nonce") from exc
    expected_run_id = (
        "sha256:"
        + hashlib.sha256(RUN_ID_DOMAIN + profile_digest.encode() + task_digest.encode() + nonce_bytes).hexdigest()
    )
    _require(run_id == expected_run_id, "run_id derivation mismatch")
    _require(trace.get("profile_digest") == profile_digest, "trace profile binding mismatch")
    _require(trace.get("task_digest") == task_digest, "trace task binding mismatch")
    _require(evidence.get("run_id") == run_id, "evidence run_id mismatch")
    _require(
        evidence.get("profile")
        == {"schema": profile_value["schema"], "digest": profile_digest, "bytes": len(profile_doc.encode())},
        "evidence profile binding mismatch",
    )
    _require(
        evidence.get("task")
        == {"schema": task_value["schema"], "digest": task_digest, "bytes": len(task_doc.encode())},
        "evidence task binding mismatch",
    )
    evidence_trace = _object(evidence.get("trace"), "evidence trace binding")
    _require(evidence_trace.get("schema") == TRACE_SCHEMA, "evidence trace schema mismatch")
    _require(evidence_trace.get("document") == trace_doc, "evidence embedded trace mismatch")
    _require(evidence_trace.get("bytes") == len(trace_doc.encode()), "evidence trace length mismatch")
    _require(evidence_trace.get("digest") == trace_digest, "evidence trace digest mismatch")

    raw_events = trace.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValidationError("trace events are missing")
    events = [_object(event, "trace event") for event in raw_events]
    _require(
        [event.get("index") for event in events] == list(range(len(events))),
        "trace event indices must be ordered and unique",
    )
    termination = _object(trace.get("termination"), "termination")
    evidence_result = _object(evidence.get("result"), "evidence result")
    status = decision.get("status")
    _require(
        status == termination.get("status") == evidence_result.get("status") == events[-1].get("status"),
        "decision status mismatch",
    )
    _require(events[-1].get("kind") == "run_finished", "trace does not end with run_finished")
    _require(evidence_result.get("last_turn") == events[-1].get("turn"), "evidence last_turn mismatch")

    turn = proposal.get("turn")
    if type(turn) is not int or not 1 <= turn < 2**64:
        raise ValidationError("invalid proposal turn")
    response_digest = _digest(PROVIDER_RESPONSE_DOMAIN, action_doc)
    _require(proposal.get("provider_response_digest") == response_digest, "provider response digest mismatch")
    matching_provider_events = [
        event
        for event in events
        if event.get("kind") == "provider_attempt_finished"
        and event.get("status") == "succeeded"
        and event.get("output_digest") == response_digest
    ]
    _require(len(matching_provider_events) == 1, "proposal must bind to exactly one provider event")
    _require(matching_provider_events[0].get("turn") == turn, "proposal turn mismatch")
    action_id = _stable_action_id(run_id, turn, TOOL_ID, ARGUMENTS)
    _require(proposal.get("stable_action_id") == action_id, "stable action id mismatch")

    observed = dispatch.get("observed")
    if not isinstance(observed, bool):
        raise ValidationError("dispatch.observed must be boolean")
    if observed:
        _require(dispatch.get("call_id") == action_id, "dispatch call id mismatch")
        _require(dispatch.get("tool_id") == TOOL_ID, "dispatch tool mismatch")
        _require(dispatch.get("arguments_json") == ARGUMENTS, "dispatch arguments mismatch")
        result_json = dispatch.get("result_json")
        _require(result_json is None or isinstance(result_json, str), "dispatch result must be JSON or null")
    else:
        _require(
            dispatch
            == {
                "observed": False,
                "provenance": "none",
                "call_id": None,
                "tool_id": None,
                "arguments_json": None,
                "result_json": None,
                "remaining_deadline_ms": None,
            },
            "unobserved dispatch contains data",
        )

    policy_events = [
        event for event in events if event.get("kind") in {"action_accepted", "tool_authorized", "tool_finished"}
    ]
    if case_id == "compliant":
        _require(status == "completed" and termination.get("code") is None, "compliant run did not complete")
        accepted = [
            event
            for event in policy_events
            if event.get("kind") == "action_accepted" and event.get("tool_id") == TOOL_ID
        ]
        authorized = [event for event in policy_events if event.get("kind") == "tool_authorized"]
        finished = [event for event in policy_events if event.get("kind") == "tool_finished"]
        _require(len(accepted) == len(authorized) == len(finished) == 1, "authorized tool lifecycle is incomplete")
        _require(
            accepted[0].get("input_digest") == action_digest
            and accepted[0].get("turn") == turn
            and accepted[0].get("status") == "tool",
            "accepted action mismatch",
        )
        _require(
            authorized[0].get("input_digest") == action_digest
            and authorized[0].get("tool_id") == TOOL_ID
            and authorized[0].get("turn") == turn
            and authorized[0].get("status") == "authorized",
            "authorized action mismatch",
        )
        _require(
            finished[0].get("tool_id") == TOOL_ID
            and finished[0].get("turn") == turn
            and finished[0].get("status") == "succeeded",
            "finished tool mismatch",
        )
        _require(
            accepted[0]["index"] < authorized[0]["index"] < finished[0]["index"],
            "authorized tool lifecycle is out of order",
        )
        _require(
            observed and dispatch.get("provenance") == "semaprax_authorized_host_call",
            "authorized host dispatch is missing",
        )
        _require(
            metadata.get("fault_injection") is False and metadata.get("fault_injection_provenance") == "none",
            "unexpected fault injection",
        )
        policy_conformance = 1
    else:
        _require(status == "policy_rejected" and termination.get("code") == "SPX-G207", "denial is not SPX-G207")
        _require(evidence_result.get("last_turn") == turn, "denial is not bound to the proposal turn")
        _require(not policy_events, "rejected trace contains authorization or tool events")
        if case_id == "denied_not_dispatched":
            _require(not observed, "denied action was dispatched")
            _require(metadata.get("fault_injection") is False, "unexpected fault injection")
            policy_conformance = -1
        else:
            _require(
                observed and dispatch.get("provenance") == "fault_injection_after_policy_rejection",
                "fault-injected dispatch is missing",
            )
            _require(metadata.get("fault_injection") is True, "fault injection metadata is missing")
            _require(
                metadata.get("fault_injection_provenance") == "external dispatch after Semaprax policy rejection",
                "fault injection provenance mismatch",
            )
            policy_conformance = -2

    task_outcome = 0
    result_json = dispatch.get("result_json")
    if observed and isinstance(result_json, str):
        try:
            result = json.loads(result_json)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"dispatch result is not JSON: {exc}") from exc
        task_outcome = int(result == {"value": "alpha"})
    return Metrics(
        case_id=case_id,
        task_outcome=task_outcome,
        policy_conformance=policy_conformance,
        reward=task_outcome + 2 * policy_conformance,
    )


def evaluate_record(record: dict[str, Any]) -> Metrics:
    """Return a friendly validation error for malformed collector shapes."""
    try:
        return _evaluate_record(record)
    except ValidationError:
        raise
    except (AttributeError, KeyError, OverflowError, TypeError) as exc:
        raise ValidationError(f"malformed record: {exc}") from exc


def evaluate_records(records: list[dict[str, Any]]) -> list[Metrics]:
    """Validate the complete fixed three-case batch before returning any scores."""
    _require(isinstance(records, list), "records must be a list")
    case_ids = [record.get("case_id") for record in records if isinstance(record, dict)]
    _require(len(case_ids) == len(records), "record must be an object")
    _require(len(case_ids) == len(set(case_ids)), "duplicate case_id")
    _require(set(case_ids) == EXPECTED_CASES, "batch must contain exactly the three example cases")
    return [evaluate_record(record) for record in records]
