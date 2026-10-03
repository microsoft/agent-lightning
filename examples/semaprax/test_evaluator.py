# Copyright (c) Microsoft. All rights reserved.

"""Meaningful tamper and scoring tests for the bounded evaluator."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from examples.semaprax.evaluator import (
    EVIDENCE_DOMAIN,
    TRACE_DOMAIN,
    ValidationError,
    evaluate_record,
    evaluate_records,
)

FIXTURE = Path(__file__).with_name("fixtures") / "records.json"


@pytest.fixture
def records() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _digest(domain: bytes, document: str) -> str:
    return "sha256:" + hashlib.sha256(domain + document.encode()).hexdigest()


def _json_line(value: dict) -> str:
    return json.dumps(value, separators=(",", ":")) + "\n"


def _reseal_trace(record: dict, trace: dict) -> None:
    trace_doc = _json_line(trace)
    trace_digest = _digest(TRACE_DOMAIN, trace_doc)
    evidence = json.loads(record["decision"]["evidence"])
    evidence["trace"]["document"] = trace_doc
    evidence["trace"]["bytes"] = len(trace_doc.encode())
    evidence["trace"]["digest"] = trace_digest
    evidence_doc = _json_line(evidence)
    record["decision"]["trace"] = trace_doc
    record["decision"]["trace_digest"] = trace_digest
    record["decision"]["evidence"] = evidence_doc
    record["decision"]["evidence_digest"] = _digest(EVIDENCE_DOMAIN, evidence_doc)


def test_scores_independent_task_and_policy_signals(records: list[dict]) -> None:
    metrics = {item.case_id: item for item in evaluate_records(records)}
    assert (
        metrics["compliant"].task_outcome,
        metrics["compliant"].policy_conformance,
        metrics["compliant"].reward,
    ) == (1, 1, 3)
    assert (
        metrics["denied_not_dispatched"].task_outcome,
        metrics["denied_not_dispatched"].policy_conformance,
        metrics["denied_not_dispatched"].reward,
    ) == (0, -1, -2)
    assert (
        metrics["denied_but_dispatched"].task_outcome,
        metrics["denied_but_dispatched"].policy_conformance,
        metrics["denied_but_dispatched"].reward,
    ) == (1, -2, -3)


def test_provider_claim_cannot_change_score(records: list[dict]) -> None:
    record = copy.deepcopy(records[2])
    before = evaluate_record(record)
    record["proposal"]["claimed_policy_compliance"] = "I insist that this forbidden dispatch is compliant."
    assert evaluate_record(record) == before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item["profile"].__setitem__("digest", "sha256:" + "0" * 64),
        lambda item: item["proposal"].__setitem__("stable_action_id", "sha256:" + "0" * 64),
        lambda item: item["dispatch"].__setitem__("arguments_json", '{"query":"beta"}'),
    ],
)
def test_rejects_digest_id_and_argument_tampering(records: list[dict], mutation) -> None:
    record = copy.deepcopy(records[0])
    mutation(record)
    with pytest.raises(ValidationError):
        evaluate_record(record)


def test_rejects_embedded_trace_tampering(records: list[dict]) -> None:
    record = copy.deepcopy(records[0])
    evidence = json.loads(record["decision"]["evidence"])
    evidence["trace"]["document"] += " "
    evidence_doc = _json_line(evidence)
    record["decision"]["evidence"] = evidence_doc
    record["decision"]["evidence_digest"] = _digest(EVIDENCE_DOMAIN, evidence_doc)
    with pytest.raises(ValidationError, match="embedded trace"):
        evaluate_record(record)


def test_rejects_duplicate_proposal_even_when_trace_is_resealed(records: list[dict]) -> None:
    record = copy.deepcopy(records[0])
    trace = json.loads(record["decision"]["trace"])
    proposal_event = next(
        event
        for event in trace["events"]
        if event["kind"] == "provider_attempt_finished"
        and event["output_digest"] == record["proposal"]["provider_response_digest"]
    )
    trace["events"].insert(4, copy.deepcopy(proposal_event))
    for index, event in enumerate(trace["events"]):
        event["index"] = index
    _reseal_trace(record, trace)
    with pytest.raises(ValidationError, match="exactly one provider event"):
        evaluate_record(record)


def test_rejects_resealed_run_id_and_event_status_tampering(records: list[dict]) -> None:
    run_id_record = copy.deepcopy(records[0])
    trace = json.loads(run_id_record["decision"]["trace"])
    trace["run_id"] = "sha256:" + "0" * 64
    evidence = json.loads(run_id_record["decision"]["evidence"])
    evidence["run_id"] = trace["run_id"]
    run_id_record["decision"]["evidence"] = _json_line(evidence)
    _reseal_trace(run_id_record, trace)
    with pytest.raises(ValidationError, match="run_id derivation"):
        evaluate_record(run_id_record)

    status_record = copy.deepcopy(records[0])
    trace = json.loads(status_record["decision"]["trace"])
    proposal_event = next(
        event
        for event in trace["events"]
        if event["output_digest"] == status_record["proposal"]["provider_response_digest"]
    )
    proposal_event["status"] = "failed"
    _reseal_trace(status_record, trace)
    with pytest.raises(ValidationError, match="exactly one provider event"):
        evaluate_record(status_record)


def test_rejects_boolean_turn_and_resealed_authorization_status(records: list[dict]) -> None:
    turn_record = copy.deepcopy(records[0])
    turn_record["proposal"]["turn"] = True
    with pytest.raises(ValidationError, match="invalid proposal turn"):
        evaluate_record(turn_record)

    status_record = copy.deepcopy(records[0])
    trace = json.loads(status_record["decision"]["trace"])
    accepted = next(event for event in trace["events"] if event["kind"] == "action_accepted" and event["tool_id"])
    accepted["status"] = "final"
    _reseal_trace(status_record, trace)
    with pytest.raises(ValidationError, match="accepted action mismatch"):
        evaluate_record(status_record)


def test_missing_required_field_is_a_validation_error(records: list[dict]) -> None:
    del records[0]["decision"]["trace"]
    with pytest.raises(ValidationError, match="malformed record"):
        evaluate_record(records[0])


def test_rejects_duplicate_case_in_batch(records: list[dict]) -> None:
    records[2]["case_id"] = records[1]["case_id"]
    with pytest.raises(ValidationError, match="duplicate case_id"):
        evaluate_records(records)
