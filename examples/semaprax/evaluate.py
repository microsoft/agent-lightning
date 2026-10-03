# Copyright (c) Microsoft. All rights reserved.

"""Evaluate captured Semaprax records offline or publish them to Agent Lightning."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from agentlightning.client import AgentLightningSyncClient
from examples.semaprax.evaluator import evaluate_records

DEFAULT_RECORDS = Path(__file__).with_name("fixtures") / "records.json"


def load_records(path: Path = DEFAULT_RECORDS) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError("records file must contain a JSON array of objects")
    return value


def _checked(response: httpx.Response) -> httpx.Response:
    response.raise_for_status()
    return response


def publish_records(
    records: list[dict[str, Any]],
    *,
    base_url: str | None = None,
    key: str | None = None,
    client: Any | None = None,
) -> list[str]:
    """Validate the complete batch, then publish its observations and rewards."""
    metrics = evaluate_records(records)
    owned_client: AgentLightningSyncClient | None = None
    if client is None:
        if not base_url or not key:
            raise ValueError("base_url and key are required when client is not supplied")
        owned_client = AgentLightningSyncClient(base_url=base_url.rstrip("/"), key=key, max_retries=0)
        client = owned_client
    active_client: Any = client

    rollout_ids: list[str] = []
    try:
        for record, score in zip(records, metrics, strict=True):
            created = _checked(
                active_client.post(
                    "/api/rollouts",
                    json=[
                        {
                            "input": {
                                "data_id": f"semaprax-{record['case_id']}",
                                "record_schema": record["schema"],
                            },
                            "is_train": False,
                        }
                    ],
                )
            ).json()[0]
            rollout_id = created["rollout_id"]
            rollout_ids.append(rollout_id)
            try:
                _checked(active_client.patch(f"/api/rollouts/{rollout_id}", json={"status": {"state": "running"}}))

                event_url = f"/api/rollouts/{rollout_id}/attempt/0/events"
                event_payloads = (
                    (
                        "semaprax_proposal",
                        {
                            "schema": record["proposal"]["schema"],
                            "stable_action_id": record["proposal"]["stable_action_id"],
                            "turn": record["proposal"]["turn"],
                            "tool_id": record["proposal"]["tool_id"],
                            "arguments_json": record["proposal"]["arguments_json"],
                            "provider_response_digest": record["proposal"]["provider_response_digest"],
                        },
                    ),
                    (
                        "semaprax_decision",
                        {
                            "status": record["decision"]["status"],
                            "trace_digest": record["decision"]["trace_digest"],
                            "evidence_digest": record["decision"]["evidence_digest"],
                        },
                    ),
                    ("semaprax_dispatch", record["dispatch"]),
                    (
                        "semaprax_metrics",
                        {
                            "task_outcome": score.task_outcome,
                            "policy_conformance": score.policy_conformance,
                        },
                    ),
                    ("reward", {"value": score.reward}),
                )
                # Event POSTs are deliberately never retried: a transport failure may
                # happen after the server committed a non-idempotent event.
                for event_type, data in event_payloads:
                    _checked(active_client.post(event_url, json={"event_type": event_type, "data": data}))
                _checked(
                    active_client.patch(
                        f"/api/rollouts/{rollout_id}",
                        json={"status": {"state": "succeeded", "last_attempt_id": "0"}},
                    )
                )
            except Exception as error:
                # A success response can be lost after commit. The server rejects
                # changing that terminal state; preserve the publication error.
                try:
                    _checked(
                        active_client.patch(
                            f"/api/rollouts/{rollout_id}",
                            json={"status": {"state": "failed", "error_message": "Semaprax publication failed"}},
                        )
                    )
                except Exception:
                    error.add_note(f"Could not mark rollout {rollout_id} failed after publication failed.")
                raise
    finally:
        if owned_client is not None:
            owned_client.close()
    return rollout_ids


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--agl-base-url")
    parser.add_argument("--agl-key")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if bool(args.agl_base_url) != bool(args.agl_key):
        raise SystemExit("--agl-base-url and --agl-key must be supplied together")
    records = load_records(args.records)
    metrics = evaluate_records(records)
    output: dict[str, Any] = {"metrics": [item.as_dict() for item in metrics]}
    if args.agl_base_url:
        output["rollout_ids"] = publish_records(records, base_url=args.agl_base_url, key=args.agl_key)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
