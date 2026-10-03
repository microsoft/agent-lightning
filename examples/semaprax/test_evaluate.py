# Copyright (c) Microsoft. All rights reserved.

"""Agent Lightning publication integration tests for the Semaprax example."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from agentlightning.schemas import Event, Rollout
from agentlightning.server.app import create_app
from agentlightning.server.store import _events, _models, _rollouts, _terminal_order
from agentlightning.verl.agl_rollout_manager import AglRolloutManagerBase, EnqueuedRollout
from examples.semaprax.evaluate import publish_records
from examples.semaprax.evaluator import ValidationError

FIXTURE = Path(__file__).with_name("fixtures") / "records.json"
KEY = "semaprax-test-key"


class _Manager(AglRolloutManagerBase):
    def __init__(self, client: TestClient) -> None:
        self._test_client = client

    def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
        raw = self._test_client.get(f"/api/rollouts/{rollout_id}/events").json()
        triplet = self._test_client.get(f"/api/rollouts/{rollout_id}/events", params={"format": "triplet"}).json()
        return [Event.model_validate(item) for item in raw], [Event.model_validate(item) for item in triplet]


@pytest.fixture
def records() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def client():
    _rollouts.clear()
    _events.clear()
    _models.clear()
    _terminal_order.clear()
    app = create_app(
        {
            "key": KEY,
            "default_proxy": {
                "model_name": "test-model",
                "train": {"temperature": 1},
                "val": {"temperature": 0},
            },
        }
    )
    with TestClient(app, headers={"Authorization": f"Bearer {KEY}"}) as value:
        yield value
    _rollouts.clear()
    _events.clear()
    _models.clear()
    _terminal_order.clear()


def test_publish_persists_rewards_for_cpu_rollout_consumption(records: list[dict], client: TestClient) -> None:
    rollout_ids = publish_records(records, client=client)
    assert len(rollout_ids) == 3

    expected = {"semaprax-compliant": 3, "semaprax-denied_not_dispatched": -2, "semaprax-denied_but_dispatched": -3}
    manager = _Manager(client)
    for rollout_id in rollout_ids:
        rollout = Rollout.model_validate(client.get(f"/api/rollouts/{rollout_id}").json()["rollout"])
        events = client.get(f"/api/rollouts/{rollout_id}/events").json()
        assert [event["event_type"] for event in events] == [
            "semaprax_proposal",
            "semaprax_decision",
            "semaprax_dispatch",
            "semaprax_metrics",
            "reward",
        ]
        reward = expected[rollout.input["data_id"]]
        assert events[-1]["data"] == {"value": reward}
        completed = manager._build_completed_rollout(
            EnqueuedRollout(
                data_id=rollout.input["data_id"],
                rollout_id=rollout_id,
                step=0,
                sample_idx_in_step=0,
                enqueue_time=time.time(),
            ),
            rollout,
        )
        assert completed.final_reward == reward
        assert completed.triplets == []


class _FailureClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def _response(self, method: str, url: str, status: int, payload: Any) -> httpx.Response:
        return httpx.Response(status, json=payload, request=httpx.Request(method, f"http://test{url}"))

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("POST", url))
        if url == "/api/rollouts":
            return self._response("POST", url, 201, [{"rollout_id": "rollout-1"}])
        return self._response("POST", url, 500, {"detail": "failed"})

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("PATCH", url))
        return self._response("PATCH", url, 200, {})


def test_event_http_failure_propagates_without_retry(records: list[dict]) -> None:
    client = _FailureClient()
    with pytest.raises(httpx.HTTPStatusError):
        publish_records(records, client=client)
    event_url = "/api/rollouts/rollout-1/attempt/0/events"
    assert client.calls.count(("POST", event_url)) == 1


def test_invalid_batch_makes_no_http_calls(records: list[dict]) -> None:
    client = _FailureClient()
    invalid = copy.deepcopy(records)
    invalid[2]["proposal"]["stable_action_id"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        publish_records(invalid, client=client)
    assert client.calls == []
