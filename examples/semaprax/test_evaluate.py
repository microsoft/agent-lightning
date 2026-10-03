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


@pytest.mark.parametrize("case_id", [[], {}, None, 123, True])
def test_non_string_case_id_makes_no_http_calls(records: list[dict], case_id: object) -> None:
    client = _FailureClient()
    records[0]["case_id"] = case_id
    with pytest.raises(ValidationError, match="case_id must be a string"):
        publish_records(records, client=client)
    assert client.calls == []


class _PublicationFailureClient:
    """Inject failures around requests to the real in-process rollout store."""

    def __init__(
        self, client: TestClient, failure: str, cleanup_failure: str | None = None, failure_on_case: int = 0
    ) -> None:
        self.client = client
        self.failure = failure
        self.cleanup_failure = cleanup_failure
        self.failure_on_case = failure_on_case
        self.calls: list[tuple[str, str, Any]] = []
        self.created_ids: list[str] = []
        self.rollout_id: str | None = None
        self.close_calls = 0

    def _failure_response(self, method: str, url: str) -> httpx.Response:
        return httpx.Response(503, request=httpx.Request(method, f"http://test{url}"))

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("POST", url, kwargs["json"]))
        if url == "/api/rollouts":
            response = self.client.post(url, **kwargs)
            rollout_id = response.json()[0]["rollout_id"]
            assert isinstance(rollout_id, str)
            self.rollout_id = rollout_id
            self.created_ids.append(rollout_id)
            return response
        should_fail = len(self.created_ids) - 1 == self.failure_on_case
        if should_fail and self.failure == "event_http":
            return self._failure_response("POST", url)
        response = self.client.post(url, **kwargs)
        if should_fail and self.failure == "event_transport_after_commit":
            raise httpx.ReadTimeout("publication response lost", request=response.request)
        return response

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        self.calls.append(("PATCH", url, kwargs["json"]))
        state = kwargs["json"]["status"]["state"]
        if len(self.created_ids) - 1 != self.failure_on_case:
            return self.client.patch(url, **kwargs)
        if self.failure == f"{state}_http" or (state == "failed" and self.cleanup_failure == "http"):
            return self._failure_response("PATCH", url)
        if state == "failed" and self.cleanup_failure == "transport":
            raise httpx.ConnectError("cleanup unavailable", request=httpx.Request("PATCH", f"http://test{url}"))
        response = self.client.patch(url, **kwargs)
        if state == "succeeded" and self.failure == "succeeded_transport_after_commit":
            raise httpx.ReadTimeout("publication response lost", request=response.request)
        return response

    def close(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("failure", "expected_state", "expected_events"),
    [
        ("running_http", "failed", 0),
        ("event_http", "failed", 0),
        ("event_transport_after_commit", "failed", 1),
        ("succeeded_http", "failed", 5),
        ("succeeded_transport_after_commit", "succeeded", 5),
    ],
)
def test_publication_failure_attempts_terminal_cleanup_without_event_retry(
    records: list[dict], client: TestClient, failure: str, expected_state: str, expected_events: int
) -> None:
    failing = _PublicationFailureClient(client, failure)
    error_type = httpx.ReadTimeout if "transport" in failure else httpx.HTTPStatusError
    with pytest.raises(error_type):
        publish_records(records, client=failing)

    assert failing.rollout_id is not None
    rollout_id = failing.rollout_id
    rollout = client.get(f"/api/rollouts/{rollout_id}").json()["rollout"]
    assert rollout["status"]["state"] == expected_state
    assert len(_rollouts) == 1
    events = client.get(f"/api/rollouts/{rollout_id}/events").json()
    assert len(events) == expected_events
    event_types = [
        payload["event_type"] for method, url, payload in failing.calls if method == "POST" and "events" in url
    ]
    assert len(event_types) == len(set(event_types))
    assert (
        sum(method == "PATCH" and payload["status"]["state"] == "failed" for method, _, payload in failing.calls) == 1
    )


@pytest.mark.parametrize("cleanup_failure", ["http", "transport"])
def test_failed_cleanup_preserves_original_publication_error(
    records: list[dict], client: TestClient, cleanup_failure: str
) -> None:
    failing = _PublicationFailureClient(client, "event_http", cleanup_failure)
    with pytest.raises(httpx.HTTPStatusError) as caught:
        publish_records(records, client=failing)

    assert failing.rollout_id is not None
    assert caught.value.request.url.path == f"/api/rollouts/{failing.rollout_id}/attempt/0/events"
    assert (
        sum(method == "PATCH" and payload["status"]["state"] == "failed" for method, _, payload in failing.calls) == 1
    )
    assert [method for method, url, _ in failing.calls if "events" in url] == ["POST"]
    assert any(failing.rollout_id in note for note in caught.value.__notes__)


def test_later_publication_failure_preserves_prior_success(records: list[dict], client: TestClient) -> None:
    failing = _PublicationFailureClient(client, "event_http", failure_on_case=1)
    with pytest.raises(httpx.HTTPStatusError):
        publish_records(records, client=failing)

    assert len(failing.created_ids) == 2
    states = [
        client.get(f"/api/rollouts/{rollout_id}").json()["rollout"]["status"]["state"]
        for rollout_id in failing.created_ids
    ]
    assert states == ["succeeded", "failed"]


def test_owned_client_closes_after_publication_failure(
    records: list[dict], client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = _PublicationFailureClient(client, "event_http")
    monkeypatch.setattr("examples.semaprax.evaluate.AgentLightningSyncClient", lambda **kwargs: failing)
    with pytest.raises(httpx.HTTPStatusError):
        publish_records(records, base_url="http://test", key=KEY)
    assert failing.close_calls == 1
