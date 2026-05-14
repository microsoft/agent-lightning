"""Tests for AglLiteRolloutBridge — the store-interaction methods (new code).

Tests the methods that replace AgentModeDaemon's store calls:
  - set_up_data_and_server (register model + enqueue)
  - run_until_all_finished (poll)
  - _async_fetch_rollout_result (fetch triplets)
  - cleanup_agent_jobs (optional scoped Kubernetes Job cleanup)
  - clear_data_and_server (reset)

The tensor construction (get_train_data_batch) is covered separately in
test_rollout_bridge_train_batch.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from agl_lite.client import AglLiteClient
from agl_lite.verl.rollout_bridge import (
    AglLiteRolloutBridge,
    RolloutLegacy,
    _to_native,
    get_left_padded_ids_and_attention_mask,
    get_right_padded_ids_and_attention_mask,
)


class FakeCleanupK8sClient:
    def __init__(self, jobs: list[dict[str, Any]]) -> None:
        self.jobs = {job["metadata"]["name"]: job for job in jobs}
        self.list_calls: list[tuple[str, str]] = []
        self.deleted: list[tuple[str, str]] = []

    async def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        self.list_calls.append((namespace, label_selector))
        return list(self.jobs.values())

    async def delete_job(self, name: str, namespace: str) -> None:
        self.deleted.append((name, namespace))
        self.jobs.pop(name, None)


def _cleanup_job(name: str, rollout_id: str | None, managed: bool = True) -> dict[str, Any]:
    labels: dict[str, str] = {}
    if managed:
        labels["app.kubernetes.io/managed-by"] = "agl-lite"
    if rollout_id is not None:
        labels["agl-lite/rollout-id"] = rollout_id
    return {"metadata": {"name": name, "labels": labels}}


# ---- Utility tests (copied code, sanity check) ----


class TestUtilities:
    def test_left_pad(self):
        ids, mask = get_left_padded_ids_and_attention_mask([1, 2, 3], 5, 0)
        assert ids == [0, 0, 1, 2, 3]
        assert mask == [0, 0, 1, 1, 1]

    def test_left_pad_truncate(self):
        ids, mask = get_left_padded_ids_and_attention_mask([1, 2, 3, 4, 5], 3, 0)
        assert ids == [3, 4, 5]  # keeps last max_length tokens
        assert mask == [1, 1, 1]

    def test_right_pad(self):
        ids, mask = get_right_padded_ids_and_attention_mask([1, 2], 4, 0)
        assert ids == [1, 2, 0, 0]
        assert mask == [1, 1, 0, 0]

    def test_right_pad_truncate(self):
        ids, mask = get_right_padded_ids_and_attention_mask([1, 2, 3, 4], 2, 0)
        assert ids == [1, 2]
        assert mask == [1, 1]

    def test_to_native(self):
        import numpy as np

        result = _to_native({"a": np.int64(1), "b": [np.float64(2.5)]})
        assert result == {"a": 1, "b": [2.5]}
        assert isinstance(result["a"], int)


class TestAgentJobCleanup:
    def _bridge(
        self,
        k8s: FakeCleanupK8sClient,
        *,
        cleanup_agent_jobs: bool = True,
        cleanup_namespace: str | None = "agent-ns",
    ) -> AglLiteRolloutBridge:
        return AglLiteRolloutBridge(
            agl_base_url="http://test",
            agl_key="test-key",
            train_rollout_n=1,
            train_information={"model": "test-model"},
            tokenizer=None,
            mini_batch_size=2,
            pad_token_id=0,
            cleanup_agent_jobs=cleanup_agent_jobs,
            cleanup_namespace=cleanup_namespace,
            cleanup_k8s_client=k8s,
        )

    def test_cleanup_requires_namespace_when_enabled(self) -> None:
        with pytest.raises(ValueError, match="cleanup_namespace"):
            AglLiteRolloutBridge(
                agl_base_url="http://test",
                agl_key="test-key",
                train_rollout_n=1,
                train_information={"model": "test-model"},
                tokenizer=None,
                mini_batch_size=2,
                pad_token_id=0,
                cleanup_agent_jobs=True,
            )

    @pytest.mark.asyncio
    async def test_cleanup_deletes_only_tracked_managed_jobs(self) -> None:
        k8s = FakeCleanupK8sClient(
            [
                _cleanup_job("job-r1", "r1"),
                _cleanup_job("job-r2", "r2"),
                _cleanup_job("job-r3", "r3"),
                _cleanup_job("foreign-r1", "r1", managed=False),
                _cleanup_job("missing-rollout-label", None),
            ]
        )
        bridge = self._bridge(k8s)
        bridge._enqueue_order = ["r1", "r2"]

        await bridge._async_cleanup_tracked_agent_jobs()

        assert k8s.list_calls == [("agent-ns", "app.kubernetes.io/managed-by=agl-lite")]
        assert set(k8s.deleted) == {("job-r1", "agent-ns"), ("job-r2", "agent-ns")}
        assert set(k8s.jobs) == {"job-r3", "foreign-r1", "missing-rollout-label"}

    @pytest.mark.asyncio
    async def test_cleanup_disabled_does_not_touch_k8s(self) -> None:
        k8s = FakeCleanupK8sClient([_cleanup_job("job-r1", "r1")])
        bridge = self._bridge(k8s, cleanup_agent_jobs=False, cleanup_namespace=None)
        bridge._enqueue_order = ["r1"]

        await bridge._async_cleanup_tracked_agent_jobs()

        assert k8s.list_calls == []
        assert k8s.deleted == []
        assert set(k8s.jobs) == {"job-r1"}

    def test_clear_does_not_cleanup_k8s_jobs(self) -> None:
        k8s = FakeCleanupK8sClient([_cleanup_job("job-r1", "r1")])
        bridge = self._bridge(k8s)
        bridge._enqueue_order = ["r1"]
        bridge._total_tasks_queued = 1

        bridge.clear_data_and_server()

        assert k8s.list_calls == []
        assert k8s.deleted == []
        assert set(k8s.jobs) == {"job-r1"}


# ---- Bridge store-interaction tests ----


class TestBridgeStoreInteraction:
    """Test the NEW code in AglLiteRolloutBridge (store interaction via AglLiteClient).

    Uses the real agl-lite server (via ASGI transport) to verify HTTP calls.
    """

    @pytest.fixture()
    def app(self):
        from agl_lite.server.app import create_app
        from agl_lite.server.config import ServerSettings

        return create_app(ServerSettings(key="test-key"))

    @pytest.fixture()
    def bridge(self, app):
        """Create a bridge with an AglLiteClient backed by ASGI transport."""
        import httpx
        from starlette.testclient import TestClient

        # Use TestClient to trigger lifespan (initializes store)
        with TestClient(app) as _tc:
            transport = httpx.ASGITransport(app=app)
            async_client = httpx.AsyncClient(
                base_url="http://test",
                headers={"Authorization": "Bearer test-key"},
                transport=transport,
            )

            client = AglLiteClient(base_url="http://test", agl_key="test-key")
            client._client = async_client

            d = AglLiteRolloutBridge(
                agl_base_url="http://test",
                agl_key="test-key",
                train_rollout_n=1,
                train_information={"model": "test-model"},
                tokenizer=None,
                mini_batch_size=2,
                pad_token_id=0,
            )
            d.client = client
            yield d

    async def _set_up(self, bridge: AglLiteRolloutBridge, data, server_addresses, is_train: bool = True) -> None:
        bridge.clear_data_and_server()
        bridge.is_train = is_train
        await bridge._async_register_and_enqueue(bridge.client, data, server_addresses, is_train)

    @pytest.mark.asyncio
    async def test_set_up_registers_model_and_enqueues(self, bridge: AglLiteRolloutBridge):
        """set_up_data_and_server registers model and creates rollouts."""
        data = {"prompt": ["What is 2+2?", "What is 3+3?"]}

        await self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        # Should have queued 2 rollouts (1 per sample, train_rollout_n=1)
        assert bridge._total_tasks_queued == 2
        assert len(bridge._task_id_to_original_sample) == 2

        # Verify rollouts exist in the store via client
        for rid in bridge._task_id_to_original_sample:
            rollout = await bridge.client.get_rollout(rid)
            assert rollout.status == "queuing"

    @pytest.mark.asyncio
    async def test_set_up_multiple_rollouts_per_sample(self, bridge: AglLiteRolloutBridge):
        """train_rollout_n > 1 creates multiple rollouts per sample."""
        bridge.train_rollout_n = 3
        data = {"prompt": ["hello"]}

        await self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        assert bridge._total_tasks_queued == 3

    @pytest.mark.asyncio
    async def test_fetch_rollout_result_extracts_triplets(self, bridge: AglLiteRolloutBridge):
        """_async_fetch_rollout_result converts format=triplet events to RolloutLegacy."""
        from agl_lite.schemas.api import PostEventRequest

        data = {"prompt": ["test"]}
        await self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        rid = next(iter(bridge._task_id_to_original_sample))

        # Simulate what the gateway + agent would produce: model_request + reward
        await bridge.client.post_event(
            rid,
            "pod-1",
            PostEventRequest(
                event_type="model_request",
                data={
                    "request": {"model": "m", "messages": [], "return_token_ids": True},
                    "response": [
                        {
                            "choices": [{"delta": {"content": "hi"}, "token_ids": [10, 20]}],
                            "prompt_token_ids": [1, 2, 3],
                        },
                        {"choices": [{"delta": {"content": "!"}, "token_ids": [30]}]},
                    ],
                    "server": {"model": "m", "version": 1, "endpoint": "http://x"},
                },
            ),
        )
        await bridge.client.post_event(
            rid,
            "pod-1",
            PostEventRequest(
                event_type="reward",
                data={"value": 0.85, "message": "correct", "source": "agent", "reason": "computed"},
            ),
        )

        legacy = await bridge._async_fetch_rollout_result(rid)

        assert isinstance(legacy, RolloutLegacy)
        assert legacy.rollout_id == rid
        assert legacy.final_reward == 0.85
        assert legacy.reward_source == "agent"
        assert legacy.reward_reason == "computed"
        assert len(legacy.events) == 2
        assert len(legacy.triplet_events) == 2
        assert legacy.events[0]["data"]["request"]["model"] == "m"
        assert legacy.triplet_events[0]["data"]["prompt_token_ids"] == [1, 2, 3]
        assert legacy.triplets is not None
        assert len(legacy.triplets) == 1

        t = legacy.triplets[0]
        assert t.prompt["token_ids"] == [1, 2, 3]
        assert t.response["token_ids"] == [10, 20, 30]
        assert t.reward == 0.85  # assigned to last triplet

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, bridge: AglLiteRolloutBridge):
        data = {"prompt": ["test"]}
        await self._set_up(bridge, data, ["localhost:8000"], is_train=True)
        assert bridge._total_tasks_queued == 1

        bridge.clear_data_and_server()

        assert bridge._total_tasks_queued == 0
        assert len(bridge._completed_rollouts) == 0
        assert len(bridge._task_id_to_original_sample) == 0
