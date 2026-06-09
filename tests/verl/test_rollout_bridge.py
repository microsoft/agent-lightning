"""Tests for AglLiteRolloutBridge — the store-interaction methods (new code).

Tests the methods that replace AgentModeDaemon's store calls:
  - set_up_data_and_server (register model + enqueue)
    - run_until_all_finished (poll)
    - _fetch_rollout_result (fetch triplets)
  - clear_data_and_server (reset)

The tensor construction (get_train_data_batch) is covered separately in
test_rollout_bridge_train_batch.py.
"""

from __future__ import annotations

from typing import Any

import pytest

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas import RolloutCreate, RolloutState
from agl_lite.verl.rollout_bridge import (
    AglLiteRolloutBridge,
    CompletedRollout,
    _to_native,
    get_left_padded_ids_and_attention_mask,
    get_right_padded_ids_and_attention_mask,
)


class RecordingEnqueueHook(RolloutHooks):
    def __init__(self) -> None:
        self.requests: list[RolloutCreate] = []

    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        self.requests.append(request)
        metadata = dict(request.metadata or {})
        metadata["hooked"] = True
        return request.model_copy(update={"metadata": metadata})


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


class TestAsyncGroupFinishSelection:
    def _bridge(self) -> AglLiteRolloutBridge:
        return AglLiteRolloutBridge(
            agl_base_url="http://test",
            agl_key="test-key",
            train_rollout_n=2,
            model="test-model",
            tokenizer=None,
            mini_batch_size=2,
            pad_token_id=0,
        )

    def test_failed_rids_make_group_consumable_via_placeholder(self) -> None:
        bridge = self._bridge()
        bridge._rid_to_data_id = {"r1": "d1", "r2": "d1", "r3": "d2", "r4": "d2"}
        bridge._data_id_to_rids = {"d1": {"r1", "r2"}, "d2": {"r3", "r4"}}
        bridge._step_new_rids = {"r1", "r2", "r3", "r4"}
        bridge._rollout_status = {
            "r1": RolloutState.SUCCEEDED,
            "r2": RolloutState.FAILED,
            "r3": RolloutState.SUCCEEDED,
            "r4": RolloutState.FAILED,
        }

        assert bridge._finished_data_ids(bridge._active_groups()) == {"d1", "d2"}
        assert "r2" not in bridge._completed_rollouts


# ---- Bridge store-interaction tests ----


class TestBridgeStoreInteraction:
    """Test the NEW code in AglLiteRolloutBridge (store interaction via sync HTTP client).

    Uses the real agl-lite server (via ASGI transport) to verify HTTP calls.
    """

    @pytest.fixture()
    def app(self):
        from agl_lite.server.app import create_app

        return create_app(
            {
                "key": "test-key",
                "admin_key": "test-admin-key",
                "default_proxy": {
                    "model_name": "test-model",
                    "train": {"temperature": 1},
                    "val": {"temperature": 0},
                },
            }
        )

    @pytest.fixture()
    def bridge(self, app):
        """Create a bridge with a TestClient-backed sync HTTP client."""
        from starlette.testclient import TestClient

        # Use TestClient to trigger lifespan (initializes store)
        with TestClient(app) as client:
            client.headers.update({"Authorization": "Bearer test-key"})
            d = AglLiteRolloutBridge(
                agl_base_url="http://test",
                agl_key="test-key",
                train_rollout_n=1,
                model="test-model",
                tokenizer=None,
                mini_batch_size=2,
                pad_token_id=0,
            )
            d.client = client
            yield d

    def _set_up(self, bridge: AglLiteRolloutBridge, data, server_addresses, is_train: bool = True) -> None:
        bridge.clear_data_and_server()
        bridge.is_train = is_train
        bridge._register_and_enqueue(data, server_addresses, is_train)

    def _mark_succeeded(self, bridge: AglLiteRolloutBridge, rollout_id: str, attempt_id: str = "pod-1") -> None:
        running = bridge.client.patch(
            f"/api/rollouts/{rollout_id}",
            json={"status": {"state": "running", "last_attempt_id": attempt_id}},
        )
        running.raise_for_status()
        succeeded = bridge.client.patch(
            f"/api/rollouts/{rollout_id}",
            json={"status": {"state": "succeeded", "last_attempt_id": attempt_id}},
        )
        succeeded.raise_for_status()

    @pytest.mark.asyncio
    async def test_set_up_registers_model_and_enqueues(self, bridge: AglLiteRolloutBridge):
        """set_up_data_and_server registers model and creates rollouts."""
        data = {"prompt": ["What is 2+2?", "What is 3+3?"]}

        self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        # Should have queued 2 rollouts (1 per sample, train_rollout_n=1)
        assert bridge._total_tasks_queued == 2
        assert len(bridge._task_id_to_original_sample) == 2

        # Verify rollouts exist in the store via client
        for rid in bridge._task_id_to_original_sample:
            rollout = bridge._get_rollout(rid)
            assert rollout.status.state == RolloutState.QUEUING

    @pytest.mark.asyncio
    async def test_set_up_multiple_rollouts_per_sample(self, bridge: AglLiteRolloutBridge):
        """train_rollout_n > 1 creates multiple rollouts per sample."""
        bridge.train_rollout_n = 3
        data = {"prompt": ["hello"]}

        self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        assert bridge._total_tasks_queued == 3

    @pytest.mark.asyncio
    async def test_async_diff_enqueue_applies_on_enqueue_hook(self, bridge: AglLiteRolloutBridge):
        """Async top-up enqueue must use the same on_enqueue hook path as sync setup."""
        hook = RecordingEnqueueHook()
        bridge._hooks = hook
        bridge.train_rollout_n = 2
        data = {"prompt": ["hello"], "data_id": ["data-0"]}

        n_new = bridge._register_and_enqueue_diff(
            data,
            ["localhost:8000"],
            async_train_batch_size=1,
        )

        assert n_new == 1
        assert len(hook.requests) == 2
        assert bridge._total_tasks_queued == 2
        for rid in bridge._enqueue_order:
            rollout = bridge._get_rollout(rid)
            assert rollout.metadata.hooked is True
            assert rollout.input["data_id"] == "data-0"

    @pytest.mark.asyncio
    async def test_fetch_rollout_result_extracts_triplets(self, bridge: AglLiteRolloutBridge):
        """_fetch_rollout_result converts format=triplet events to CompletedRollout."""
        from agl_lite.schemas import EventCreate

        data = {"prompt": ["test"]}
        self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        rid = next(iter(bridge._task_id_to_original_sample))

        # Simulate what the gateway + agent would produce: model_request + reward
        bridge._post_event(
            rid,
            "pod-1",
            EventCreate(
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
        bridge._post_event(
            rid,
            "pod-1",
            EventCreate(
                event_type="reward",
                data={"value": 0.85, "message": "correct", "source": "agent", "reason": "computed"},
            ),
        )
        self._mark_succeeded(bridge, rid)

        legacy = bridge._fetch_rollout_result(rid)

        assert isinstance(legacy, CompletedRollout)
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
    async def test_fetch_rollout_result_treats_none_metadata_as_empty_dict(self, bridge: AglLiteRolloutBridge):
        """Parquet datasets may include a metadata column with null values."""
        from agl_lite.schemas import EventCreate

        data = {"prompt": ["test"], "metadata": [None]}
        self._set_up(bridge, data, ["localhost:8000"], is_train=True)

        rid = next(iter(bridge._task_id_to_original_sample))
        bridge._post_event(
            rid,
            "pod-1",
            EventCreate(
                event_type="model_request",
                data={
                    "response": [
                        {
                            "choices": [{"delta": {"content": "hi"}, "token_ids": [10]}],
                            "prompt_token_ids": [1],
                        }
                    ],
                },
            ),
        )
        bridge._post_event(
            rid,
            "pod-1",
            EventCreate(event_type="reward", data={"value": 1.0}),
        )
        self._mark_succeeded(bridge, rid)

        legacy = bridge._fetch_rollout_result(rid)

        assert legacy.metadata == {}
        assert legacy.task is not None
        assert legacy.task.metadata == {}

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, bridge: AglLiteRolloutBridge):
        data = {"prompt": ["test"]}
        self._set_up(bridge, data, ["localhost:8000"], is_train=True)
        assert bridge._total_tasks_queued == 1

        bridge.clear_data_and_server()

        assert bridge._total_tasks_queued == 0
        assert len(bridge._completed_rollouts) == 0
        assert len(bridge._task_id_to_original_sample) == 0
