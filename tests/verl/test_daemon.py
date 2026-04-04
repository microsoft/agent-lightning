"""Tests for AglLiteDaemon — the store-interaction methods (new code).

Tests the 4 methods that replace AgentModeDaemon's store calls:
  - set_up_data_and_server (register model + enqueue)
  - run_until_all_finished (poll)
  - _async_validate_data (fetch triplets)
  - clear_data_and_server (reset)

The tensor construction (get_train_data_batch) is NOT tested here — it requires
torch/verl and is copied verbatim from Agent Lightning.
"""

from __future__ import annotations

import asyncio

import pytest

from agl_lite.client import AglLiteClient
from agl_lite.verl.daemon import (
    AglLiteDaemon,
    RolloutLegacy,
    Triplet,
    _to_native,
    get_left_padded_ids_and_attention_mask,
    get_right_padded_ids_and_attention_mask,
)


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


# ---- Daemon store-interaction tests ----


class TestDaemonStoreInteraction:
    """Test the NEW code in AglLiteDaemon (store interaction via AglLiteClient).

    Uses the real agl-lite server (via ASGI transport) to verify HTTP calls.
    """

    @pytest.fixture()
    def app(self):
        from agl_lite.server.app import create_app
        from agl_lite.server.config import ServerSettings
        return create_app(ServerSettings(key="test-key"))

    @pytest.fixture()
    def daemon(self, app):
        """Create a daemon with an AglLiteClient backed by ASGI transport."""
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

            d = AglLiteDaemon(
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

    @pytest.mark.asyncio
    async def test_set_up_registers_model_and_enqueues(self, daemon: AglLiteDaemon):
        """set_up_data_and_server registers model and creates rollouts."""
        data = {"prompt": ["What is 2+2?", "What is 3+3?"]}

        await daemon._async_set_up(data, ["localhost:8000"], is_train=True)

        # Should have queued 2 rollouts (1 per sample, train_rollout_n=1)
        assert daemon._total_tasks_queued == 2
        assert len(daemon._task_id_to_original_sample) == 2

        # Verify rollouts exist in the store via client
        for rid in daemon._task_id_to_original_sample:
            rollout = await daemon.client.get_rollout(rid)
            assert rollout.status == "queuing"

    @pytest.mark.asyncio
    async def test_set_up_multiple_rollouts_per_sample(self, daemon: AglLiteDaemon):
        """train_rollout_n > 1 creates multiple rollouts per sample."""
        daemon.train_rollout_n = 3
        data = {"prompt": ["hello"]}

        await daemon._async_set_up(data, ["localhost:8000"], is_train=True)

        assert daemon._total_tasks_queued == 3

    @pytest.mark.asyncio
    async def test_validate_data_extracts_triplets(self, daemon: AglLiteDaemon):
        """_async_validate_data converts format=triplet events to RolloutLegacy."""
        from agl_lite.schemas.api import PostEventRequest

        data = {"prompt": ["test"]}
        await daemon._async_set_up(data, ["localhost:8000"], is_train=True)

        rid = list(daemon._task_id_to_original_sample.keys())[0]

        # Simulate what the gateway + agent would produce: model_request + reward
        await daemon.client.post_event(
            rid, "pod-1",
            PostEventRequest(
                event_type="model_request",
                data={
                    "request": {"model": "m", "messages": [], "return_token_ids": True},
                    "response": [
                        {"choices": [{"delta": {"content": "hi"}, "token_ids": [10, 20]}],
                         "prompt_token_ids": [1, 2, 3]},
                        {"choices": [{"delta": {"content": "!"}, "token_ids": [30]}]},
                    ],
                    "server": {"model": "m", "version": 1, "endpoint": "http://x"},
                },
            ),
        )
        await daemon.client.post_event(
            rid, "pod-1",
            PostEventRequest(
                event_type="reward",
                data={"value": 0.85, "message": "correct"},
            ),
        )

        legacy = await daemon._async_validate_data(rid)

        assert isinstance(legacy, RolloutLegacy)
        assert legacy.rollout_id == rid
        assert legacy.final_reward == 0.85
        assert len(legacy.triplets) == 1

        t = legacy.triplets[0]
        assert t.prompt["token_ids"] == [1, 2, 3]
        assert t.response["token_ids"] == [10, 20, 30]
        assert t.reward == 0.85  # assigned to last triplet

    @pytest.mark.asyncio
    async def test_clear_resets_state(self, daemon: AglLiteDaemon):
        data = {"prompt": ["test"]}
        await daemon._async_set_up(data, ["localhost:8000"], is_train=True)
        assert daemon._total_tasks_queued == 1

        daemon.clear_data_and_server()

        assert daemon._total_tasks_queued == 0
        assert len(daemon._completed_rollouts_v0) == 0
        assert len(daemon._task_id_to_original_sample) == 0
