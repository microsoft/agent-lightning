# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import httpx
import pytest

from agentlightning.client import AgentLightningSyncClient
from agentlightning.credit.types import CreditFinalizationTimeout, ProcessCreditConfig
from agentlightning.schemas import Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutMetadata, RolloutState
from agentlightning.verl.agl_rollout_manager import AglRolloutManagerBase, EnqueuedRollout


@pytest.mark.asyncio
async def test_native_manager_keeps_equal_prompt_actions_and_failed_ordinal_gap(sealed_snapshot):
    snapshot = await sealed_snapshot(same_prompt=True, failed_first=True)

    def handle(request):
        if request.url.path.endswith("/credit"):
            return httpx.Response(200, json=snapshot.model_dump(mode="json"))
        return httpx.Response(
            200, json=[{**event.model_dump(mode="json"), "timestamp": 0} for event in snapshot.events]
        )

    manager = AglRolloutManagerBase(
        agl_base_url="http://offline",
        agl_key="",
        model="m",
        step=7,
        process_credit_config=ProcessCreditConfig(enabled=True, gamma=0.5),
    )
    manager.client.close()
    manager.client = AgentLightningSyncClient(transport=httpx.MockTransport(handle), base_url="http://offline")
    rollout = Rollout(
        rollout_id="r",
        input={},
        config=RolloutConfig(),
        metadata=RolloutMetadata.model_validate(
            {"process_credit": {"version": 1, "policy_version": 7, "group_id": "task", "group_size": 1}}
        ),
        status=RolloutLifecycleStatus(state=RolloutState.SUCCEEDED, last_attempt_id="a", created_at=0, updated_at=0),
    )
    enqueued = EnqueuedRollout(rollout_id="r", data_id="task", step=7, sample_idx_in_step=0, enqueue_time=0)
    completed = manager._build_completed_rollout(enqueued, rollout)
    assert completed.triplets is not None
    assert completed.process_credit is not None
    assert len(completed.triplets) == 3
    assert [turn.metadata["process_credit"]["action_index"] for turn in completed.triplets] == [0, 1, 2]
    assert [turn.metadata["process_credit"]["request_action_index"] for turn in completed.triplets] == [1, 2, 3]
    assert completed.final_reward == 1
    assert completed.process_credit.resolved.returns == pytest.approx((0.4, 0.4, 1))
    manager.client.close()


def test_terminal_unsealed_deadline_fails_with_original_ids():
    manager = AglRolloutManagerBase(
        agl_base_url="http://offline",
        agl_key="",
        model="m",
        step=7,
        poll_interval_seconds=0.001,
        process_credit_config=ProcessCreditConfig(enabled=True, finalization_timeout_seconds=0.002),
    )
    manager.client.close()
    manager.client = AgentLightningSyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(409, json={"detail": "pending"})),
        base_url="http://offline",
    )
    rollout = Rollout(
        rollout_id="r",
        input={},
        config=RolloutConfig(),
        metadata=RolloutMetadata.model_validate({"process_credit": {}}),
        status=RolloutLifecycleStatus(state=RolloutState.FAILED, last_attempt_id="a", created_at=0, updated_at=0),
    )
    with pytest.raises(CreditFinalizationTimeout, match="r/a"):
        manager._build_completed_rollout(
            EnqueuedRollout(rollout_id="r", data_id="task", step=7, sample_idx_in_step=0, enqueue_time=0), rollout
        )
    manager.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("selected_rollout,selected_attempt", [("foreign", "a"), ("r", "foreign")])
async def test_native_manager_rejects_a_seal_from_another_selected_attempt(
    sealed_snapshot, selected_rollout, selected_attempt
):
    snapshot = await sealed_snapshot()
    manager = AglRolloutManagerBase(
        agl_base_url="http://offline",
        agl_key="",
        model="m",
        step=7,
        process_credit_config=ProcessCreditConfig(enabled=True),
    )
    manager.client.close()
    manager.client = AgentLightningSyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=snapshot.model_dump(mode="json"))),
        base_url="http://offline",
    )
    rollout = Rollout(
        rollout_id=selected_rollout,
        input={},
        config=RolloutConfig(),
        metadata=RolloutMetadata.model_validate({"process_credit": {}}),
        status=RolloutLifecycleStatus(
            state=RolloutState.SUCCEEDED, last_attempt_id=selected_attempt, created_at=0, updated_at=0
        ),
    )
    with pytest.raises(ValueError, match="identity"):
        manager._build_completed_rollout(
            EnqueuedRollout(rollout_id=selected_rollout, data_id="task", step=7, sample_idx_in_step=0, enqueue_time=0),
            rollout,
        )
    manager.client.close()
