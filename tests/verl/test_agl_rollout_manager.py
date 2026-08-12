# Copyright (c) Microsoft. All rights reserved.

"""Tests for VERL rollout manager event conversion."""

from __future__ import annotations

from agentlightning.schemas import Event, Rollout, RolloutConfig, RolloutLifecycleStatus, RolloutState
from agentlightning.verl.agl_rollout_manager import AglRolloutManagerBase, EnqueuedRollout


class _Manager(AglRolloutManagerBase):
    def __init__(self, triplet_events: list[Event]) -> None:
        self._triplet_events = triplet_events

    def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
        return self._triplet_events, self._triplet_events


def _event(event_type: str, data: dict) -> Event:
    return Event(event_type=event_type, rollout_id="rollout-1", attempt_id="0", timestamp=0.0, data=data)


def _rollout() -> Rollout:
    return Rollout(
        rollout_id="rollout-1",
        input={"prompt": "hi"},
        config=RolloutConfig(),
        status=RolloutLifecycleStatus(
            state=RolloutState.SUCCEEDED,
            last_attempt_id="0",
            created_at=0.0,
            updated_at=0.0,
        ),
    )


def test_build_completed_rollout_skips_error_and_empty_model_requests() -> None:
    manager = _Manager(
        [
            _event(
                "model_request",
                {
                    "prompt_token_ids": [],
                    "response_token_ids": [],
                    "http_status": 400,
                    "status": "error",
                },
            ),
            _event(
                "model_request",
                {
                    "prompt_token_ids": [1],
                    "response_token_ids": [],
                    "http_status": 200,
                    "status": "ok",
                },
            ),
            _event(
                "model_request",
                {
                    "prompt_token_ids": [1],
                    "response_token_ids": [2],
                    "response_log_probs": [-0.1],
                    "http_status": 200,
                    "status": "ok",
                    "server": {"model": "test-model", "version": 3},
                },
            ),
            _event("reward", {"value": 1.0}),
        ]
    )

    completed = manager._build_completed_rollout(
        EnqueuedRollout(
            data_id="data-1",
            rollout_id="rollout-1",
            step=0,
            sample_idx_in_step=0,
            enqueue_time=0.0,
        ),
        _rollout(),
    )

    assert completed.final_reward == 1.0
    assert completed.triplets is not None
    assert len(completed.triplets) == 1
    assert completed.triplets[0].prompt["token_ids"] == [1]
    assert completed.triplets[0].response["token_ids"] == [2]
    assert completed.triplets[0].response["log_probs"] == [-0.1]
    assert completed.triplets[0].reward == 1.0
