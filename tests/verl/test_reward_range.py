# Copyright (c) Microsoft. All rights reserved.

"""Keep invalid outcome rewards out of float32 training scores and metrics."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("verl")

import torch
from fastapi.testclient import TestClient

from agentlightning.schemas import Event, Rollout
from agentlightning.server.app import create_app
from agentlightning.verl.agl_rollout_manager import AglRolloutManagerBase, CompletedRollout, EnqueuedRollout, Triplet
from agentlightning.verl.rollout_adapter import RolloutAdapter


def _adapter(level: str = "transition", fillna: float = 0.0) -> RolloutAdapter:
    return RolloutAdapter(
        max_prompt_length=2,
        max_response_length=2,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level=level,
        reward_fillna_value=fillna,
    )


def _rollout(reward: float | None) -> CompletedRollout:
    return CompletedRollout(
        rollout_id="unsafe-reward-rollout",
        data_id="example",
        step=0,
        sample_idx_in_step=0,
        enqueue_time=0.0,
        final_reward=reward,
        triplets=[Triplet(prompt={"token_ids": [1]}, response={"token_ids": [2], "log_probs": [-0.1]})],
    )


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf"), 1e40, -1e40])
@pytest.mark.parametrize("level", ["transition", "trajectory"])
def test_training_rejects_rewards_that_cannot_be_finite_float32(reward: float, level: str) -> None:
    with pytest.raises(ValueError, match=r"unsafe-reward-rollout.*finite"):
        _adapter(level).get_train_data_batch([_rollout(reward)])


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
def test_validation_rejects_the_same_invalid_reward(reward: float) -> None:
    with pytest.raises(ValueError, match=r"unsafe-reward-rollout.*finite"):
        _adapter().get_test_metrics([_rollout(reward)])


@pytest.mark.parametrize("fillna", [float("nan"), float("inf"), 1e40])
def test_invalid_missing_reward_fallback_is_not_silently_used(fillna: float) -> None:
    with pytest.raises(ValueError, match=r"unsafe-reward-rollout.*finite"):
        _adapter(fillna=fillna).get_train_data_batch([_rollout(None)])


@pytest.mark.parametrize("reward", [0.0, -0.25, 1.5, torch.finfo(torch.float32).max, -torch.finfo(torch.float32).max])
def test_finite_rewards_keep_their_existing_scores(reward: float) -> None:
    batch, metrics = _adapter().get_train_data_batch([_rollout(reward)])
    assert torch.isfinite(batch.batch["token_level_scores"]).all()
    assert batch.batch["token_level_scores"].sum().item() == pytest.approx(reward)
    assert metrics["training/reward"] == reward


def test_missing_reward_keeps_configured_finite_fallback() -> None:
    batch, metrics = _adapter(fillna=-0.5).get_train_data_batch([_rollout(None)])
    assert batch.batch["token_level_scores"].sum().item() == -0.5
    assert metrics["training/n_rollouts_w_reward"] == 0


def test_validation_preserves_large_finite_rewards_without_float32_conversion() -> None:
    assert _adapter().get_test_metrics([_rollout(1e40)])["val/reward"] == 1e40


def test_finite_json_reward_is_rejected_before_training_cast_overflows() -> None:
    app = create_app(
        {"key": "test", "default_proxy": {"model_name": "test", "train": {"temperature": 1}, "val": {"temperature": 0}}}
    )
    headers = {"Authorization": "Bearer test"}
    with TestClient(app) as client:
        created = client.post("/api/rollouts", json=[{"input": {"data_id": "example"}}], headers=headers)
        created.raise_for_status()
        rollout = Rollout.model_validate(created.json()[0])
        rid = rollout.rollout_id
        try:
            event_url = f"/api/rollouts/{rid}/attempt/0/events"
            response = client.post(
                event_url,
                json={
                    "event_type": "model_request",
                    "data": {"response": {"prompt_token_ids": [1], "choices": [{"token_ids": [2]}]}},
                },
                headers=headers,
            )
            response.raise_for_status()
            client.post(
                event_url, json={"event_type": "reward", "data": {"value": 1e40}}, headers=headers
            ).raise_for_status()
            client.patch(
                f"/api/rollouts/{rid}",
                json={"status": {"state": "running", "last_attempt_id": "0"}},
                headers=headers,
            ).raise_for_status()
            terminal = client.patch(f"/api/rollouts/{rid}", json={"status": {"state": "succeeded"}}, headers=headers)
            terminal.raise_for_status()
            rollout = Rollout.model_validate(terminal.json())

            class Manager(AglRolloutManagerBase):
                def __init__(self) -> None:
                    pass

                def _fetch_rollout_events(self, rollout_id: str) -> tuple[list[Event], list[Event]]:
                    raw = client.get(f"/api/rollouts/{rollout_id}/events", headers=headers)
                    trimmed = client.get(
                        f"/api/rollouts/{rollout_id}/events", params={"format": "triplet"}, headers=headers
                    )
                    raw.raise_for_status()
                    trimmed.raise_for_status()
                    return [Event.model_validate(e) for e in raw.json()], [
                        Event.model_validate(e) for e in trimmed.json()
                    ]

            completed = Manager()._build_completed_rollout(
                EnqueuedRollout(data_id="example", rollout_id=rid, step=0, sample_idx_in_step=0, enqueue_time=0),
                rollout,
            )
            assert completed.final_reward == 1e40
            with pytest.raises(ValueError, match="finite float32") as exc:
                _adapter().get_train_data_batch([completed])
            assert rid in str(exc.value)
        finally:
            client.delete(f"/api/rollouts/{rid}", headers=headers).raise_for_status()
