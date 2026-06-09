"""Tests for AglLiteRolloutBridge training-batch construction."""

from __future__ import annotations

from typing import ClassVar

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

from agl_lite.verl.rollout_bridge import AglLiteRolloutBridge, RolloutLegacy, Triplet


class FakeTokenizer:
    eos_token_id = 99
    all_special_ids: ClassVar[list[int]] = []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


def _bridge(level: str, reward_fillna_value: float = 0.0) -> AglLiteRolloutBridge:
    return AglLiteRolloutBridge(
        agl_base_url="http://test",
        agl_key="test-key",
        train_rollout_n=1,
        model="test-model",
        tokenizer=FakeTokenizer(),
        mini_batch_size=2,
        pad_token_id=0,
        reward_fillna_value=reward_fillna_value,
        trace_aggregator={
            "level": level,
            "trajectory_max_prompt_length": 4,
            "trajectory_max_response_length": 8,
        },
    )


def _install_rollout(bridge: AglLiteRolloutBridge, rollout: RolloutLegacy, data_id: str = "data-1") -> None:
    bridge.is_train = True
    bridge._enqueue_order = [rollout.rollout_id]
    bridge._total_tasks_queued = 1
    bridge._task_id_to_original_sample = {rollout.rollout_id: {"data_id": data_id}}
    bridge._completed_rollouts = {rollout.rollout_id: rollout}


def test_transition_uses_all_triplets_not_only_last() -> None:
    bridge = _bridge("transition")
    rollout = RolloutLegacy(
        rollout_id="r1",
        final_reward=0.75,
        triplets=[
            Triplet(prompt={"token_ids": [10, 11]}, response={"token_ids": [12]}),
            Triplet(prompt={"token_ids": [20, 21]}, response={"token_ids": [22, 23]}),
        ],
    )
    _install_rollout(bridge, rollout)

    batch, metrics = bridge.get_train_data_batch(
        max_prompt_length=4,
        max_response_length=3,
        device=__import__("torch").device("cpu"),
        global_steps=0,
    )

    assert metrics["training/n_triplets"] == 2
    assert batch.batch["prompts"].tolist() == [[0, 0, 10, 11], [0, 0, 20, 21]]
    assert batch.batch["responses"].tolist() == [[12, 0, 0], [22, 23, 0]]
    assert "response_mask" not in batch.batch
    assert batch.non_tensor_batch["data_id_list"].tolist() == ["data-1", "data-1"]
    assert batch.non_tensor_batch["rollout_id_list"].tolist() == ["r1", "r1"]
    assert batch.non_tensor_batch["turn_index_list"].tolist() == [0, 1]
    assert [round(float(x), 2) for x in batch.batch["token_level_scores"].sum(-1).tolist()] == [0.75, 0.75]
    assert metrics["training/n_rollouts_w_reward"] == 1
    assert metrics["training/n_rollouts_w_any_reward"] == 1
    assert metrics["training/n_rollouts_w_fallback_reward"] == 0


def test_fallback_reward_is_reported_separately() -> None:
    bridge = _bridge("transition")
    rollout = RolloutLegacy(
        rollout_id="r1",
        final_reward=0.0,
        reward_source="fallback",
        reward_reason="no_reward_posted_by_agent",
        triplets=[Triplet(prompt={"token_ids": [10]}, response={"token_ids": [11]})],
    )
    _install_rollout(bridge, rollout)

    batch, metrics = bridge.get_train_data_batch(
        max_prompt_length=4,
        max_response_length=3,
        device=__import__("torch").device("cpu"),
        global_steps=0,
    )

    assert metrics["training/reward"] == 0.0
    assert metrics["training/n_rollouts_w_reward"] == 0
    assert metrics["training/n_rollouts_w_any_reward"] == 1
    assert metrics["training/n_rollouts_w_fallback_reward"] == 1
    assert round(float(batch.batch["token_level_scores"].sum().item()), 2) == 0.0


def test_trajectory_merges_prefix_turns_and_masks_observations() -> None:
    bridge = _bridge("trajectory")
    rollout = RolloutLegacy(
        rollout_id="r1",
        final_reward=1.0,
        triplets=[
            Triplet(prompt={"token_ids": [10, 11]}, response={"token_ids": [12, 13]}),
            Triplet(prompt={"token_ids": [10, 11, 12, 13, 20, 21]}, response={"token_ids": [22]}),
        ],
    )
    _install_rollout(bridge, rollout)

    batch, metrics = bridge.get_train_data_batch(
        max_prompt_length=4,
        max_response_length=6,
        device=__import__("torch").device("cpu"),
        global_steps=0,
    )

    assert metrics["training/n_triplets"] == 1
    assert metrics["training/n_triplets_by_turn"] == 2
    assert metrics["response_length/training/avg_by_turn"] == 1.5
    assert metrics["response_length/training/max_by_turn"] == 2
    assert metrics["response_length/training/min_by_turn"] == 1
    assert "training/avg_response_length_by_turn" not in metrics
    assert "training/max_response_length_by_turn" not in metrics
    assert "training/min_response_length_by_turn" not in metrics
    assert batch.batch["prompts"].tolist() == [[0, 0, 10, 11]]
    assert batch.batch["responses"].tolist() == [[12, 13, 20, 21, 22, 0]]
    assert batch.batch["attention_mask"].tolist() == [[0, 0, 1, 1, 1, 1, 1, 1, 1, 0]]
    assert batch.batch["response_mask"].tolist() == [[1, 1, 0, 0, 1, 0]]
    assert batch.non_tensor_batch["data_id_list"].tolist() == ["data-1"]
    assert batch.non_tensor_batch["rollout_id_list"].tolist() == ["r1"]
    assert batch.batch["token_level_scores"].tolist()[0][4] == pytest.approx(1.0)
    assert batch.batch["token_level_scores"].tolist()[0][2] == pytest.approx(0.0)
    assert batch.batch["token_level_scores"].tolist()[0][3] == pytest.approx(0.0)


def test_trajectory_splits_prefix_mismatches() -> None:
    bridge = _bridge("trajectory")
    rollout = RolloutLegacy(
        rollout_id="r1",
        final_reward=0.5,
        triplets=[
            Triplet(prompt={"token_ids": [1]}, response={"token_ids": [2]}),
            Triplet(prompt={"token_ids": [9]}, response={"token_ids": [10]}),
        ],
    )
    _install_rollout(bridge, rollout)

    batch, metrics = bridge.get_train_data_batch(
        max_prompt_length=4,
        max_response_length=3,
        device=__import__("torch").device("cpu"),
        global_steps=0,
    )

    assert metrics["training/n_triplets"] == 2
    assert metrics["training/n_unmerged_rollouts"] == 1
    assert batch.batch["prompts"].tolist() == [[0, 0, 0, 1], [0, 0, 0, 9]]
    assert batch.batch["responses"].tolist() == [[2, 0, 0], [10, 0, 0]]
    assert batch.batch["response_mask"].tolist() == [[1, 0, 0], [1, 0, 0]]


def test_missing_rollout_emits_placeholder_transition() -> None:
    bridge = _bridge("transition", reward_fillna_value=-1.0)
    bridge.is_train = True
    bridge._enqueue_order = ["missing"]
    bridge._total_tasks_queued = 1
    bridge._task_id_to_original_sample = {"missing": {"data_id": "data-missing"}}
    bridge._completed_rollouts = {}

    batch, metrics = bridge.get_train_data_batch(
        max_prompt_length=3,
        max_response_length=2,
        device=__import__("torch").device("cpu"),
        global_steps=0,
    )

    assert "training/n_placeholder_rows" not in metrics
    assert metrics["training/n_triplets"] == 1
    assert batch.batch["prompts"].tolist() == [[0, 0, 0]]
    assert batch.batch["responses"].tolist() == [[99, 0]]
    assert batch.batch["attention_mask"].tolist() == [[0, 0, 0, 1, 0]]
    assert batch.non_tensor_batch["data_id_list"].tolist() == ["data-missing"]
    assert batch.non_tensor_batch["turn_index_list"].tolist() == [-1]
    assert round(float(batch.batch["token_level_scores"].sum().item()), 2) == -1.0


def test_training_metrics_include_gateway_and_llm_aggregates() -> None:
    bridge = _bridge("transition")
    rollout = RolloutLegacy(
        rollout_id="r1",
        final_reward=1.0,
        reward_source="agent",
        reward_reason="correct",
        triplets=[Triplet(prompt={"token_ids": [10, 11]}, response={"token_ids": [12, 13]})],
        events=[
            {
                "event_type": "model_request",
                "rollout_id": "r1",
                "attempt_id": "pod-1",
                "timestamp": 1.0,
                "data": {
                    "model": "qwen/test",
                    "status": "ok",
                    "http_status": 200,
                    "latency_ms": 123.0,
                    "retry_count": 1,
                    "finish_reason": "stop",
                    "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
                    "response": {"choices": [{"finish_reason": "stop", "token_ids": [12, 13]}]},
                    "server": {"model": "qwen/test"},
                },
            },
            {
                "event_type": "reward",
                "rollout_id": "r1",
                "attempt_id": "pod-1",
                "timestamp": 2.0,
                "data": {"value": 1.0, "source": "agent", "reason": "correct", "resolved": True},
            },
            {
                "event_type": "agent_output",
                "rollout_id": "r1",
                "attempt_id": "pod-1",
                "timestamp": 3.0,
                "data": {"returncode": 0},
            },
        ],
    )
    _install_rollout(bridge, rollout)

    _, metrics = bridge.get_train_data_batch(
        max_prompt_length=4,
        max_response_length=3,
        device=__import__("torch").device("cpu"),
        global_steps=7,
    )

    assert metrics["gateway/training/request_count"] == 1
    assert metrics["gateway/training/num_succeeded_rollouts"] == 0
    assert metrics["gateway/training/rollout_completion_rate"] == 0.0
    assert metrics["gateway/training/success_count"] == 1
    assert metrics["gateway/training/error_count"] == 0
    assert metrics["gateway/training/latency_ms_mean"] == 123.0
    assert metrics["gateway/training/latency_ms_p50"] == 123.0
    assert metrics["gateway/training/latency_ms_p95"] == 123.0
    assert metrics["gateway/training/retry_count"] == 1
    assert metrics["gateway/training/http_status/200_count"] == 1
    assert metrics["gateway/training/finish_reason/stop_count"] == 1
    assert metrics["gateway/training/model/qwen_test/request_count"] == 1
    assert metrics["llm/training/prompt_tokens"] == 4
    assert metrics["llm/training/completion_tokens"] == 6
    assert metrics["llm/training/total_tokens"] == 10
    assert metrics["llm/training/tokens_per_request_mean"] == 10.0
    assert not any(key.startswith("training/num_") for key in metrics)
    assert "training/avg_rollout_latency" not in metrics
    assert "training/rollout_completion_rate" not in metrics
    assert not any(key.startswith("training/gateway/") for key in metrics)
    assert not any(key.startswith("training/llm/") for key in metrics)
    assert not any(key.startswith("training/events/") for key in metrics)
    assert not any(key.startswith("training/reward/") for key in metrics)


def test_validation_metrics_include_gateway_and_llm_aggregates() -> None:
    bridge = _bridge("transition")
    bridge.is_train = False
    bridge._enqueue_order = ["r1"]
    bridge._total_tasks_queued = 1
    bridge._task_id_to_original_sample = {"r1": {"data_id": "data-1"}}
    bridge._completed_rollouts = {
        "r1": RolloutLegacy(
            rollout_id="r1",
            final_reward=0.75,
            triplets=[Triplet(prompt={"token_ids": [10, 11]}, response={"token_ids": [12]})],
            events=[
                {
                    "event_type": "model_request",
                    "rollout_id": "r1",
                    "attempt_id": "pod-1",
                    "timestamp": 1.0,
                    "data": {
                        "model": "qwen/test",
                        "latency_ms": 50.0,
                        "http_status": 200,
                        "status": "ok",
                        "retry_count": 0,
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                        "finish_reason": "stop",
                    },
                }
            ],
        )
    }

    metrics = bridge.get_test_metrics()

    assert metrics["gateway/val/request_count"] == 1
    assert metrics["gateway/val/num_succeeded_rollouts"] == 0
    assert metrics["gateway/val/rollout_completion_rate"] == 0.0
    assert metrics["gateway/val/success_count"] == 1
    assert metrics["gateway/val/latency_ms_mean"] == 50.0
    assert metrics["gateway/val/model/qwen_test/request_count"] == 1
    assert metrics["llm/val/prompt_tokens"] == 3
    assert metrics["llm/val/completion_tokens"] == 2
    assert metrics["llm/val/total_tokens"] == 5
    assert not any(key.startswith("val/num_") for key in metrics)
    assert "val/avg_rollout_latency" not in metrics
    assert "val/rollout_completion_rate" not in metrics
    assert not any(key.startswith("val/gateway/") for key in metrics)
    assert not any(key.startswith("val/llm/") for key in metrics)
