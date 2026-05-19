"""Focused tests for async-rollout trainer glue."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from omegaconf import OmegaConf
from verl import DataProto

from agl_lite.verl.trainer import AglLiteRayPPOTrainer


class _FakeBridge:
    def __init__(self) -> None:
        self.run_kwargs: dict[str, Any] | None = None

    def async_set_up_data_and_server(self, **kwargs: Any) -> None:
        pass

    def run_until_groups_finished(self, **kwargs: Any):
        self.run_kwargs = kwargs
        return {"r1"}, set(), {"training/async/groups_finished_reached": 1}

    def commit_async_step_selection(self, **kwargs: Any) -> dict[str, Any]:
        return {"training/async/n_carry_over_out": 0}

    def async_get_train_data_batch(self, **kwargs: Any):
        batch = DataProto.from_single_dict({"input_ids": torch.zeros((1, 1), dtype=torch.long)})
        batch.batch["attention_mask"] = torch.ones((1, 1), dtype=torch.long)
        batch.batch["responses"] = torch.zeros((1, 1), dtype=torch.long)
        batch.batch["token_level_scores"] = torch.zeros((1, 1), dtype=torch.float32)
        batch.non_tensor_batch["data_id_list"] = ["d1"]
        return batch, {"training/n_placeholder_rows": 0}

    def async_cleanup_consumed(self, **kwargs: Any) -> None:
        pass


def test_async_rollout_waits_for_train_batch_size_groups_not_active_pool() -> None:
    trainer = object.__new__(AglLiteRayPPOTrainer)
    bridge = _FakeBridge()
    trainer._rollout_bridge = cast(Any, bridge)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["http://vllm:8000/v1"])
    trainer.global_steps = 3
    trainer.config = OmegaConf.create(
        {
            "agentlightning": {
                "poll_timeout_seconds": None,
                "trace_aggregator": {"level": "transition"},
            },
            "data": {
                "train_batch_size": 2,
                "max_prompt_length": 128,
                "max_response_length": 64,
            },
            "actor_rollout_ref": {
                "rollout": {"n": 4},
                "actor": {"ppo_mini_batch_size": 2},
            },
        }
    )

    trainer._async_rollout(
        new_samples_dict={"prompt": ["a", "b", "c", "d", "e"]},
        async_train_batch_size=5,
        admin_base_url="http://agl",
        admin_key="admin-key",
        gateway_retry_after_seconds=9,
        gateway_drain_timeout_seconds=1.5,
        rollout_n=4,
    )

    assert bridge.run_kwargs is not None
    assert bridge.run_kwargs["target_groups"] == 2
    assert bridge.run_kwargs["rollout_n"] == 4
    assert bridge.run_kwargs["retry_after_seconds"] == 9
    assert bridge.run_kwargs["drain_timeout"] == 1.5
