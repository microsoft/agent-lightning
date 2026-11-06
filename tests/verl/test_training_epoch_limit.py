# Copyright (c) Microsoft. All rights reserved.

"""Exercise epoch termination through fit without launching GPU workers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from verl import DataProto

from agentlightning.verl import trainer as trainer_module


def _trainer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: int = 4,
    epochs: int = 1,
    async_size: int | None = None,
    total_steps: int = 100,
    frequency: int = 10,
    skip_updates: bool = False,
) -> Any:
    trainer = cast(Any, object.__new__(trainer_module.AgentLightningRayPPOTrainer))
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "epochs",
                "logger": [],
                "val_before_train": False,
                "total_epochs": epochs,
                "test_freq": frequency,
                "save_freq": frequency,
            },
            "global_profiler": {"steps": None},
            "actor_rollout_ref": {"actor": {"policy_loss": {"loss_mode": "score_centering"}}, "rollout": {"n": 1}},
            "agentlightning": {
                "async_rollout": {"async_train_batch_size": async_size},
                "trace_aggregator": {},
                "reward_fillna_value": 0.0,
            },
            "data": {"train_batch_size": 2, "max_prompt_length": 1, "max_response_length": 1},
        }
    )
    trainer.is_async = async_size is not None
    trainer.total_training_steps = total_steps
    trainer.train_dataloader = [
        {"item": np.arange(start, min(start + 2, rows)), "fake_ids": torch.ones(min(2, rows - start), 1)}
        for start in range(0, rows, 2)
    ]
    trainer._load_checkpoint = lambda: None
    trainer.checkpoint_manager = SimpleNamespace(update_weights=lambda _step: None)
    trainer.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 0)
    trainer.tokenizer = SimpleNamespace(pad_token_id=0)
    trainer.async_rollout_manager = SimpleNamespace(server_addresses=["test"])
    trainer._resume_all_rollout_generation = lambda: None
    trainer._resume_gateway = lambda: None
    trainer._pause_and_drain_gateway = lambda **_kwargs: {}
    trainer._compute_async_rollout_metrics = lambda **_kwargs: {}
    trainer._rollout_lifecycle_metrics = lambda _rollouts: {}
    trainer.fetched = []
    trainer.completed = []
    trainer.updates = []
    trainer.validations = []
    trainer.checkpoints = []
    trainer.targets = []
    trainer._save_checkpoint = lambda: trainer.checkpoints.append(trainer.global_steps)

    def validate():
        trainer.validations.append(trainer.global_steps)
        return {"validation/reward": 1.0}

    trainer._validate = validate
    monkeypatch.setattr(trainer_module, "Tracking", lambda **_kwargs: SimpleNamespace(log=lambda **_kwargs: None))
    monkeypatch.setattr(trainer_module, "compute_timing_metrics", lambda **_kwargs: {})

    class Manager:
        next_id = 0

        def delete_model(self):
            pass

        def register_model(self, _addresses):
            pass

        def enqueue_and_wait_until_group_completed(self, data, carry, *, is_train, target_finished_group_num):
            assert is_train
            active = list(carry)
            for item in data.get("item", []):
                active.append(SimpleNamespace(data_id=str(self.next_id), item=int(item)))
                self.next_id += 1
            # A target above the available groups would hang the real manager.
            assert 0 < target_finished_group_num <= len(active)
            trainer.targets.append(target_finished_group_num)
            return active[:target_finished_group_num], active[target_finished_group_num:]

    class Adapter:
        def __init__(self, **_kwargs):
            pass

        def get_train_data_batch(self, completed, **_kwargs):
            trainer.completed.extend(row.item for row in completed)
            return DataProto.from_dict(tensors={"fake_ids": torch.ones(len(completed), 1)}), {}

    manager = Manager()
    trainer._make_rollout_manager = lambda _cls: manager
    monkeypatch.setattr(trainer_module, "RolloutAdapter", Adapter)

    def train_step(_timing, _profile):
        batch_dict = trainer._next_train_batch_dict_for_rollout()
        trainer.fetched.extend(batch_dict.get("item", []))
        batch = DataProto.from_single_dict(batch_dict)
        if trainer.is_async:
            batch, _ = trainer._rollout(batch, is_train=True)
        if skip_updates:
            return None
        trainer.updates.append(trainer.global_steps)
        return {}, batch

    trainer._train_step = train_step
    return trainer


@pytest.mark.parametrize("epochs", [1, 2])
def test_fit_stops_at_epoch_limit_before_an_extra_update(monkeypatch, epochs):
    trainer = _trainer(monkeypatch, epochs=epochs)

    trainer.fit()

    assert trainer.fetched == [0, 1, 2, 3] * epochs
    assert trainer.updates == list(range(1, 2 * epochs + 1))
    assert trainer.validations == [2 * epochs]
    assert trainer.checkpoints == [2 * epochs]


@pytest.mark.parametrize("rows,async_size", [(4, 4), (6, 5), (3, 5)])
@pytest.mark.parametrize("epochs", [1, 2])
def test_async_fit_drains_admitted_groups_without_refilling_past_epoch_limit(monkeypatch, rows, async_size, epochs):
    trainer = _trainer(monkeypatch, rows=rows, epochs=epochs, async_size=async_size)

    trainer.fit()

    assert trainer.fetched == list(range(rows)) * epochs
    assert trainer.completed == trainer.fetched
    assert trainer._carry_over_rollouts == []
    assert sum(trainer.targets) == rows * epochs
    assert len(trainer.updates) == (rows * epochs + 1) // 2
    assert trainer.validations == [trainer.updates[-1]]
    assert trainer.checkpoints == [trainer.updates[-1]]


def test_epoch_completion_does_not_repeat_scheduled_validation_or_checkpoint(monkeypatch):
    trainer = _trainer(monkeypatch, frequency=2)

    trainer.fit()

    assert trainer.validations == [2]
    assert trainer.checkpoints == [2]


def test_explicit_step_limit_still_stops_training_first(monkeypatch):
    trainer = _trainer(monkeypatch, epochs=2, total_steps=1)

    trainer.fit()

    assert trainer.fetched == [0, 1]
    assert trainer.updates == trainer.validations == trainer.checkpoints == [1]


def test_epoch_limit_stops_empty_collections_without_advancing_optimizer_steps(monkeypatch):
    trainer = _trainer(monkeypatch, skip_updates=True)

    trainer.fit()

    assert trainer.fetched == [0, 1, 2, 3]
    assert trainer.updates == trainer.validations == trainer.checkpoints == []


def test_actual_train_step_accepts_an_empty_refill_for_carry_over(monkeypatch):
    trainer = _trainer(monkeypatch, async_size=4)
    trainer.epoch = 2
    trainer.global_steps = 2
    trainer._train_dataloader_iter = None
    trainer._carry_over_rollouts = [SimpleNamespace(data_id="pending")]

    class ReachedRollout(Exception):
        pass

    def rollout(gen_batch, *, is_train):
        assert is_train
        assert gen_batch.non_tensor_batch == {}
        assert len(gen_batch) == 0
        assert gen_batch.meta_info["global_steps"] == 2
        raise ReachedRollout

    trainer._rollout = rollout
    # Exercise DataProto construction and the inherited _get_gen_batch, not the
    # simplified train-step stub used to observe the fit loop above.
    with pytest.raises(ReachedRollout):
        trainer_module.AgentLightningRayPPOTrainer._train_step(trainer, {}, False)
