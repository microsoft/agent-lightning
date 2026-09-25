# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import torch
from tensordict import TensorDict
from verl import DataProto

from agentlightning.verl import trainer as trainer_module
from agentlightning.verl.trainer import AgentLightningRayPPOTrainer, _filter_invalid_rollout_log_prob_groups


def _strict_correction_batch() -> DataProto:
    batch = DataProto(
        batch=TensorDict(
            {
                "responses": torch.arange(15).reshape(5, 3),
                "rollout_log_probs": torch.tensor(
                    [
                        [-0.1, -0.2, 0.0],
                        [0.0, 0.0, 0.0],
                        [-0.3, -0.4, 0.0],
                        [-0.5, 0.0, 0.0],
                        [-0.6, -0.7, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "rollout_log_probs_valid_mask": torch.tensor([True, False, True, True, True], dtype=torch.bool),
            },
            batch_size=5,
        )
    )
    batch.non_tensor_batch["data_id_list"] = np.array(["bad", "bad", "good", "good", "good"], dtype=object)
    batch.non_tensor_batch["rollout_id_list"] = np.array(["bad-1", "bad-2", "good-1", "good-1", "good-2"], dtype=object)
    batch.non_tensor_batch["rollout_log_probs_invalid_reason_list"] = np.array(
        ["valid", "missing", "valid", "valid", "valid"], dtype=object
    )
    batch.non_tensor_batch["extra"] = np.array(["a", "b", "c", "d", "e"], dtype=object)
    return batch


def _expected_metrics(*, groups: int, rollouts: int, rows: int, missing: int = 0) -> dict[str, int]:
    return {
        "training/rollout_correction/n_groups_dropped_invalid_log_probs": groups,
        "training/rollout_correction/n_rollouts_dropped_invalid_log_probs": rollouts,
        "training/rollout_correction/n_rows_dropped_invalid_log_probs": rows,
        "training/rollout_correction/n_invalid_rows/missing": missing,
        "training/rollout_correction/n_invalid_rows/length_mismatch": 0,
        "training/rollout_correction/n_invalid_rows/non_finite": 0,
    }


def test_filter_invalid_drops_entire_group_and_preserves_survivor_order() -> None:
    batch, metrics = _filter_invalid_rollout_log_prob_groups(_strict_correction_batch())

    assert batch.batch["responses"].tolist() == [[6, 7, 8], [9, 10, 11], [12, 13, 14]]
    assert batch.non_tensor_batch["data_id_list"].tolist() == ["good", "good", "good"]
    assert batch.non_tensor_batch["rollout_id_list"].tolist() == ["good-1", "good-1", "good-2"]
    assert batch.non_tensor_batch["extra"].tolist() == ["c", "d", "e"]
    assert "rollout_log_probs_valid_mask" not in batch.batch
    assert "rollout_log_probs_invalid_reason_list" not in batch.non_tensor_batch
    assert metrics == _expected_metrics(groups=1, rollouts=2, rows=2, missing=1)


def test_filter_invalid_all_valid_has_zero_metrics() -> None:
    source = _strict_correction_batch()
    source.batch["rollout_log_probs_valid_mask"][:] = True
    source.non_tensor_batch["rollout_log_probs_invalid_reason_list"][:] = "valid"

    batch, metrics = _filter_invalid_rollout_log_prob_groups(source)

    assert len(batch) == 5
    assert "rollout_log_probs_valid_mask" not in batch.batch
    assert "rollout_log_probs_invalid_reason_list" not in batch.non_tensor_batch
    assert metrics == _expected_metrics(groups=0, rollouts=0, rows=0)


def test_filter_invalid_all_groups_invalid_returns_empty_batch() -> None:
    source = _strict_correction_batch()
    source.batch["rollout_log_probs_valid_mask"][2] = False
    source.non_tensor_batch["rollout_log_probs_invalid_reason_list"][2] = "length_mismatch"

    batch, metrics = _filter_invalid_rollout_log_prob_groups(source)

    assert len(batch) == 0
    assert metrics == {
        **_expected_metrics(groups=2, rollouts=4, rows=5, missing=1),
        "training/rollout_correction/n_invalid_rows/length_mismatch": 1,
    }


@pytest.mark.parametrize(
    "field",
    [
        "rollout_log_probs",
        "rollout_log_probs_valid_mask",
        "data_id_list",
        "rollout_id_list",
        "rollout_log_probs_invalid_reason_list",
    ],
)
def test_filter_invalid_requires_strict_fields(field: str) -> None:
    source = _strict_correction_batch()
    if field in source.batch:
        source.batch.pop(field)
    else:
        source.non_tensor_batch.pop(field)

    with pytest.raises((KeyError, ValueError), match=field):
        _filter_invalid_rollout_log_prob_groups(source)


def test_filter_invalid_rejects_misaligned_reason_list() -> None:
    source = _strict_correction_batch()
    source.non_tensor_batch["rollout_log_probs_invalid_reason_list"] = np.array(["valid"], dtype=object)

    with pytest.raises(ValueError, match="rollout_log_probs_invalid_reason_list"):
        _filter_invalid_rollout_log_prob_groups(source)


def test_filter_invalid_rejects_unknown_reason() -> None:
    source = _strict_correction_batch()
    source.non_tensor_batch["rollout_log_probs_invalid_reason_list"][1] = "unknown"

    with pytest.raises(ValueError, match="unknown"):
        _filter_invalid_rollout_log_prob_groups(source)


def test_filter_invalid_rejects_non_finite_survivor() -> None:
    source = _strict_correction_batch()
    source.batch["rollout_log_probs"][2, 0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        _filter_invalid_rollout_log_prob_groups(source)


@pytest.mark.parametrize("bypass_mode", [None, False, True])
@pytest.mark.parametrize("is_train", [False, True])
def test_strict_adapter_enabled_only_for_training_correction(
    monkeypatch: pytest.MonkeyPatch, bypass_mode: bool | None, is_train: bool
) -> None:
    trainer = object.__new__(AgentLightningRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {"rollout_correction": None if bypass_mode is None else {"bypass_mode": bypass_mode}},
            "agentlightning": {
                "trace_aggregator": {"level": "transition"},
                "reward_fillna_value": 0,
            },
            "data": {"max_prompt_length": 4, "max_response_length": 3},
            "actor_rollout_ref": {"actor": {"megatron": {"router_replay": {"mode": "disabled"}}}},
        }
    )
    trainer.is_async = False
    trainer.global_steps = 1
    trainer.tokenizer = type("Tokenizer", (), {"pad_token_id": 0})()
    trainer.async_rollout_manager = type("Servers", (), {"server_addresses": ["local"]})()
    monkeypatch.setattr(trainer, "_resume_all_rollout_generation", lambda: None)
    monkeypatch.setattr(trainer, "_abort_all_rollout_requests", lambda: None)
    monkeypatch.setattr(trainer, "_rollout_lifecycle_metrics", lambda _rollouts: {})

    class Manager:
        def delete_model(self) -> None:
            pass

        def register_model(self, _addresses: list[str]) -> None:
            pass

        def enqueue_and_wait_until_completed(self, _data: dict, *, is_train: bool) -> list:
            return []

    monkeypatch.setattr(trainer, "_make_rollout_manager", lambda _cls: Manager())
    captured: dict[str, object] = {}

    class Adapter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def get_train_data_batch(self, _rollouts: list, *, global_steps: int) -> tuple[DataProto, dict]:
            return DataProto(batch=None), {}

        def get_test_metrics(self, _rollouts: list, *, global_steps: int) -> dict:
            return {}

    monkeypatch.setattr(trainer_module, "RolloutAdapter", Adapter)
    trainer._rollout(DataProto(batch=None), is_train=is_train)

    assert captured["require_rollout_log_probs"] is (is_train and bypass_mode is not None)


def _train_step_trainer(monkeypatch: pytest.MonkeyPatch, rollout_batch: DataProto) -> AgentLightningRayPPOTrainer:
    trainer = object.__new__(AgentLightningRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "rollout_correction": {"bypass_mode": False},
                "use_kl_in_reward": False,
                "gamma": 1,
                "lam": 1,
            },
            "actor_rollout_ref": {
                "rollout": {"n": 2, "temperature": 1},
                "actor": {"ppo_mini_batch_size": 1, "policy_loss": {"loss_mode": "vanilla"}},
            },
            "agentlightning": {},
            "trainer": {"balance_batch": True, "critic_warmup": 0},
        }
    )
    trainer.global_steps = 1
    trainer.use_reference_policy = False
    trainer.use_critic = False
    trainer.checkpoint_manager = type(
        "Checkpoint", (), {"sleep_replicas": lambda self: None, "update_weights": lambda self, step: None}
    )()
    initial = {"responses": torch.zeros(1, 3, dtype=torch.long)}
    monkeypatch.setattr(trainer, "_next_train_batch_dict_for_rollout", lambda: initial)
    monkeypatch.setattr(trainer, "_get_gen_batch", lambda batch: batch)
    monkeypatch.setattr(trainer, "_rollout", lambda _batch, *, is_train: (rollout_batch, {}))
    return trainer


def test_train_step_filters_before_drop_sizing_and_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _strict_correction_batch()
    batch.batch["response_mask"] = torch.ones(5, 3)
    batch.batch["attention_mask"] = torch.ones(5, 3)
    batch.batch["token_level_scores"] = torch.ones(5, 3)
    batch.batch["is_drop_mask"] = torch.tensor([False, False, False, True, False])
    trainer = _train_step_trainer(monkeypatch, batch)
    events: list[str] = []

    real_filter = trainer_module._filter_invalid_rollout_log_prob_groups

    def record_filter(value: DataProto) -> tuple[DataProto, dict[str, int]]:
        events.append("filter")
        return real_filter(value)

    monkeypatch.setattr(trainer_module, "_filter_invalid_rollout_log_prob_groups", record_filter)
    real_getitem = DataProto.__getitem__

    def record_slice(value: DataProto, key: object) -> DataProto:
        if "is_drop_mask" in value.batch and "rollout_log_probs_valid_mask" not in value.batch:
            events.append("marked_drop")
        return real_getitem(value, key)  # pyright: ignore[reportReturnType]

    monkeypatch.setattr(DataProto, "__getitem__", record_slice)

    class ActorConfig:
        def __init__(self) -> None:
            self.policy_loss = {"loss_mode": "vanilla"}

        @property
        def ppo_mini_batch_size(self) -> int:
            events.append("sizing")
            return 1

    config = trainer.config
    trainer.config = SimpleNamespace(
        algorithm=config.algorithm,
        actor_rollout_ref=SimpleNamespace(rollout=config.actor_rollout_ref.rollout, actor=ActorConfig()),
        agentlightning=config.agentlightning,
        trainer=config.trainer,
    )
    monkeypatch.setattr(trainer, "_balance_batch", lambda _batch, *, metrics: events.append("balance"))

    def old_log_prob(value: DataProto) -> tuple[DataProto, int]:
        events.append("old_log_prob")
        assert value.non_tensor_batch["data_id_list"].tolist() == ["good", "good"]
        return DataProto(batch=TensorDict({"old_log_probs": torch.zeros(len(value), 3)}, batch_size=len(value))), 0

    monkeypatch.setattr(trainer, "_compute_old_log_prob", old_log_prob)

    def correction(value: DataProto, _config: object) -> tuple[DataProto, dict]:
        events.append("correction")
        assert value.non_tensor_batch["data_id_list"].tolist() == ["good", "good"]
        assert "rollout_log_probs_valid_mask" not in value.batch
        assert "rollout_log_probs_invalid_reason_list" not in value.non_tensor_batch
        return value, {}

    monkeypatch.setattr("verl.trainer.ppo.rollout_corr_helper.compute_rollout_correction_and_add_to_batch", correction)
    monkeypatch.setattr(
        trainer_module, "compute_advantage", lambda value, **kwargs: (events.append("advantage"), value)[1]
    )
    monkeypatch.setattr(trainer_module, "compute_data_metrics", lambda **kwargs: {})
    monkeypatch.setattr(trainer, "_update_actor", lambda _batch: type("Output", (), {"meta_info": {"metrics": {}}})())

    metrics, trained_batch = trainer._train_step({}, False)

    assert trained_batch is not None
    assert events == ["filter", "marked_drop", "sizing", "balance", "old_log_prob", "correction", "advantage"]
    assert metrics["training/n_sample_collected"] == 5
    assert metrics["training/n_sample_trained"] == 2


def test_train_step_all_invalid_skips_worker_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _strict_correction_batch()
    batch.batch["rollout_log_probs_valid_mask"][:] = False
    batch.non_tensor_batch["rollout_log_probs_invalid_reason_list"][:] = "missing"
    batch.batch["response_mask"] = torch.ones(5, 3)
    batch.batch["attention_mask"] = torch.ones(5, 3)
    batch.batch["token_level_scores"] = torch.ones(5, 3)
    trainer = _train_step_trainer(monkeypatch, batch)

    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail("worker computation ran after all correction groups were rejected")

    monkeypatch.setattr(trainer.checkpoint_manager, "sleep_replicas", fail)
    monkeypatch.setattr(trainer.checkpoint_manager, "update_weights", fail)
    for name in ("_compute_old_log_prob", "_compute_values", "_update_actor", "_update_critic"):
        monkeypatch.setattr(trainer, name, fail)

    metrics, trained_batch = trainer._train_step({}, False)

    assert trained_batch is None
    assert metrics["training/n_sample_collected"] == 5
    assert metrics["training/n_sample_trained"] == 0
    assert metrics["training/rollout_correction/n_groups_dropped_invalid_log_probs"] == 2


def test_fit_logs_skipped_step_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = object.__new__(AgentLightningRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "skipped-step",
                "logger": [],
                "val_before_train": False,
                "test_freq": 0,
                "save_freq": 0,
                "total_epochs": 1,
            },
            "global_profiler": {"steps": None},
        }
    )
    trainer.total_training_steps = 1
    trainer.checkpoint_manager = type("Checkpoint", (), {"update_weights": lambda self, _step: None})()
    monkeypatch.setattr(trainer, "_load_checkpoint", lambda: None)

    metrics = {
        "training/rollout_correction/n_groups_dropped_invalid_log_probs": 2,
        "training/rollout_correction/n_rows_dropped_invalid_log_probs": 4,
    }
    monkeypatch.setattr(trainer, "_train_step", lambda _timing, _profile: (metrics, None))

    log_calls: list[tuple[dict[str, int], int]] = []

    class Logger:
        def log(self, *, data: dict[str, int], step: int) -> None:
            log_calls.append((data, step))

    monkeypatch.setattr(trainer_module, "Tracking", lambda **_kwargs: Logger())

    progress: dict[str, int | bool] = {"updates": 0, "closed": False}

    class Progress:
        def update(self, count: int) -> None:
            progress["updates"] = int(progress["updates"]) + count

        def close(self) -> None:
            progress["closed"] = True

    monkeypatch.setattr(trainer_module, "tqdm", lambda **_kwargs: Progress())

    def fail(*_args: object, **_kwargs: object) -> None:
        pytest.fail("skipped step reached successful-step work")

    for name in ("_validate", "_save_checkpoint"):
        monkeypatch.setattr(trainer, name, fail)
    monkeypatch.setattr(trainer_module, "compute_timing_metrics", fail)
    monkeypatch.setattr(trainer_module, "compute_throughout_metrics", fail)

    trainer.fit()

    assert log_calls == [(metrics, 1)]
    assert progress == {"updates": 1, "closed": True}
    assert trainer.global_steps == 2
