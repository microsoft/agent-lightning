# Copyright (c) Microsoft. All rights reserved.

"""Checkpoint round trips use real loader/file state and stub only GPU workers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from agentlightning.verl.trainer import AgentLightningRayPPOTrainer


class _Inputs(Dataset[int]):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> int:
        if not 0 <= index < 12:
            raise IndexError(index)
        return index


def _collate(rows: list[int]) -> dict[str, Any]:
    return {
        "id": torch.tensor(rows),
        "label": np.array([f"sample-{i}" for i in rows], dtype=object),
        "payload": [{"value": i} for i in rows],
    }


class _Actor:
    def save_checkpoint(self, local_path: str, remote_path: str | None, step: int, **kwargs: Any) -> None:
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "step.txt").write_text(str(step))

    def load_checkpoint(self, path: str, **kwargs: Any) -> None:
        self.loaded_step = int((Path(path) / "step.txt").read_text())


def _trainer(root: Path, *, is_async: bool = True) -> Any:
    # Model/Ray initialization is irrelevant to saving the trainer's input state.
    trainer: Any = object.__new__(AgentLightningRayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(root),
                "default_hdfs_dir": None,
                "resume_mode": "auto",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
        }
    )
    trainer.is_async = is_async
    trainer.global_steps = 1
    trainer.epoch = 1
    trainer.train_dataloader = StatefulDataLoader(_Inputs(), batch_size=4, shuffle=False, collate_fn=_collate)
    trainer._train_dataloader_iter = None
    trainer._train_dataloader_buf = None
    trainer._carry_over_rollouts = []
    trainer.use_critic = False
    trainer.actor_rollout_wg = _Actor()
    return trainer


@pytest.mark.parametrize("resume_mode", ["auto", "resume_path"])
@pytest.mark.parametrize("first_size,step", [(2, 1), (4, 1), (6, 3), (4, 3), (10, 1), (14, 1)])
def test_async_checkpoint_preserves_unqueued_inputs(
    tmp_path: Path, first_size: int, step: int, resume_mode: str
) -> None:
    original = _trainer(tmp_path / "original")
    original._next_train_batch_dict_with_size(first_size)
    original.global_steps = step
    original._save_checkpoint()

    # Explicit resume must work even when the new output directory is different.
    root = tmp_path / "original" if resume_mode == "auto" else tmp_path / "new-output"
    resumed = _trainer(root)
    resumed.config.trainer.resume_mode = resume_mode
    if resume_mode == "resume_path":
        resumed.config.trainer.resume_from_path = str(tmp_path / "original" / f"global_step_{step}")
    resumed.global_steps = 0
    resumed._load_checkpoint()
    assert resumed.global_steps == step
    assert resumed.actor_rollout_wg.loaded_step == step
    assert resumed.epoch == original.epoch

    position = first_size
    for size in (2, 3, 4):
        expected_ids = [i % 12 for i in range(position, position + size)]
        uninterrupted = original._next_train_batch_dict_with_size(size)
        restored = resumed._next_train_batch_dict_with_size(size)
        assert uninterrupted["id"].tolist() == expected_ids
        assert restored["id"].tolist() == expected_ids
        assert restored["label"].tolist() == [f"sample-{i}" for i in expected_ids]
        assert restored["payload"] == [{"value": i} for i in expected_ids]
        position += size


@pytest.mark.parametrize("is_async", [False, True])
def test_legacy_checkpoint_still_loads(tmp_path: Path, is_async: bool) -> None:
    original = _trainer(tmp_path, is_async=is_async)
    original._next_train_batch_dict()
    RayPPOTrainer._save_checkpoint(original)

    resumed = _trainer(tmp_path, is_async=is_async)
    resumed._load_checkpoint()
    assert resumed.actor_rollout_wg.loaded_step == 1
    assert resumed._next_train_batch_dict()["id"].tolist() == [4, 5, 6, 7]


def test_resume_disabled_does_not_restore_async_inputs(tmp_path: Path) -> None:
    original = _trainer(tmp_path)
    original._next_train_batch_dict_with_size(2)
    original._save_checkpoint()
    resumed = _trainer(tmp_path)
    resumed.config.trainer.resume_mode = "disable"
    resumed._load_checkpoint()
    assert resumed._next_train_batch_dict_with_size(2)["id"].tolist() == [0, 1]
    assert not hasattr(resumed.actor_rollout_wg, "loaded_step")


def test_fit_keeps_restored_epoch_and_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = _trainer(tmp_path)
    original._next_train_batch_dict_with_size(14)
    original._save_checkpoint()

    resumed = _trainer(tmp_path)
    resumed.config.trainer.update(
        project_name="test", experiment_name="test", logger=[], val_before_train=False, total_epochs=3
    )
    resumed.config.global_profiler = {"steps": None}
    resumed.total_training_steps = 5
    resumed.checkpoint_manager = SimpleNamespace(update_weights=lambda step: None)
    observed = {}

    class StopAfterInput(Exception):
        pass

    def train_step(*args: Any) -> None:
        observed["epoch"] = resumed.epoch
        observed["ids"] = resumed._next_train_batch_dict_with_size(2)["id"].tolist()
        raise StopAfterInput

    monkeypatch.setattr(resumed, "_train_step", train_step)
    with pytest.raises(StopAfterInput):
        resumed.fit()
    assert observed == {"epoch": 2, "ids": [2, 3]}


def test_failed_input_save_does_not_publish_new_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    original = _trainer(tmp_path)
    original._next_train_batch_dict_with_size(2)
    original._save_checkpoint()
    original._next_train_batch_dict_with_size(2)
    original.global_steps = 2
    save = torch.save

    def fail_input_save(state: Any, path: Any, *args: Any, **kwargs: Any) -> None:
        if Path(path).name == "agentlightning_input.pt.tmp":
            raise OSError("disk full")
        save(state, path, *args, **kwargs)

    monkeypatch.setattr(torch, "save", fail_input_save)
    with pytest.raises(OSError, match="disk full"):
        original._save_checkpoint()

    assert (tmp_path / "latest_checkpointed_iteration.txt").read_text() == "1"
    assert not (tmp_path / "global_step_2" / "actor").exists()
    resumed = _trainer(tmp_path)
    resumed._load_checkpoint()
    assert resumed._next_train_batch_dict_with_size(2)["id"].tolist() == [2, 3]


def test_sync_checkpoint_retains_verl_format(tmp_path: Path) -> None:
    original = _trainer(tmp_path, is_async=False)
    original._next_train_batch_dict()
    original._save_checkpoint()
    assert not (tmp_path / "global_step_1" / "agentlightning_input.pt").exists()
    resumed = _trainer(tmp_path, is_async=False)
    resumed._load_checkpoint()
    assert resumed._next_train_batch_dict()["id"].tolist() == [4, 5, 6, 7]


def test_auto_resume_without_checkpoint_starts_at_first_input(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path)
    trainer.global_steps = 0
    trainer._load_checkpoint()
    assert trainer._next_train_batch_dict_with_size(2)["id"].tolist() == [0, 1]


def test_unknown_input_checkpoint_version_is_rejected(tmp_path: Path) -> None:
    original = _trainer(tmp_path)
    original._next_train_batch_dict_with_size(2)
    original._save_checkpoint()
    path = tmp_path / "global_step_1" / "agentlightning_input.pt"
    torch.save({"version": 999}, path)
    with pytest.raises(ValueError, match="input checkpoint version"):
        _trainer(tmp_path)._load_checkpoint()


def test_shuffled_async_input_order_survives_restart(tmp_path: Path) -> None:
    original = _trainer(tmp_path)
    resumed = _trainer(tmp_path)
    for trainer, seed in ((original, 7), (resumed, 99)):
        dataset = _Inputs()
        sampler = create_rl_sampler(OmegaConf.create({"shuffle": True, "seed": seed}), dataset)
        trainer.train_dataloader = StatefulDataLoader(dataset, batch_size=4, sampler=sampler, collate_fn=_collate)
    original._next_train_batch_dict_with_size(6)
    original._save_checkpoint()
    resumed._load_checkpoint()
    for size in (2, 5, 3):
        expected = original._next_train_batch_dict_with_size(size)
        actual = resumed._next_train_batch_dict_with_size(size)
        torch.testing.assert_close(actual["id"], expected["id"])
        assert actual["label"].tolist() == expected["label"].tolist()
        assert actual["payload"] == expected["payload"]
