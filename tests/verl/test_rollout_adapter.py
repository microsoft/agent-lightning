# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import io
import json
import sys
import zipfile
from types import SimpleNamespace
from typing import ClassVar

import pytest

pytest.importorskip("torch")
pytest.importorskip("tensordict")
pytest.importorskip("verl")

import torch

from agl_lite.verl.agl_rollout_manager import CompletedRollout, Triplet
from agl_lite.verl.rollout_adapter import RolloutAdapter


class FakeTokenizer:
    all_special_ids: ClassVar[list[int]] = []

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


class FakeTable:
    def __init__(self, columns: list[str]) -> None:
        self.columns = columns
        self.rows: list[list[object]] = []

    def add_data(self, *values: object) -> None:
        self.rows.append(list(values))


class FakeArtifactFile:
    def __init__(self, artifact: FakeArtifact, name: str, mode: str) -> None:
        self.artifact = artifact
        self.name = name
        self.mode = mode
        self.content: str | bytes = b"" if "b" in mode else ""

    def __enter__(self) -> FakeArtifactFile:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.artifact.files.append({"name": self.name, "content": self.content})

    def write(self, value: str | bytes) -> int:
        if "b" in self.mode:
            assert isinstance(value, bytes)
            assert isinstance(self.content, bytes)
            self.content += value
            return len(value)
        assert isinstance(value, str)
        assert isinstance(self.content, str)
        self.content += value
        return len(value)


class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict[str, object] | None = None) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.files: list[dict[str, str | bytes]] = []

    def new_file(self, name: str, mode: str = "w") -> FakeArtifactFile:
        assert mode in {"w", "wb"}
        return FakeArtifactFile(self, name, mode)


class FakeRun:
    id = "fake-run"

    def __init__(self) -> None:
        self.artifacts: list[FakeArtifact] = []

    def log_artifact(self, artifact: FakeArtifact) -> None:
        self.artifacts.append(artifact)


def _logged_table(logged: list[tuple[dict[str, FakeTable], int]], key: str) -> tuple[FakeTable, int]:
    matches = [(data[key], step) for data, step in logged if key in data]
    assert len(matches) == 1
    return matches[0]


def _zipped_jsonl_records(
    artifact: FakeArtifact,
    *,
    artifact_path: str,
    jsonl_name: str,
) -> list[dict[str, object]]:
    matches = [file for file in artifact.files if file["name"] == artifact_path]
    assert len(matches) == 1
    content = matches[0]["content"]
    assert isinstance(content, bytes)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.namelist() == [jsonl_name]
        jsonl_text = archive.read(jsonl_name).decode("utf-8")
    return [json.loads(line) for line in jsonl_text.splitlines()]


def _install_fake_wandb(monkeypatch: pytest.MonkeyPatch, logged: list[tuple[dict[str, FakeTable], int]]) -> FakeRun:
    run = FakeRun()
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(
            run=run,
            Table=FakeTable,
            Artifact=FakeArtifact,
            log=lambda data, step: logged.append((data, step)),
        ),
    )
    return run


def _adapter() -> RolloutAdapter:
    return RolloutAdapter(
        max_prompt_length=4,
        max_response_length=3,
        device=torch.device("cpu"),
        pad_token_id=0,
        trace_aggregator_level="trajectory",
        tokenizer=FakeTokenizer(),
    )


def _triplet(prompt_ids: list[int], response_ids: list[int]) -> Triplet:
    return Triplet(
        prompt={"token_ids": prompt_ids},
        response={"token_ids": response_ids, "log_probs": [-0.1] * len(response_ids)},
    )


def test_trajectory_prefix_mismatch_uploads_trace_merge_table_to_wandb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[dict[str, FakeTable], int]] = []
    _install_fake_wandb(monkeypatch, logged)

    rollout = CompletedRollout(
        rollout_id="r1",
        data_id="data-1",
        step=17,
        sample_idx_in_step=0,
        enqueue_time=0.0,
        final_reward=0.5,
        triplets=[
            _triplet([1], [2]),
            _triplet([9], [10]),
        ],
    )

    batch, metrics = _adapter().get_train_data_batch([rollout], global_steps=17)

    assert metrics["training/n_unmerged_rollouts"] == 1
    assert metrics["training/n_trace_merge_mismatch_rows"] == 1
    assert batch.batch["prompts"].tolist() == [[0, 0, 0, 1], [0, 0, 0, 9]]
    assert batch.batch["responses"].tolist() == [[2, 0, 0], [10, 0, 0]]
    assert batch.batch["response_mask"].tolist() == [[1, 0, 0], [1, 0, 0]]

    table, step = _logged_table(logged, "training/trace_merge_mismatches")
    assert step == 17
    assert table.columns[0] == "global_steps"
    assert table.columns[-2:] == ["previous_trace", "current_trace"]
    assert table.rows == [
        [
            17,
            "r1",
            "data-1",
            1,
            False,
            False,
            True,
            1,
            1,
            2,
            2,
            "1 2",
            "9 10",
        ]
    ]


def test_trajectory_prefix_mismatch_wandb_table_is_capped_at_100(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[dict[str, FakeTable], int]] = []
    _install_fake_wandb(monkeypatch, logged)
    rollout = CompletedRollout(
        rollout_id="r1",
        data_id="data-1",
        step=9,
        sample_idx_in_step=0,
        enqueue_time=0.0,
        final_reward=0.5,
        triplets=[_triplet([turn * 2 + 1], [turn * 2 + 2]) for turn in range(102)],
    )

    _adapter().get_train_data_batch([rollout], global_steps=9)

    table, _ = _logged_table(logged, "training/trace_merge_mismatches")
    assert len(table.rows) == 100
    assert table.rows[0][3] == 1
    assert table.rows[-1][3] == 100


def test_training_step_uploads_24_compact_rollout_trajectories_to_wandb_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[dict[str, FakeTable], int]] = []
    run = _install_fake_wandb(monkeypatch, logged)
    rollouts = [
        CompletedRollout(
            rollout_id=f"r{index}",
            data_id=f"data-{index}",
            step=23,
            sample_idx_in_step=index,
            enqueue_time=0.0,
            final_reward=float(index),
            triplets=[
                _triplet([index], [index + 100]),
                _triplet([index, index + 100, index + 200], [index + 300]),
            ],
        )
            for index in range(1, 27)
    ]

    _adapter().get_train_data_batch(rollouts, global_steps=23)

    table, step = _logged_table(logged, "training/rollout_trajectories")
    assert step == 23
    assert table.columns == ["global_steps", "trajectory_artifact", "trajectory_artifact_path", "row_count"]
    artifact_name = "rollout-trajectories-fake-run-step-23"
    artifact_path = "step_23/rollout_trajectories.jsonl.zip"
    assert table.rows == [[23, artifact_name, artifact_path, 24]]

    assert len(run.artifacts) == 1
    artifact = run.artifacts[0]
    assert artifact.name == artifact_name
    assert artifact.type == "rollout_trajectories"
    assert artifact.metadata == {"global_steps": 23, "row_count": 24, "format": "jsonl.zip"}
    assert [file["name"] for file in artifact.files] == [artifact_path]

    records = _zipped_jsonl_records(
        artifact,
        artifact_path=artifact_path,
        jsonl_name="rollout_trajectories.jsonl",
    )
    assert len(records) == 24
    assert all(set(record) == {"rollout_id", "reward", "prompt", "response"} for record in records)
    assert [record["rollout_id"] for record in records] == [f"r{index}" for index in range(1, 25)]
    assert records[0] == {"rollout_id": "r1", "reward": 1.0, "prompt": "1 101 201", "response": "301"}
    assert records[-1] == {
        "rollout_id": "r24",
        "reward": 24.0,
        "prompt": "24 124 224",
        "response": "324",
    }


def test_validation_uploads_all_compact_rollout_trajectories_to_wandb_zip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[tuple[dict[str, FakeTable], int]] = []
    run = _install_fake_wandb(monkeypatch, logged)
    rollouts = [
        CompletedRollout(
            rollout_id=f"vr{index}",
            data_id=f"val-data-{index}",
            step=31,
            sample_idx_in_step=index,
            enqueue_time=0.0,
            final_reward=float(index),
            triplets=[
                _triplet([index], [index + 10]),
                _triplet([index, index + 10, index + 20], [index + 30]),
            ],
        )
        for index in range(1, 8)
    ]

    metrics = _adapter().get_test_metrics(rollouts, global_steps=31)

    assert metrics["val/reward"] == 4.0
    assert metrics["val/n_rollouts"] == 7
    assert metrics["val/n_rollouts_w_trace"] == 7
    assert metrics["val/n_rollouts_w_reward"] == 7
    assert metrics["val/mean_response_length_per_turn"] == 1.0
    assert metrics["val/mean_total_response_length_per_rollout"] == 2.0
    assert metrics["val/turn_count"] == 2.0

    table, step = _logged_table(logged, "val/rollout_trajectories")
    assert step == 31
    assert table.columns == ["global_steps", "trajectory_artifact", "trajectory_artifact_path", "row_count"]
    artifact_name = "validation-trajectories-fake-run-step-31"
    artifact_path = "step_31/validation_trajectories.jsonl.zip"
    assert table.rows == [[31, artifact_name, artifact_path, 7]]

    assert len(run.artifacts) == 1
    artifact = run.artifacts[0]
    assert artifact.name == artifact_name
    assert artifact.type == "validation_trajectories"
    assert artifact.metadata == {"global_steps": 31, "row_count": 7, "format": "jsonl.zip"}

    records = _zipped_jsonl_records(
        artifact,
        artifact_path=artifact_path,
        jsonl_name="validation_trajectories.jsonl",
    )
    assert len(records) == 7
    assert all(set(record) == {"rollout_id", "reward", "prompt", "response"} for record in records)
    assert [record["rollout_id"] for record in records] == [f"vr{index}" for index in range(1, 8)]
    assert records[0] == {"rollout_id": "vr1", "reward": 1.0, "prompt": "1 11 21", "response": "31"}
    assert records[-1] == {"rollout_id": "vr7", "reward": 7.0, "prompt": "7 17 27", "response": "37"}
