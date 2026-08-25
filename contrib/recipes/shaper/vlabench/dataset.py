# Copyright (c) Microsoft. All rights reserved.

"""Deterministic VLABench optimization and validation splits used by SHAPER."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, cast

TRACK_NAME = "track_4_semantic_instruction"

TRAIN_EPISODES: Mapping[str, Sequence[int]] = {
    "select_fruit": (0, 1, 2),
    "select_toy": (0, 1, 2),
    "select_book": (0, 1, 2),
    "add_condiment": (0, 1, 2),
    "select_painting": (0, 1, 2),
}

VALIDATION_EPISODES: Mapping[str, Sequence[int]] = {
    "select_fruit": (4, 5, 6),
    "select_toy": (4, 5, 6),
    "select_book": (4, 5, 6),
    "add_condiment": (4, 5, 6),
    "select_painting": (4, 5, 6),
    "select_poker": (1, 2, 3),
    "select_mahjong": (1, 2, 3),
    "insert_flower": (1, 2, 3),
}


def track_path(vlabench_root: Path, track_name: str = TRACK_NAME) -> Path:
    """Return the official deterministic track JSON path."""

    return vlabench_root / "configs" / "evaluation" / "tracks" / f"{track_name}.json"


def load_track(vlabench_root: Path, track_name: str = TRACK_NAME) -> dict[str, list[dict[str, Any]]]:
    """Load one official VLABench evaluation track."""

    path = track_path(vlabench_root, track_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"VLABench track not found at {path}. VLABENCH_ROOT must point to the inner VLABench package directory."
        )
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"VLABench track must be a JSON object: {path}")
    output: dict[str, list[dict[str, Any]]] = {}
    for task_name, values in cast(dict[str, object], raw).items():
        if not isinstance(values, list):
            raise ValueError(f"Invalid episode list for VLABench task {task_name!r}.")
        episode_values = cast(list[Any], values)
        if not all(isinstance(item, dict) for item in episode_values):
            raise ValueError(f"Invalid episode list for VLABench task {task_name!r}.")
        output[task_name] = cast(list[dict[str, Any]], episode_values)
    return output


def materialize_split(
    track: Mapping[str, Sequence[dict[str, Any]]],
    specification: Mapping[str, Sequence[int]],
    *,
    max_steps: int,
) -> list[dict[str, Any]]:
    """Materialize explicit episode indices as JSON-serializable AGL tasks."""

    tasks: list[dict[str, Any]] = []
    for task_name, indices in specification.items():
        episodes = track.get(task_name)
        if episodes is None:
            raise KeyError(f"Task {task_name!r} is absent from the selected VLABench track.")
        for index in indices:
            if index < 0 or index >= len(episodes):
                raise IndexError(f"Episode {index} is out of range for {task_name!r} ({len(episodes)} available).")
            tasks.append(
                {
                    "task_id": f"{task_name}/ep_{index:03d}",
                    "task_name": task_name,
                    "episode_index": index,
                    "episode_config": episodes[index],
                    "max_steps": max_steps,
                }
            )
    return tasks


def load_reported_protocol_datasets(
    vlabench_root: Path,
    *,
    track_name: str = TRACK_NAME,
    max_steps: int = 400,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the reported 15-episode optimization and 24-episode validation protocol."""

    track = load_track(vlabench_root, track_name)
    return (
        materialize_split(track, TRAIN_EPISODES, max_steps=max_steps),
        materialize_split(track, VALIDATION_EPISODES, max_steps=max_steps),
    )


def task_ids(tasks: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return task IDs for diagnostics and split tests."""

    return [str(task["task_id"]) for task in tasks]
