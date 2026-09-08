# Copyright (c) Microsoft. All rights reserved.

"""Explicit ESI-Bench split loading without exposing labels to the planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, cast


def normalize_question_id(value: object) -> str:
    """Normalize HF and runner question identifiers to four digit strings."""

    text = str(value).strip()
    if text.lower().startswith("q_"):
        text = text[2:]
    if not text.isdigit():
        raise ValueError(f"Invalid ESI-Bench question id: {value!r}")
    return text.zfill(4)


def read_split(path: Path) -> list[str]:
    """Read one ordered, duplicate-free question-id manifest."""

    if not path.is_file():
        raise FileNotFoundError(f"ESI-Bench split does not exist: {path}")
    ids = [normalize_question_id(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"ESI-Bench split is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"ESI-Bench split contains duplicate ids: {path}")
    return ids


def load_question_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Index the official Hugging Face JSONL export by normalized id."""

    if not path.is_file():
        raise FileNotFoundError(f"Official ESI-Bench questions.jsonl not found: {path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            row = cast(dict[str, Any], value)
            question_id = normalize_question_id(row.get("id"))
            if question_id in rows:
                raise ValueError(f"Duplicate ESI-Bench question id {question_id} in {path}")
            rows[question_id] = row
    return rows


def load_question_row(path: Path, question_id: object) -> dict[str, Any]:
    """Load one scorer-side HF row without placing labels in an AGL task."""

    normalized = normalize_question_id(question_id)
    if not path.is_file():
        raise FileNotFoundError(f"Official ESI-Bench questions.jsonl not found: {path}")
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: object = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            row = cast(dict[str, Any], value)
            if normalize_question_id(row.get("id")) == normalized:
                return row
    raise KeyError(f"Question {normalized} is absent from the official ESI-Bench JSONL export.")


def index_canonical_questions(json_root: Path) -> dict[str, Path]:
    """Index the official runner JSON files while ignoring aggregate manifests."""

    resolved_root = json_root.expanduser().resolve()
    if not resolved_root.is_dir():
        raise FileNotFoundError(f"Official ESI-Bench dataset/json_clean directory not found: {resolved_root}")
    paths: dict[str, Path] = {}
    for path in sorted(resolved_root.rglob("*.json")):
        value: object = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or "id" not in value:
            continue
        row = cast(dict[str, Any], value)
        question_id = normalize_question_id(row["id"])
        previous = paths.get(question_id)
        if previous is not None:
            raise ValueError(f"Duplicate canonical ESI-Bench question {question_id}: {previous} and {path}")
        if not str(row.get("runner_task", "")).strip():
            raise ValueError(f"Canonical ESI-Bench question {question_id} has no runner_task: {path}")
        paths[question_id] = path.resolve()
    if not paths:
        raise ValueError(f"No canonical ESI-Bench question files found below {resolved_root}")
    return paths


def resolve_canonical_question(json_root: Path, relative_path: object, question_id: object) -> Path:
    """Resolve a task's canonical runner JSON without allowing path traversal."""

    root = json_root.expanduser().resolve()
    normalized = normalize_question_id(question_id)
    if not isinstance(relative_path, str) or not relative_path.strip():
        path = index_canonical_questions(root).get(normalized)
        if path is None:
            raise KeyError(f"Question {normalized} is absent from canonical ESI-Bench JSON files.")
    else:
        path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Canonical question path escapes dataset/json_clean: {path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Canonical ESI-Bench question does not exist: {path}")
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or normalize_question_id(cast(dict[str, Any], value).get("id")) != normalized:
        raise ValueError(f"Canonical ESI-Bench question id mismatch for {path}; expected {normalized}.")
    return path


def materialize_split(
    rows: Mapping[str, dict[str, Any]],
    question_ids: Iterable[str],
    *,
    max_steps: int,
    canonical_questions: Mapping[str, Path] | None = None,
    canonical_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Build tasks consumed by the official runner.

    The AGL task contains identifiers and runner configuration only. The worker
    loads the scorer-owned HF row by ID after it has crossed into the isolated
    simulator process. Labels never enter planner or harness inputs; after the
    episode finishes, the worker may attach them to post-hoc diagnostic
    metadata for the Judger described by the SHAPER protocol.
    """

    tasks: list[dict[str, Any]] = []
    for raw_id in question_ids:
        question_id = normalize_question_id(raw_id)
        row = rows.get(question_id)
        if row is None:
            raise KeyError(f"Question {question_id} is absent from the official ESI-Bench JSONL export.")
        runner_task = str(row.get("runner_task", "")).strip()
        if not runner_task:
            raise ValueError(f"Question {question_id} has no runner_task.")
        task: dict[str, Any] = {
            "task_id": f"esi/{question_id}",
            "question_id": question_id,
            "runner_task": runner_task,
            "max_steps": max_steps,
        }
        if canonical_questions is not None:
            if canonical_root is None:
                raise ValueError("canonical_root is required with canonical_questions.")
            canonical_path = canonical_questions.get(question_id)
            if canonical_path is None:
                raise KeyError(f"Question {question_id} is absent from canonical ESI-Bench JSON files.")
            canonical_value: object = json.loads(canonical_path.read_text(encoding="utf-8"))
            if not isinstance(canonical_value, dict):
                raise ValueError(f"Canonical ESI-Bench question must be an object: {canonical_path}")
            canonical_row = cast(dict[str, Any], canonical_value)
            if str(canonical_row.get("runner_task", "")).strip() != runner_task:
                raise ValueError(f"Canonical and HF runner_task disagree for ESI-Bench question {question_id}.")
            task["question_relpath"] = canonical_path.resolve().relative_to(canonical_root.resolve()).as_posix()
        tasks.append(task)
    return tasks


def load_datasets(
    questions_jsonl: Path,
    train_split: Path,
    validation_split: Path,
    *,
    max_steps: int = 30,
    canonical_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load explicit disjoint optimization and fixed-validation datasets."""

    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    train_ids = read_split(train_split)
    validation_ids = read_split(validation_split)
    overlap = sorted(set(train_ids) & set(validation_ids))
    if overlap:
        raise ValueError("ESI-Bench train/validation splits overlap: " + ", ".join(overlap))
    rows = load_question_rows(questions_jsonl)
    canonical_questions = index_canonical_questions(canonical_root) if canonical_root is not None else None
    return (
        materialize_split(
            rows,
            train_ids,
            max_steps=max_steps,
            canonical_questions=canonical_questions,
            canonical_root=canonical_root,
        ),
        materialize_split(
            rows,
            validation_ids,
            max_steps=max_steps,
            canonical_questions=canonical_questions,
            canonical_root=canonical_root,
        ),
    )


def task_ids(tasks: Iterable[Mapping[str, Any]]) -> list[str]:
    """Return stable task ids for split and provenance tests."""

    return [str(task["task_id"]) for task in tasks]
