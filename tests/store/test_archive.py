"""Tests for InMemoryStore — archive and purge."""

import json
from pathlib import Path

import pytest

from agl_lite.schemas.api import ArchiveBackend, EnqueueRolloutRequest, UpdateRolloutRequest
from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.rollout import RolloutConfig, RolloutStatus
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def _enqueue(store: InMemoryStore, **kwargs):
    kwargs.setdefault("input", {"prompt": "test"})
    kwargs.setdefault("config", RolloutConfig(image="agent:v1"))
    return store.enqueue_rollout(EnqueueRolloutRequest(**kwargs))


def _update(store: InMemoryStore, rollout_id: str, status: str, expected_version: int, **kwargs):
    return store.update_rollout(
        rollout_id, UpdateRolloutRequest(status=status, expected_version=expected_version, **kwargs)
    )


def _make_terminal_rollout(
    store: InMemoryStore, status: RolloutStatus = RolloutStatus.SUCCEEDED, **enqueue_kwargs
) -> str:
    """Helper: create a rollout and move it to a terminal state. Returns rollout_id."""
    r = _enqueue(store, **enqueue_kwargs)
    if status == RolloutStatus.SUCCEEDED:
        _update(store, r.rollout_id, "running", 1)
        _update(store, r.rollout_id, "succeeded", 2, succeeded_attempt_id="pod-1")
    else:
        _update(store, r.rollout_id, status.value, 1)
    return r.rollout_id


class TestArchivePurge:
    def test_purge_without_backend(self, store: InMemoryStore):
        rid = _make_terminal_rollout(store)
        store.add_event(rid, "pod-1", "model_request", {"model": "gpt-4"})
        result = store.archive_rollouts([rid])
        assert result.archived == 1
        assert result.purged == 1
        assert result.path is None
        # Verify purged.
        assert not store.rollout_exists(rid)
        with pytest.raises(NotFoundError):
            store.get_rollout(rid)

    def test_reject_non_terminal(self, store: InMemoryStore):
        r = _enqueue(store)
        with pytest.raises(ValueError, match="non-terminal"):
            store.archive_rollouts([r.rollout_id])

    def test_reject_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.archive_rollouts(["nonexistent"])

    def test_write_jsonl(self, store: InMemoryStore, tmp_path: Path):
        rid = _make_terminal_rollout(store)
        store.add_event(rid, "pod-1", "model_request", {"model": "gpt-4"})
        store.add_event(rid, "pod-1", "reward", {"value": 0.85})

        archive_path = tmp_path / "archive.jsonl"
        backend = ArchiveBackend(path=str(archive_path))
        result = store.archive_rollouts([rid], backend=backend)
        assert result.path == str(archive_path)
        assert archive_path.exists()

        # Parse JSONL.
        lines = archive_path.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["rollout"]["rollout_id"] == rid
        assert record["rollout"]["status"] == "succeeded"
        assert len(record["events"]) == 2
        assert record["events"][0]["event_type"] == "model_request"
        assert record["events"][1]["event_type"] == "reward"

    def test_jsonl_includes_resources(self, store: InMemoryStore, tmp_path: Path):
        res = store.add_resources({"system_prompt": "Be helpful"})
        r = _enqueue(store, resources_id=res.resources_id)
        _update(store, r.rollout_id, "running", 1)
        _update(store, r.rollout_id, "succeeded", 2, succeeded_attempt_id="pod-1")

        archive_path = tmp_path / "archive.jsonl"
        store.archive_rollouts([r.rollout_id], backend=ArchiveBackend(path=str(archive_path)))

        record = json.loads(archive_path.read_text().strip())
        assert record["resources"]["resources_id"] == res.resources_id
        assert record["resources"]["resources"]["system_prompt"] == "Be helpful"

    def test_jsonl_append(self, store: InMemoryStore, tmp_path: Path):
        """Multiple archive calls append to the same file."""
        archive_path = tmp_path / "archive.jsonl"
        backend = ArchiveBackend(path=str(archive_path))

        rid1 = _make_terminal_rollout(store)
        store.archive_rollouts([rid1], backend=backend)

        rid2 = _make_terminal_rollout(store)
        store.archive_rollouts([rid2], backend=backend)

        lines = archive_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_multiple_rollouts_in_one_call(self, store: InMemoryStore, tmp_path: Path):
        rid1 = _make_terminal_rollout(store)
        rid2 = _make_terminal_rollout(store, status=RolloutStatus.TERMINAL_FAILED)
        store.add_event(rid1, "pod-1", "model_request", {})
        store.add_event(rid2, "pod-2", "model_request", {})

        archive_path = tmp_path / "archive.jsonl"
        result = store.archive_rollouts([rid1, rid2], backend=ArchiveBackend(path=str(archive_path)))
        assert result.archived == 2
        assert result.purged == 2
        assert not store.rollout_exists(rid1)
        assert not store.rollout_exists(rid2)

        lines = archive_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_multiple_attempts_in_archive(self, store: InMemoryStore, tmp_path: Path):
        """Events from all attempts are included, sorted by timestamp."""
        r = _enqueue(store)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"attempt": 1})
        store.add_event(r.rollout_id, "pod-2", "model_request", {"attempt": 2})
        _update(store, r.rollout_id, "running", 1)
        _update(store, r.rollout_id, "succeeded", 2, succeeded_attempt_id="pod-2")

        archive_path = tmp_path / "archive.jsonl"
        store.archive_rollouts([r.rollout_id], backend=ArchiveBackend(path=str(archive_path)))

        record = json.loads(archive_path.read_text().strip())
        assert len(record["events"]) == 2  # both attempts

    def test_creates_parent_dirs(self, store: InMemoryStore, tmp_path: Path):
        rid = _make_terminal_rollout(store)
        archive_path = tmp_path / "nested" / "dir" / "archive.jsonl"
        store.archive_rollouts([rid], backend=ArchiveBackend(path=str(archive_path)))
        assert archive_path.exists()
