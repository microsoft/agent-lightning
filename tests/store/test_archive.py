"""Tests for InMemoryStore — archive and purge."""

import json
from pathlib import Path

import pytest

from agl_lite.schemas.api import ArchiveBackend
from agl_lite.schemas.errors import NotFoundError
from agl_lite.schemas.rollout import RolloutConfig, RolloutStatus
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def config() -> RolloutConfig:
    return RolloutConfig(image="agent:v1")


def _make_terminal_rollout(
    store: InMemoryStore, config: RolloutConfig, status: RolloutStatus = RolloutStatus.SUCCEEDED
) -> str:
    """Helper: create a rollout and move it to a terminal state. Returns rollout_id."""
    r = store.enqueue_rollout(input={"prompt": "test"}, config=config)
    if status == RolloutStatus.SUCCEEDED:
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        store.update_rollout(r.rollout_id, status, expected_version=2, succeeded_attempt_id="pod-1")
    else:
        store.update_rollout(r.rollout_id, status, expected_version=1)
    return r.rollout_id


class TestArchivePurge:
    def test_purge_without_backend(self, store: InMemoryStore, config: RolloutConfig):
        rid = _make_terminal_rollout(store, config)
        store.add_event(rid, "pod-1", "model_request", {"model": "gpt-4"})
        result = store.archive_rollouts([rid])
        assert result.archived == 1
        assert result.purged == 1
        assert result.path is None
        # Verify purged.
        assert not store.rollout_exists(rid)
        with pytest.raises(NotFoundError):
            store.get_rollout(rid)

    def test_reject_non_terminal(self, store: InMemoryStore, config: RolloutConfig):
        r = store.enqueue_rollout(input={}, config=config)
        with pytest.raises(ValueError, match="non-terminal"):
            store.archive_rollouts([r.rollout_id])

    def test_reject_not_found(self, store: InMemoryStore):
        with pytest.raises(NotFoundError):
            store.archive_rollouts(["nonexistent"])

    def test_write_jsonl(self, store: InMemoryStore, config: RolloutConfig, tmp_path: Path):
        rid = _make_terminal_rollout(store, config)
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

    def test_jsonl_includes_resources(self, store: InMemoryStore, config: RolloutConfig, tmp_path: Path):
        res = store.add_resources({"system_prompt": "Be helpful"})
        r = store.enqueue_rollout(input={}, config=config, resources_id=res.resources_id)
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        store.update_rollout(r.rollout_id, RolloutStatus.SUCCEEDED, expected_version=2, succeeded_attempt_id="pod-1")

        archive_path = tmp_path / "archive.jsonl"
        store.archive_rollouts([r.rollout_id], backend=ArchiveBackend(path=str(archive_path)))

        record = json.loads(archive_path.read_text().strip())
        assert record["resources"]["resources_id"] == res.resources_id
        assert record["resources"]["resources"]["system_prompt"] == "Be helpful"

    def test_jsonl_append(self, store: InMemoryStore, config: RolloutConfig, tmp_path: Path):
        """Multiple archive calls append to the same file."""
        archive_path = tmp_path / "archive.jsonl"
        backend = ArchiveBackend(path=str(archive_path))

        rid1 = _make_terminal_rollout(store, config)
        store.archive_rollouts([rid1], backend=backend)

        rid2 = _make_terminal_rollout(store, config)
        store.archive_rollouts([rid2], backend=backend)

        lines = archive_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_multiple_rollouts_in_one_call(self, store: InMemoryStore, config: RolloutConfig, tmp_path: Path):
        rid1 = _make_terminal_rollout(store, config)
        rid2 = _make_terminal_rollout(store, config, status=RolloutStatus.TERMINAL_FAILED)
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

    def test_multiple_attempts_in_archive(self, store: InMemoryStore, config: RolloutConfig, tmp_path: Path):
        """Events from all attempts are included, sorted by timestamp."""
        r = store.enqueue_rollout(input={}, config=config)
        store.add_event(r.rollout_id, "pod-1", "model_request", {"attempt": 1})
        store.add_event(r.rollout_id, "pod-2", "model_request", {"attempt": 2})
        store.update_rollout(r.rollout_id, RolloutStatus.RUNNING, expected_version=1)
        store.update_rollout(r.rollout_id, RolloutStatus.SUCCEEDED, expected_version=2, succeeded_attempt_id="pod-2")

        archive_path = tmp_path / "archive.jsonl"
        store.archive_rollouts([r.rollout_id], backend=ArchiveBackend(path=str(archive_path)))

        record = json.loads(archive_path.read_text().strip())
        assert len(record["events"]) == 2  # both attempts

    def test_creates_parent_dirs(self, store: InMemoryStore, config: RolloutConfig, tmp_path: Path):
        rid = _make_terminal_rollout(store, config)
        archive_path = tmp_path / "nested" / "dir" / "archive.jsonl"
        store.archive_rollouts([rid], backend=ArchiveBackend(path=str(archive_path)))
        assert archive_path.exists()
