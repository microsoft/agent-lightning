"""Tests for artifact event handling in InMemoryStore."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.store.memory import InMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> InMemoryStore:
    return InMemoryStore(artifact_dir=str(tmp_path / "artifacts"))


@pytest.fixture
def rollout_id(store: InMemoryStore) -> str:
    rollouts = store.enqueue_rollouts([EnqueueRolloutRequest(input={"task": "test"})])
    return rollouts[0].rollout_id


class TestArtifactEvents:
    def test_artifact_written_to_disk(self, store: InMemoryStore, rollout_id: str, tmp_path: Path) -> None:
        """Artifact content is written to disk under artifact_dir/<rollout_id>/."""
        event = store.add_event(rollout_id, "attempt-1", "artifact", {
            "filename": "test_output.txt",
            "content": "PASSED: test_foo\nFAILED: test_bar\n",
        })

        # Event data replaced with reference (no content).
        assert "content" not in event.data
        assert event.data["filename"] == "test_output.txt"
        assert event.data["size"] == 34
        assert "path" in event.data

        # File exists on disk with correct content.
        artifact_path = Path(event.data["path"])
        assert artifact_path.exists()
        assert artifact_path.read_text() == "PASSED: test_foo\nFAILED: test_bar\n"

    def test_artifact_default_filename(self, store: InMemoryStore, rollout_id: str) -> None:
        """Artifact without filename gets default name."""
        event = store.add_event(rollout_id, "attempt-1", "artifact", {
            "content": "some data",
        })
        assert event.data["filename"] == "artifact.bin"

    def test_artifact_in_rollout_subdir(self, store: InMemoryStore, rollout_id: str, tmp_path: Path) -> None:
        """Each rollout gets its own subdirectory."""
        store.add_event(rollout_id, "attempt-1", "artifact", {
            "filename": "log.txt",
            "content": "hello",
        })
        expected_dir = tmp_path / "artifacts" / rollout_id
        assert expected_dir.is_dir()
        assert (expected_dir / "log.txt").exists()

    def test_multiple_artifacts_same_rollout(self, store: InMemoryStore, rollout_id: str) -> None:
        """Multiple artifacts for the same rollout are stored separately."""
        e1 = store.add_event(rollout_id, "attempt-1", "artifact", {
            "filename": "patch.diff",
            "content": "--- a/foo.py\n+++ b/foo.py\n",
        })
        e2 = store.add_event(rollout_id, "attempt-1", "artifact", {
            "filename": "test_output.txt",
            "content": "all tests passed",
        })
        assert Path(e1.data["path"]).exists()
        assert Path(e2.data["path"]).exists()
        assert e1.data["path"] != e2.data["path"]

    def test_non_artifact_event_unchanged(self, store: InMemoryStore, rollout_id: str) -> None:
        """Regular events are not affected by artifact handling."""
        event = store.add_event(rollout_id, "attempt-1", "agent_output", {
            "answer": "42",
            "content": "this field should stay",
        })
        assert event.data["content"] == "this field should stay"
        assert event.data["answer"] == "42"

    def test_artifact_queryable(self, store: InMemoryStore, rollout_id: str) -> None:
        """Artifact events are queryable like any other event."""
        store.add_event(rollout_id, "attempt-1", "artifact", {
            "filename": "log.txt",
            "content": "test data",
        })
        store.add_event(rollout_id, "attempt-1", "agent_output", {"answer": "42"})

        artifacts = store.query_events(rollout_id, attempt_id="attempt-1", event_type="artifact")
        assert len(artifacts) == 1
        assert artifacts[0].data["filename"] == "log.txt"
        assert "content" not in artifacts[0].data

    def test_artifact_large_content(self, store: InMemoryStore, rollout_id: str) -> None:
        """Large artifact content is handled correctly."""
        large_content = "x" * 1_000_000  # 1 MB
        event = store.add_event(rollout_id, "attempt-1", "artifact", {
            "filename": "big_log.txt",
            "content": large_content,
        })
        assert event.data["size"] == 1_000_000
        assert Path(event.data["path"]).read_text() == large_content
