"""Tests for API request/response body models."""

from agl_lite.schemas.api import (
    ArchiveBackend,
    ArchiveRequest,
    EnqueueBatchRequest,
    EnqueueRolloutRequest,
    PatchRolloutRequest,
    PostEventRequest,
    RegisterModelRequest,
)
from agl_lite.schemas.rollout import RolloutConfig, RolloutStatus


class TestEnqueueRolloutRequest:
    def test_minimal(self):
        r = EnqueueRolloutRequest(input={"prompt": "hello"})
        assert r.config is None
        assert r.resources_id is None

    def test_with_config(self):
        r = EnqueueRolloutRequest(
            input={"prompt": "hello"},
            config=RolloutConfig(image="agent:v1"),
            resources_id="res-1",
        )
        assert r.config.image == "agent:v1"


class TestEnqueueBatchRequest:
    def test_batch(self):
        b = EnqueueBatchRequest(
            config=RolloutConfig(image="agent:v1"),
            resources_id="res-1",
            rollouts=[
                EnqueueRolloutRequest(input={"prompt": "task 1"}),
                EnqueueRolloutRequest(input={"prompt": "task 2"}),
            ],
        )
        assert len(b.rollouts) == 2
        assert b.config.image == "agent:v1"


class TestPatchRolloutRequest:
    def test_status_only(self):
        p = PatchRolloutRequest(status=RolloutStatus.RUNNING)
        dumped = p.model_dump(exclude_unset=True)
        assert dumped == {"status": RolloutStatus.RUNNING}

    def test_with_optional_fields(self):
        p = PatchRolloutRequest(status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="pod-uid-1")
        dumped = p.model_dump(exclude_unset=True)
        assert dumped == {"status": RolloutStatus.SUCCEEDED, "succeeded_attempt_id": "pod-uid-1"}

    def test_empty_patch(self):
        p = PatchRolloutRequest()
        dumped = p.model_dump(exclude_unset=True)
        assert dumped == {}

    def test_non_status_field_only(self):
        p = PatchRolloutRequest(job_name="agl-rollout-abc")
        dumped = p.model_dump(exclude_unset=True)
        assert dumped == {"job_name": "agl-rollout-abc"}


class TestPostEventRequest:
    def test_reward_event(self):
        e = PostEventRequest(event_type="reward", data={"value": 0.85})
        assert e.event_type == "reward"

    def test_empty_data(self):
        e = PostEventRequest(event_type="custom")
        assert e.data == {}


class TestRegisterModelRequest:
    def test_with_version(self):
        m = RegisterModelRequest(endpoint="http://vllm:8000/v1", version=42)
        assert m.version == 42

    def test_default_version(self):
        m = RegisterModelRequest(endpoint="http://vllm:8000/v1")
        assert m.version == 0


class TestArchiveRequest:
    def test_without_backend(self):
        a = ArchiveRequest(rollout_ids=["r1", "r2"])
        assert a.backend is None

    def test_with_jsonl_backend(self):
        a = ArchiveRequest(
            rollout_ids=["r1"],
            backend=ArchiveBackend(path="/data/archive.jsonl"),
        )
        assert a.backend.type == "jsonl"
        assert a.backend.path == "/data/archive.jsonl"
