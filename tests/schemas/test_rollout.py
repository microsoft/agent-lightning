"""Tests for rollout schemas — state transitions, config, validation."""

from agl_lite.schemas.rollout import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    Mount,
    Rollout,
    RolloutConfig,
    RolloutStatus,
)


class TestRolloutStatus:
    def test_terminal_statuses(self):
        assert RolloutStatus.SUCCEEDED in TERMINAL_STATUSES
        assert RolloutStatus.TERMINAL_FAILED in TERMINAL_STATUSES
        assert RolloutStatus.CANCELLED in TERMINAL_STATUSES
        assert RolloutStatus.QUEUING not in TERMINAL_STATUSES
        assert RolloutStatus.RUNNING not in TERMINAL_STATUSES

    def test_valid_transitions_from_queuing(self):
        allowed = VALID_TRANSITIONS[RolloutStatus.QUEUING]
        assert RolloutStatus.RUNNING in allowed
        assert RolloutStatus.TERMINAL_FAILED in allowed
        assert RolloutStatus.CANCELLED in allowed
        assert RolloutStatus.SUCCEEDED not in allowed  # can't skip running

    def test_valid_transitions_from_running(self):
        allowed = VALID_TRANSITIONS[RolloutStatus.RUNNING]
        assert RolloutStatus.SUCCEEDED in allowed
        assert RolloutStatus.TERMINAL_FAILED in allowed
        assert RolloutStatus.CANCELLED in allowed
        assert RolloutStatus.QUEUING not in allowed  # no going backwards

    def test_terminal_states_have_no_transitions(self):
        for status in TERMINAL_STATUSES:
            assert VALID_TRANSITIONS[status] == set()


class TestRolloutConfig:
    def test_minimal_config(self):
        config = RolloutConfig(image="my-agent:latest")
        assert config.image == "my-agent:latest"
        assert config.command == []
        assert config.environment_variables == {}
        assert config.mount == []
        assert config.timeout is None
        assert config.max_retries is None

    def test_full_config(self):
        config = RolloutConfig(
            image="my-agent:v2",
            command=["python", "solve.py"],
            environment_variables={"MODE": "train", "DEBUG": "1"},
            mount=[Mount(name="data", mount_path="/data", source="my-pvc", read_only=True)],
            timeout=600,
            max_retries=3,
        )
        assert config.command == ["python", "solve.py"]
        assert config.environment_variables["MODE"] == "train"
        assert len(config.mount) == 1
        assert config.mount[0].mount_path == "/data"
        assert config.timeout == 600
        assert config.max_retries == 3


class TestRollout:
    def test_defaults(self):
        r = Rollout(
            rollout_id="r1",
            input={"prompt": "hello"},
            config=RolloutConfig(image="agent:v1"),
            created_at=1000.0,
            updated_at=1000.0,
        )
        assert r.status == RolloutStatus.QUEUING
        assert r.cancel_requested is False
        assert r.resources_id is None
        assert r.job_name is None
        assert r.succeeded_attempt_id is None
        assert r.error_message is None
        assert r.version == 1

    def test_with_resources_id(self):
        r = Rollout(
            rollout_id="r1",
            input={"prompt": "hello"},
            config=RolloutConfig(image="agent:v1"),
            resources_id="res-42",
            created_at=1000.0,
            updated_at=1000.0,
        )
        assert r.resources_id == "res-42"
