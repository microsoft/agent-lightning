"""Tests for event schemas."""

from agl_lite.schemas.event import AttemptInfo, Event, ModelRequestData, RewardData


class TestEvent:
    def test_basic_event(self):
        e = Event(
            event_id="e1",
            event_type="model_request",
            rollout_id="r1",
            attempt_id="pod-uid-1",
            timestamp=1000.0,
            data={"model": "gpt-4", "request": {}, "response": {}},
        )
        assert e.event_type == "model_request"
        assert e.rollout_id == "r1"
        assert e.attempt_id == "pod-uid-1"

    def test_user_defined_event_type(self):
        e = Event(
            event_id="e2",
            event_type="tool_result",
            rollout_id="r1",
            attempt_id="pod-uid-1",
            timestamp=1000.0,
            data={"tool_name": "execute_code", "output": "hello\n", "exit_code": 0},
        )
        assert e.event_type == "tool_result"
        assert e.data["exit_code"] == 0


class TestModelRequestData:
    def test_full_model_request(self):
        d = ModelRequestData(
            model="gpt-4",
            model_version=42,
            request={"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7},
            response={"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 5}},
            latency_ms=1234.5,
            status="ok",
        )
        assert d.model_version == 42
        assert d.status == "ok"

    def test_defaults(self):
        d = ModelRequestData(
            model="gpt-4",
            request={},
            response={},
            latency_ms=100.0,
        )
        assert d.model_version is None
        assert d.adjusted_params is None
        assert d.status == "ok"


class TestRewardData:
    def test_basic_reward(self):
        r = RewardData(value=0.85, message="all tests passed")
        assert r.value == 0.85
        assert r.message == "all tests passed"

    def test_reward_without_message(self):
        r = RewardData(value=0.0)
        assert r.message is None


class TestAttemptInfo:
    def test_attempt_info(self):
        a = AttemptInfo(
            attempt_id="pod-uid-1",
            first_seen=1000.0,
            last_seen=1005.0,
            event_count=5,
        )
        assert a.event_count == 5
