"""Tests for math-poc hooks (mock and vllm)."""

from __future__ import annotations

import json

import pytest

from agl_lite.schemas.api import EnqueueRolloutRequest, PatchRolloutRequest
from agl_lite.schemas.rollout import RolloutStatus
from agl_lite.store.memory import InMemoryStore


class TestMathMockHooks:
    @pytest.fixture
    def hooks(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("mock_hooks", "examples/math-poc/mock/hooks.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MathMockHooks()

    def test_on_enqueue_even_correct(self, hooks) -> None:
        req = EnqueueRolloutRequest(
            input={"question": "What is 2+2?", "answer": "4"},
            metadata={"sample_idx_in_batch": 0},
        )
        result = hooks.on_enqueue(req)
        task_input = json.loads(result.config.environment_variables["AGL_TASK_INPUT"])
        assert "\\boxed{4}" in task_input
        assert result.config.image == "math-agent:dev"

    def test_on_enqueue_odd_wrong(self, hooks) -> None:
        req = EnqueueRolloutRequest(
            input={"question": "What is 2+2?", "answer": "4"},
            metadata={"sample_idx_in_batch": 1},
        )
        result = hooks.on_enqueue(req)
        task_input = json.loads(result.config.environment_variables["AGL_TASK_INPUT"])
        assert "\\boxed{WRONG}" in task_input

    def test_on_succeeded_correct(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input={"question": "What is 2+2?", "answer": "4"},
                metadata={"sample_idx_in_batch": 0},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "4"})
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="a1"
        ))
        events = store.query_events(rid)
        rewards = [e for e in events if e.event_type == "reward"]
        assert len(rewards) == 1
        assert rewards[0].data["value"] == 1.0

    def test_on_succeeded_wrong(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input={"question": "What is 2+2?", "answer": "4"},
                metadata={"sample_idx_in_batch": 0},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "5"})
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="a1"
        ))
        events = store.query_events(rid)
        rewards = [e for e in events if e.event_type == "reward"]
        assert len(rewards) == 1
        assert rewards[0].data["value"] == 0.0


class TestMathVllmHooks:
    @pytest.fixture
    def hooks(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("vllm_hooks", "examples/math-poc/vllm/hooks.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.MathVllmHooks()

    def test_on_enqueue_plain_question(self, hooks) -> None:
        req = EnqueueRolloutRequest(
            input={"question": "What is 2+2?", "answer": "4"},
            metadata={"sample_idx_in_batch": 0},
        )
        result = hooks.on_enqueue(req)
        task_input = json.loads(result.config.environment_variables["AGL_TASK_INPUT"])
        assert task_input == "What is 2+2?"  # plain, no boxed
        assert "boxed" not in task_input

    def test_on_succeeded_numeric_correct(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input={"question": "What is 2+2?", "answer": "4"},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "4.0"})
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="a1"
        ))
        events = store.query_events(rid)
        rewards = [e for e in events if e.event_type == "reward"]
        assert len(rewards) == 1
        assert rewards[0].data["value"] == 1.0
        assert rewards[0].data["reason"] == "correct"

    def test_on_succeeded_numeric_wrong(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input={"question": "What is 2+2?", "answer": "4"},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "5"})
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="a1"
        ))
        events = store.query_events(rid)
        rewards = [e for e in events if e.event_type == "reward"]
        assert rewards[0].data["value"] == 0.0
        assert "wrong" in rewards[0].data["reason"]
