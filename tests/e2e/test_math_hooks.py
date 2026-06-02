"""Tests for math-poc hooks (mock and vllm)."""

from __future__ import annotations

import json

import pytest

from agl_lite.schemas import RolloutCreate, RolloutPatch
from agl_lite.schemas import RolloutState
from agl_lite.server.store import InMemoryStore

# Minimal pod spec matching examples/math-poc/job-template.yaml
_POD_SPEC = {
    "containers": [
        {
            "name": "agent",
            "image": "math-agent:dev",
            "command": ["python", "/app/qa_agent.py"],
        }
    ]
}


def _get_task_input(result) -> str:
    """Extract AGL_TASK_INPUT value from pod spec container env."""
    env = result.config.pod_spec["containers"][0].get("env", [])
    for e in env:
        if e["name"] == "AGL_TASK_INPUT":
            return e["value"]
    raise KeyError("AGL_TASK_INPUT not found in container env")


class TestMathMockHooks:
    @pytest.fixture
    def hooks(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("mock_hooks", "examples/math-poc/mock/hooks.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        h = mod.MathMockHooks()
        h._pod_spec = _POD_SPEC
        return h

    def test_on_enqueue_even_correct(self, hooks) -> None:
        req = RolloutCreate(
            input={"question": "What is 2+2?", "answer": "4"},
            metadata={"sample_idx_in_batch": 0},
        )
        result = hooks.on_enqueue(req)
        task_input = json.loads(_get_task_input(result))
        assert "\\boxed{4}" in task_input
        assert result.input == {"question": "What is 2+2?", "answer": "4"}

    def test_on_enqueue_odd_wrong(self, hooks) -> None:
        req = RolloutCreate(
            input={"question": "What is 2+2?", "answer": "4"},
            metadata={"sample_idx_in_batch": 1},
        )
        result = hooks.on_enqueue(req)
        task_input = json.loads(_get_task_input(result))
        assert "\\boxed{WRONG}" in task_input

    def test_on_succeeded_correct(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            RolloutCreate(
                input={"question": "What is 2+2?", "answer": "4"},
                metadata={"sample_idx_in_batch": 0},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "4"})
        store.update_rollout(rid, RolloutPatch(status=RolloutState.RUNNING))
        store.update_rollout(rid, RolloutPatch(
            status=RolloutState.SUCCEEDED, last_attempt_id="a1"
        ))
        rewards = [e for e in store.query_events(rid) if e.event_type == "reward"]
        assert len(rewards) == 1
        assert rewards[0].data["value"] == 1.0

    def test_on_succeeded_wrong(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            RolloutCreate(
                input={"question": "What is 2+2?", "answer": "4"},
                metadata={"sample_idx_in_batch": 0},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "5"})
        store.update_rollout(rid, RolloutPatch(status=RolloutState.RUNNING))
        store.update_rollout(rid, RolloutPatch(
            status=RolloutState.SUCCEEDED, last_attempt_id="a1"
        ))
        rewards = [e for e in store.query_events(rid) if e.event_type == "reward"]
        assert len(rewards) == 1
        assert rewards[0].data["value"] == 0.0


class TestMathVllmHooks:
    @pytest.fixture
    def hooks(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec = spec_from_file_location("vllm_hooks", "examples/math-poc/vllm/hooks.py")
        mod = module_from_spec(spec)
        spec.loader.exec_module(mod)
        h = mod.MathVllmHooks()
        h._pod_spec = _POD_SPEC
        return h

    def test_on_enqueue_plain_question(self, hooks) -> None:
        req = RolloutCreate(
            input={"question": "What is 2+2?", "answer": "4"},
            metadata={"sample_idx_in_batch": 0},
        )
        result = hooks.on_enqueue(req)
        task_input = json.loads(_get_task_input(result))
        assert task_input == "What is 2+2?"
        assert "boxed" not in task_input
        assert result.input == {"question": "What is 2+2?", "answer": "4"}

    def test_on_succeeded_numeric_correct(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            RolloutCreate(input={"question": "What is 2+2?", "answer": "4"})
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "4.0"})
        store.update_rollout(rid, RolloutPatch(status=RolloutState.RUNNING))
        store.update_rollout(rid, RolloutPatch(
            status=RolloutState.SUCCEEDED, last_attempt_id="a1"
        ))
        rewards = [e for e in store.query_events(rid) if e.event_type == "reward"]
        assert len(rewards) == 1
        assert rewards[0].data["value"] == 1.0
        assert rewards[0].data["reason"] == "correct"

    def test_on_succeeded_numeric_wrong(self, hooks) -> None:
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            RolloutCreate(input={"question": "What is 2+2?", "answer": "4"})
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "5"})
        store.update_rollout(rid, RolloutPatch(status=RolloutState.RUNNING))
        store.update_rollout(rid, RolloutPatch(
            status=RolloutState.SUCCEEDED, last_attempt_id="a1"
        ))
        rewards = [e for e in store.query_events(rid) if e.event_type == "reward"]
        assert rewards[0].data["value"] == 0.0
        assert "wrong" in rewards[0].data["reason"]
