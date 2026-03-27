"""Tests for SWE-bench hooks."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.event import Event
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutMetadata, RolloutStatus
from agl_lite.store.memory import InMemoryStore


# Load hooks module dynamically to avoid swebench import at test collection time.
@pytest.fixture
def hooks():
    """Load SWEBenchHooks, mocking swebench imports."""
    mock_test_spec = MagicMock()
    mock_test_spec.eval_script = "#!/bin/bash\necho test"
    mock_test_spec.FAIL_TO_PASS = ["test_foo"]
    mock_test_spec.PASS_TO_PASS = ["test_bar"]
    mock_test_spec.instance_id = "repo__issue-123"

    with patch.dict("sys.modules", {
        "swebench": MagicMock(),
        "swebench.harness": MagicMock(),
        "swebench.harness.grading": MagicMock(),
        "swebench.harness.test_spec": MagicMock(),
        "swebench.harness.test_spec.test_spec": MagicMock(),
    }):
        import importlib
        import examples.swe_bench.hooks as hooks_mod
        importlib.reload(hooks_mod)
        hooks_mod.make_test_spec = MagicMock(return_value=mock_test_spec)
        yield hooks_mod


@pytest.fixture
def swe_instance() -> dict:
    """Minimal SWE-bench instance for testing."""
    return {
        "instance_id": "astropy__astropy-12907",
        "problem_statement": "Separability matrix bug with nested CompoundModels",
        "repo": "astropy/astropy",
        "base_commit": "abc123",
        "FAIL_TO_PASS": '["test_foo"]',
        "PASS_TO_PASS": '["test_bar"]',
        "version": "4.3",
    }


class TestOnEnqueue:
    def test_sets_per_instance_image(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        assert result.config.image == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"

    def test_sets_env_vars(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        env = result.config.environment_variables
        assert env["AGL_TASK_INPUT"] == swe_instance["problem_statement"]
        assert "AGL_EVAL_SCRIPT" in env
        assert "AGL_EVAL_META" in env

    def test_eval_meta_contains_test_lists(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        meta = json.loads(result.config.environment_variables["AGL_EVAL_META"])
        assert "FAIL_TO_PASS" in meta
        assert "PASS_TO_PASS" in meta
        assert "instance_id" in meta

    def test_eval_meta_contains_repo_and_version(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        meta = json.loads(result.config.environment_variables["AGL_EVAL_META"])
        assert meta["repo"] == "astropy/astropy"
        assert meta["version"] == "4.3"

    def test_sets_coding_agent_from_env(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        with patch.dict(os.environ, {"AGL_CODING_AGENT": "claude_code"}):
            req = EnqueueRolloutRequest(input=swe_instance)
            result = hook.on_enqueue(req)
        assert result.config.environment_variables["AGL_CODING_AGENT"] == "claude_code"

    def test_default_coding_agent(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("AGL_CODING_AGENT", None)
            req = EnqueueRolloutRequest(input=swe_instance)
            result = hook.on_enqueue(req)
        assert result.config.environment_variables["AGL_CODING_AGENT"] == "claude_code"

    def test_rejects_missing_instance_id(self, hooks) -> None:
        hook = hooks.SWEBenchHooks()
        req = EnqueueRolloutRequest(input={"problem_statement": "no id"})
        with pytest.raises(ValueError, match="instance_id"):
            hook.on_enqueue(req)

    def test_creates_config_if_none(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        req = EnqueueRolloutRequest(input=swe_instance, config=None)
        result = hook.on_enqueue(req)
        assert result.config is not None
        assert result.config.image != ""


def _make_rollout(instance: dict, rollout_id: str = "test-123", status: RolloutStatus = RolloutStatus.SUCCEEDED) -> Rollout:
    import time
    now = time.time()
    return Rollout(
        rollout_id=rollout_id,
        status=status,
        input=instance,
        config=RolloutConfig(image="test"),
        metadata=RolloutMetadata(),
        succeeded_attempt_id="attempt-1" if status == RolloutStatus.SUCCEEDED else None,
        created_at=now,
        updated_at=now,
    )


class TestOnSucceeded:
    def test_no_reward_posted_when_container_already_posted(self, hooks, swe_instance: dict) -> None:
        """on_succeeded should do nothing if a reward event already exists."""
        hook = hooks.SWEBenchHooks()
        rollout = _make_rollout(swe_instance)

        events = {
            "attempt-1": [
                Event(event_type="reward", rollout_id="test-123", attempt_id="attempt-1",
                      timestamp=1.0, data={"value": 1.0, "resolved": True}),
            ]
        }

        store = InMemoryStore()
        store._rollouts["test-123"] = rollout
        store._events["test-123"] = {}

        hook.on_succeeded(rollout, events, store)

        # No additional reward event should be posted.
        reward_events = store.query_events("test-123", attempt_id="attempt-1", event_type="reward")
        assert len(reward_events) == 0  # Store has no new events; existing one was in the input dict.

    def test_fallback_reward_when_no_reward_event(self, hooks, swe_instance: dict) -> None:
        """on_succeeded posts zero reward if container didn't post one."""
        hook = hooks.SWEBenchHooks()
        rollout = _make_rollout(swe_instance)

        events = {
            "attempt-1": [
                Event(event_type="agent_output", rollout_id="test-123", attempt_id="attempt-1",
                      timestamp=1.0, data={"patch": "diff", "instance_id": "astropy__astropy-12907"}),
            ]
        }

        store = InMemoryStore()
        store._rollouts["test-123"] = rollout
        store._events["test-123"] = {}

        hook.on_succeeded(rollout, events, store)

        reward_events = store.query_events("test-123", attempt_id="attempt-1", event_type="reward")
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 0.0
        assert "no reward event" in reward_events[0].data["reason"]

    def test_fallback_reward_with_empty_events(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        rollout = _make_rollout(swe_instance)

        events: dict[str, list[Any]] = {"attempt-1": []}

        store = InMemoryStore()
        store._rollouts["test-123"] = rollout
        store._events["test-123"] = {}

        hook.on_succeeded(rollout, events, store)

        reward_events = store.query_events("test-123", attempt_id="attempt-1", event_type="reward")
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 0.0


class TestOnFailed:
    def test_posts_zero_reward(self, hooks, swe_instance: dict) -> None:
        hook = hooks.SWEBenchHooks()
        rollout = _make_rollout(swe_instance, rollout_id="test-456", status=RolloutStatus.TERMINAL_FAILED)

        store = InMemoryStore()
        store._rollouts["test-456"] = rollout
        store._events["test-456"] = {}

        hook.on_failed(rollout, store)

        reward_events = store.query_events("test-456", attempt_id="unknown", event_type="reward")
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 0.0
        assert reward_events[0].data["reason"] == "rollout failed"
