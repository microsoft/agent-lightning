"""Tests for SWE-bench hooks."""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.event import Event
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutMetadata, RolloutStatus
from agl_lite.store.memory import InMemoryStore

# Minimal pod spec that on_startup would normally load from disk.
_MINIMAL_POD_SPEC = {
    "containers": [
        {
            "name": "agent",
            "image": "placeholder",
            "command": ["bash", "/agl/agents/entrypoint.sh"],
            "imagePullPolicy": "IfNotPresent",
        }
    ],
    "activeDeadlineSeconds": 5400,
}


# Load hooks module dynamically to avoid swebench import at test collection time.
@pytest.fixture
def hooks():
    """Load SWEBenchHooks module, mocking swebench imports."""
    def make_mock_test_spec(instance: dict[str, Any], namespace: str | None = None, **_: Any) -> MagicMock:
        instance_id = instance["instance_id"]
        safe_id = instance_id.lower().replace("__", "_1776_")
        image_prefix = f"{namespace}/" if namespace else ""
        mock_test_spec = MagicMock()
        mock_test_spec.eval_script = "#!/bin/bash\necho test"
        mock_test_spec.FAIL_TO_PASS = ["test_foo"]
        mock_test_spec.PASS_TO_PASS = ["test_bar"]
        mock_test_spec.instance_id = instance_id
        mock_test_spec.instance_image_key = f"{image_prefix}sweb.eval.x86_64.{safe_id}:latest"
        return mock_test_spec

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
        hooks_mod.make_test_spec = MagicMock(side_effect=make_mock_test_spec)
        yield hooks_mod


@pytest.fixture
def hook(hooks):
    """SWEBenchHooks instance with pod spec pre-loaded (bypasses on_startup file I/O)."""
    import copy
    h = hooks.SWEBenchHooks()
    h._pod_spec = copy.deepcopy(_MINIMAL_POD_SPEC)
    return h


@pytest.fixture
def swe_instance() -> dict:
    return {
        "instance_id": "astropy__astropy-12907",
        "problem_statement": "Separability matrix bug with nested CompoundModels",
        "repo": "astropy/astropy",
        "base_commit": "abc123",
        "version": "4.3",
    }


class TestOnStartup:
    def test_loads_pod_spec_from_agl_env_var(self, hooks, tmp_path) -> None:
        """Base on_startup reads AGL_POD_SPEC_TEMPLATE; SWEBenchHooks inherits this."""
        import yaml
        pod_spec = {"containers": [{"name": "agent", "image": "test:v1"}]}
        template_file = tmp_path / "job-template.yaml"
        template_file.write_text(yaml.dump(pod_spec))

        h = hooks.SWEBenchHooks()
        with patch.dict(os.environ, {"AGL_POD_SPEC_TEMPLATE": str(template_file)}):
            from agl_lite.store.memory import InMemoryStore
            h.on_startup(InMemoryStore())

        assert h._pod_spec is not None
        assert h._pod_spec["containers"][0]["name"] == "agent"

    def test_missing_env_var_leaves_pod_spec_none(self, hooks) -> None:
        """No AGL_POD_SPEC_TEMPLATE → _pod_spec stays None (copy_pod_spec raises later)."""
        h = hooks.SWEBenchHooks()
        env = {k: v for k, v in os.environ.items() if k != "AGL_POD_SPEC_TEMPLATE"}
        with patch.dict(os.environ, env, clear=True):
            h.on_startup(InMemoryStore())  # must not raise
        assert h._pod_spec is None


class TestOnEnqueue:
    def test_sets_per_instance_image_in_pod_spec(self, hook, swe_instance) -> None:
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        agent = RolloutHooks.get_container(result.config.pod_spec, "agent")
        assert agent["image"] == "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"

    def test_env_vars_injected_into_agent_container(self, hook, swe_instance) -> None:
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        agent = RolloutHooks.get_container(result.config.pod_spec, "agent")
        env = {e["name"]: e["value"] for e in agent.get("env", [])}
        assert env["AGL_TASK_INPUT"] == swe_instance["problem_statement"]
        assert "AGL_EVAL_SCRIPT" in env
        assert "AGL_EVAL_META" in env

    def test_eval_meta_contains_test_lists(self, hook, swe_instance) -> None:
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        agent = RolloutHooks.get_container(result.config.pod_spec, "agent")
        env = {e["name"]: e["value"] for e in agent.get("env", [])}
        meta = json.loads(env["AGL_EVAL_META"])
        assert "FAIL_TO_PASS" in meta
        assert "PASS_TO_PASS" in meta
        assert meta["instance_id"] == "astropy__astropy-12907"
        assert meta["repo"] == "astropy/astropy"
        assert meta["version"] == "4.3"

    def test_timeout_hoisted_to_config(self, hook, swe_instance) -> None:
        """activeDeadlineSeconds from pod spec root is moved to config.timeout."""
        req = EnqueueRolloutRequest(input=swe_instance)
        result = hook.on_enqueue(req)
        assert result.config.timeout == 5400
        assert "activeDeadlineSeconds" not in result.config.pod_spec

    def test_coding_agent_from_env(self, hook, swe_instance) -> None:
        with patch.dict(os.environ, {"AGL_CODING_AGENT": "claude_code"}):
            req = EnqueueRolloutRequest(input=swe_instance)
            result = hook.on_enqueue(req)
        agent = RolloutHooks.get_container(result.config.pod_spec, "agent")
        env = {e["name"]: e["value"] for e in agent.get("env", [])}
        assert env["AGL_CODING_AGENT"] == "claude_code"

    def test_creates_config_if_none(self, hook, swe_instance) -> None:
        req = EnqueueRolloutRequest(input=swe_instance, config=None)
        result = hook.on_enqueue(req)
        assert result.config is not None
        assert result.config.pod_spec is not None

    def test_rejects_missing_instance_id(self, hook) -> None:
        req = EnqueueRolloutRequest(input={"problem_statement": "no id"})
        with pytest.raises(ValueError, match="instance_id"):
            hook.on_enqueue(req)

    def test_original_pod_spec_not_mutated(self, hook, swe_instance) -> None:
        """Each on_enqueue call gets a fresh deep copy — no cross-request mutation."""
        original_image = hook._pod_spec["containers"][0]["image"]
        req = EnqueueRolloutRequest(input=swe_instance)
        hook.on_enqueue(req)
        assert hook._pod_spec["containers"][0]["image"] == original_image


def _make_rollout(instance: dict, rollout_id: str = "test-123",
                  status: RolloutStatus = RolloutStatus.SUCCEEDED) -> Rollout:
    import time
    now = time.time()
    return Rollout(
        rollout_id=rollout_id,
        status=status,
        input=instance,
        config=RolloutConfig(),
        metadata=RolloutMetadata(),
        succeeded_attempt_id="attempt-1" if status == RolloutStatus.SUCCEEDED else None,
        created_at=now,
        updated_at=now,
    )


class TestOnSucceeded:
    def test_no_extra_reward_when_already_posted(self, hook, swe_instance) -> None:
        rollout = _make_rollout(swe_instance)
        events = {
            "attempt-1": [
                Event(event_type="reward", rollout_id="test-123", attempt_id="attempt-1",
                      timestamp=1.0, data={"value": 1.0}),
            ]
        }
        store = InMemoryStore()
        store._rollouts["test-123"] = rollout
        store._events["test-123"] = {}
        hook.on_succeeded(rollout, events, store)
        # No new events added to store.
        assert store._events["test-123"] == {}

    def test_fallback_reward_when_none_posted(self, hook, swe_instance) -> None:
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
    def test_posts_zero_reward(self, hook, swe_instance) -> None:
        rollout = _make_rollout(swe_instance, rollout_id="test-456",
                                status=RolloutStatus.TERMINAL_FAILED)
        store = InMemoryStore()
        store._rollouts["test-456"] = rollout
        store._events["test-456"] = {}
        hook.on_failed(rollout, store)
        reward_events = store.query_events("test-456", attempt_id="failed", event_type="reward")
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 0.0
        assert reward_events[0].data["reason"] == "terminal_failed"
