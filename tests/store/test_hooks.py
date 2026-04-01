"""Tests for rollout lifecycle hooks."""

from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agl_lite.hooks import RolloutHooks, load_hooks
from agl_lite.schemas.api import EnqueueRolloutRequest, PatchRolloutRequest
from agl_lite.schemas.rollout import RolloutConfig, RolloutMetadata, RolloutStatus
from agl_lite.store.memory import InMemoryStore


# ── Test hook implementations ────────────────────────────────────────


class TransformHooks(RolloutHooks):
    """on_enqueue: set pod_spec with image, stash ground_truth in metadata."""

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        raw = request.input if isinstance(request.input, dict) else {}
        if request.metadata is None:
            request.metadata = {}
        if isinstance(request.metadata, dict):
            request.metadata["ground_truth"] = raw.get("ground_truth", "")
        request.config = request.config or RolloutConfig()
        request.config.pod_spec = {
            "containers": [
                {
                    "name": "agent",
                    "image": "test-agent:dev",
                    "env": [{"name": "INJECTED", "value": "true"}],
                }
            ]
        }
        return request


class RewardHooks(RolloutHooks):
    """on_succeeded: compute reward from rollout.input + events, post reward event."""

    def on_succeeded(self, rollout: Any, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        # ground_truth can come from rollout.input or metadata (stashed by on_enqueue)
        gt = ""
        if isinstance(rollout.input, dict):
            gt = rollout.input.get("ground_truth", "")
        elif hasattr(rollout.metadata, "ground_truth"):
            gt = rollout.metadata.ground_truth
        # Find agent answer from events.
        answer = None
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "agent_output":
                    answer = evt.data.get("answer")
        reward = 1.0 if answer == gt else 0.0
        attempt_id = rollout.succeeded_attempt_id or "test"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {"value": reward})


class FailHooks(RolloutHooks):
    """on_failed: post a zero reward."""

    def on_failed(self, rollout: Any, store: InMemoryStore) -> None:
        store.add_event(rollout.rollout_id, "fail", "reward", {"value": 0.0, "reason": "failed"})


class ErrorHooks(RolloutHooks):
    """on_succeeded raises — should not crash the store."""

    def on_succeeded(self, rollout: Any, events: dict, store: InMemoryStore) -> None:
        raise RuntimeError("hook exploded")


# ── Tests ────────────────────────────────────────────────────────────


class TestOnStartup:
    def test_base_loads_pod_spec_from_env(self, tmp_path) -> None:
        """Base on_startup reads AGL_POD_SPEC_TEMPLATE and populates self._pod_spec."""
        import yaml
        pod_spec = {"containers": [{"name": "agent", "image": "auto:v1"}]}
        f = tmp_path / "pod-spec.yaml"
        f.write_text(yaml.dump(pod_spec))

        hooks = RolloutHooks()
        with patch.dict(os.environ, {"AGL_POD_SPEC_TEMPLATE": str(f)}):
            hooks.on_startup(InMemoryStore())

        assert hooks._pod_spec is not None
        assert hooks._pod_spec["containers"][0]["image"] == "auto:v1"

    def test_base_no_op_when_env_unset(self) -> None:
        """Base on_startup is a no-op when AGL_POD_SPEC_TEMPLATE is not set."""
        env = {k: v for k, v in os.environ.items() if k != "AGL_POD_SPEC_TEMPLATE"}
        hooks = RolloutHooks()
        with patch.dict(os.environ, env, clear=True):
            hooks.on_startup(InMemoryStore())  # must not raise
        assert hooks._pod_spec is None

    def test_subclass_can_call_super(self, tmp_path) -> None:
        """Subclass calling super().on_startup gets the env-loaded pod spec."""
        import yaml
        pod_spec = {"containers": [{"name": "agent", "image": "base:v1"}]}
        f = tmp_path / "pod-spec.yaml"
        f.write_text(yaml.dump(pod_spec))

        class MyHooks(RolloutHooks):
            index_loaded: bool = False

            def on_startup(self, store: InMemoryStore) -> None:
                super().on_startup(store)
                self.index_loaded = True  # simulate extra setup

        h = MyHooks()
        with patch.dict(os.environ, {"AGL_POD_SPEC_TEMPLATE": str(f)}):
            h.on_startup(InMemoryStore())

        assert h._pod_spec is not None
        assert h.index_loaded is True

    def test_copy_pod_spec_deep_copies(self) -> None:
        hooks = RolloutHooks()
        hooks._pod_spec = {"containers": [{"name": "agent", "image": "base:v1"}]}
        copy1 = hooks.copy_pod_spec()
        copy1["containers"][0]["image"] = "modified:v1"
        assert hooks._pod_spec["containers"][0]["image"] == "base:v1"  # original unchanged

    def test_copy_pod_spec_raises_if_not_loaded(self) -> None:
        hooks = RolloutHooks()
        with pytest.raises(RuntimeError, match="no pod spec loaded"):
            hooks.copy_pod_spec()

    def test_get_container_found(self) -> None:
        pod_spec = {"containers": [{"name": "agent", "image": "x"}, {"name": "sidecar"}]}
        c = RolloutHooks.get_container(pod_spec, "agent")
        assert c["image"] == "x"

    def test_get_container_not_found(self) -> None:
        pod_spec = {"containers": [{"name": "agent"}]}
        with pytest.raises(KeyError, match="nonexistent"):
            RolloutHooks.get_container(pod_spec, "nonexistent")


class TestOnEnqueue:
    def test_transforms_request(self) -> None:
        store = InMemoryStore(hooks=TransformHooks())
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input={"question": "What is 2+2?", "ground_truth": "4"},
            )
        ])
        # input stays as the raw dataset row (hook no longer transforms it)
        assert rollout.input == {"question": "What is 2+2?", "ground_truth": "4"}
        # metadata should have ground_truth stashed by hook (extra field)
        assert rollout.metadata.ground_truth == "4"
        # config.pod_spec set by hook with image and env var
        assert rollout.config.pod_spec is not None
        agent = RolloutHooks.get_container(rollout.config.pod_spec, "agent")
        assert agent["image"] == "test-agent:dev"
        assert any(e["name"] == "INJECTED" for e in agent.get("env", []))

    def test_no_hooks_passthrough(self) -> None:
        store = InMemoryStore()  # no hooks
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input="plain question",
            )
        ])
        assert rollout.input == "plain question"
        assert rollout.metadata == RolloutMetadata()

    def test_hook_error_prevents_creation(self) -> None:
        class BadEnqueueHook(RolloutHooks):
            def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
                raise ValueError("bad input")

        store = InMemoryStore(hooks=BadEnqueueHook())
        with pytest.raises(ValueError, match="bad input"):
            store.enqueue_rollouts([EnqueueRolloutRequest(input="test")])
        # No rollout should have been created.
        assert len(store._rollouts) == 0


class TestOnSucceeded:
    def _setup_succeeded_rollout(self, hooks: RolloutHooks) -> tuple[InMemoryStore, str]:
        """Helper: create rollout, add agent_output, transition to SUCCEEDED."""
        store = InMemoryStore(hooks=hooks)
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input="What is 2+2?",
                metadata={"ground_truth": "4"},
            )
        ])
        rid = rollout.rollout_id
        # Simulate agent posting output.
        store.add_event(rid, "attempt-1", "agent_output", {"answer": "4"})
        # Transition: QUEUING → RUNNING → SUCCEEDED
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="attempt-1"
        ))
        return store, rid

    def test_reward_posted_atomically(self) -> None:
        store, rid = self._setup_succeeded_rollout(RewardHooks())
        # Reward event should exist (posted by hook during transition).
        events = store.query_events(rid)
        reward_events = [e for e in events if e.event_type == "reward"]
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 1.0

    def test_wrong_answer_zero_reward(self) -> None:
        store = InMemoryStore(hooks=RewardHooks())
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(
                input="What is 2+2?",
                metadata={"ground_truth": "4"},
            )
        ])
        rid = rollout.rollout_id
        store.add_event(rid, "a1", "agent_output", {"answer": "5"})  # wrong
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.SUCCEEDED, succeeded_attempt_id="a1"
        ))
        events = store.query_events(rid)
        reward_events = [e for e in events if e.event_type == "reward"]
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 0.0

    def test_hook_error_does_not_crash_transition(self) -> None:
        store, rid = self._setup_succeeded_rollout(ErrorHooks())
        # Rollout should still be SUCCEEDED despite hook error.
        rollout = store.get_rollout(rid)
        assert rollout.status == RolloutStatus.SUCCEEDED
        # No reward event (hook crashed).
        events = store.query_events(rid)
        reward_events = [e for e in events if e.event_type == "reward"]
        assert len(reward_events) == 0

    def test_no_hooks_no_reward(self) -> None:
        store = InMemoryStore()  # no hooks
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(input="test")
        ])
        rid = rollout.rollout_id
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.SUCCEEDED))
        events = store.query_events(rid)
        assert len([e for e in events if e.event_type == "reward"]) == 0


class TestOnFailed:
    def test_on_failed_posts_zero_reward(self) -> None:
        store = InMemoryStore(hooks=FailHooks())
        [rollout] = store.enqueue_rollouts([
            EnqueueRolloutRequest(input="test")
        ])
        rid = rollout.rollout_id
        store.update_rollout(rid, PatchRolloutRequest(status=RolloutStatus.RUNNING))
        store.update_rollout(rid, PatchRolloutRequest(
            status=RolloutStatus.TERMINAL_FAILED, error_message="job crashed"
        ))
        events = store.query_events(rid)
        reward_events = [e for e in events if e.event_type == "reward"]
        assert len(reward_events) == 1
        assert reward_events[0].data["value"] == 0.0


class TestLoadHooks:
    def test_load_from_file(self, tmp_path: Path) -> None:
        hooks_file = tmp_path / "my_hooks.py"
        hooks_file.write_text(textwrap.dedent("""\
            from agl_lite.hooks import RolloutHooks
            from agl_lite.schemas.api import EnqueueRolloutRequest

            class MyHooks(RolloutHooks):
                def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
                    request.input = "transformed"
                    return request
        """))
        hooks = load_hooks(str(hooks_file))
        assert type(hooks).__name__ == "MyHooks"
        # Verify it works.
        req = EnqueueRolloutRequest(input="original")
        result = hooks.on_enqueue(req)
        assert result.input == "transformed"

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_hooks("/nonexistent/hooks.py")

    def test_load_no_subclass(self, tmp_path: Path) -> None:
        hooks_file = tmp_path / "empty.py"
        hooks_file.write_text("x = 1\n")
        with pytest.raises(ValueError, match="No RolloutHooks subclass"):
            load_hooks(str(hooks_file))

    def test_load_multiple_subclasses(self, tmp_path: Path) -> None:
        hooks_file = tmp_path / "multi.py"
        hooks_file.write_text(textwrap.dedent("""\
            from agl_lite.hooks import RolloutHooks

            class A(RolloutHooks): pass
            class B(RolloutHooks): pass
        """))
        with pytest.raises(ValueError, match="Multiple RolloutHooks subclasses"):
            load_hooks(str(hooks_file))
