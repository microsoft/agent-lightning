"""Tests for rollout lifecycle hooks."""

from __future__ import annotations

import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from agl_lite.hooks import RolloutHooks, load_hooks
from agl_lite.schemas.api import EnqueueRolloutRequest, PatchRolloutRequest
from agl_lite.schemas.rollout import RolloutConfig, RolloutMetadata, RolloutStatus
from agl_lite.store.memory import InMemoryStore


# ── Test hook implementations ────────────────────────────────────────


class TransformHooks(RolloutHooks):
    """on_enqueue: set config, stash ground_truth in metadata."""

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        raw = request.input if isinstance(request.input, dict) else {}
        if request.metadata is None:
            request.metadata = {}
        if isinstance(request.metadata, dict):
            request.metadata["ground_truth"] = raw.get("ground_truth", "")
        request.config = request.config or RolloutConfig(image="")
        request.config.image = "test-agent:dev"
        request.config.environment_variables["INJECTED"] = "true"
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
        # config should be set by hook
        assert rollout.config.image == "test-agent:dev"
        assert rollout.config.environment_variables["INJECTED"] == "true"

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
