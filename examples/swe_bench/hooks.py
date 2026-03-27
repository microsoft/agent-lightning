"""SWE-bench hooks — task-specific logic for SWE-bench rollouts.

on_enqueue:  set per-instance Docker image, generate eval_script,
             inject env vars (AGL_TASK_INPUT, AGL_EVAL_SCRIPT, AGL_EVAL_META).

on_succeeded / on_failed: post zero-reward fallback if container didn't post one.
  Grading is done in the container using official swebench tools.
"""

from __future__ import annotations

import json
import os
from typing import Any

from swebench.harness.test_spec.test_spec import make_test_spec

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig
from agl_lite.store.memory import InMemoryStore


class SWEBenchHooks(RolloutHooks):

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        instance = request.input
        if not isinstance(instance, dict) or "instance_id" not in instance:
            raise ValueError("SWE-bench input must be a dict with 'instance_id'")

        instance_id = instance["instance_id"]

        # 1. Set per-instance Docker image.
        safe_id = instance_id.lower().replace("__", "_1776_")
        if request.config is None:
            request.config = RolloutConfig(image="")
        request.config.image = f"swebench/sweb.eval.x86_64.{safe_id}:latest"

        # 2. Generate eval_script via swebench (pure CPU, ~ms).
        test_spec = make_test_spec(instance)

        # 3. Inject env vars for the container entrypoint.
        request.config.environment_variables.update({
            "AGL_TASK_INPUT": instance.get("problem_statement", ""),
            "AGL_EVAL_SCRIPT": test_spec.eval_script,
            "AGL_EVAL_META": json.dumps({
                "FAIL_TO_PASS": test_spec.FAIL_TO_PASS,
                "PASS_TO_PASS": test_spec.PASS_TO_PASS,
                "instance_id": instance_id,
                "repo": instance.get("repo", ""),
                "version": instance.get("version", ""),
            }),
            "AGL_CODING_AGENT": os.environ.get("AGL_CODING_AGENT", "claude_code"),
        })

        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        """Post fallback reward if container didn't post one."""
        if self._has_reward_event(events):
            return  # Container already posted reward — nothing to do.

        instance_id = ""
        if isinstance(rollout.input, dict):
            instance_id = rollout.input.get("instance_id", "")

        attempt_id = rollout.succeeded_attempt_id or "unknown"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {
            "value": 0.0,
            "resolved": False,
            "instance_id": instance_id,
            "reason": "no reward event from container",
        })

    def on_failed(self, rollout: Rollout, store: InMemoryStore) -> None:
        instance_id = ""
        if isinstance(rollout.input, dict):
            instance_id = rollout.input.get("instance_id", "")
        attempt_id = "unknown"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {
            "value": 0.0,
            "resolved": False,
            "instance_id": instance_id,
            "reason": "rollout failed",
        })

    @staticmethod
    def _has_reward_event(events: dict[str, list[Any]]) -> bool:
        """Check if any attempt has a reward event."""
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "reward":
                    return True
        return False
