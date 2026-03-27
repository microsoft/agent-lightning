"""SWE-bench hooks — task-specific logic for SWE-bench rollouts.

on_enqueue:  set per-instance Docker image, generate eval_script,
             inject env vars (AGL_TASK_INPUT, AGL_EVAL_SCRIPT, AGL_EVAL_META).

on_succeeded: read test_output artifact from disk, grade using official
              swebench get_eval_report(), post reward event.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec

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
            }),
            "AGL_CODING_AGENT": os.environ.get("AGL_CODING_AGENT", "claude_code"),
        })

        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        instance = rollout.input
        if not isinstance(instance, dict):
            return
        instance_id = instance.get("instance_id", "")

        # 1. Extract patch from agent_output event.
        patch = self._extract_patch(events)

        # 2. Find test_output artifact (written to disk by store).
        artifact_path = self._find_artifact(events, "test_output.txt")

        # 3. Grade using official swebench tools.
        reward = 0.0
        resolved = False
        reason = "no artifact"

        if artifact_path and Path(artifact_path).exists():
            try:
                test_spec = make_test_spec(instance)
                prediction = {
                    "instance_id": instance_id,
                    "model_patch": patch or "",
                    "model_name_or_path": "agl-lite",
                }
                report = get_eval_report(
                    test_spec=test_spec,
                    prediction=prediction,
                    test_log_path=artifact_path,
                    include_tests_status=True,
                )
                resolved = report.get(instance_id, {}).get("resolved", False)
                reward = 1.0 if resolved else 0.0
                reason = "resolved" if resolved else "not resolved"
            except Exception as e:
                reason = f"grading error: {e}"
        elif patch is None:
            reason = "no patch"

        # 4. Post reward event.
        attempt_id = rollout.succeeded_attempt_id or "unknown"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {
            "value": reward,
            "resolved": resolved,
            "instance_id": instance_id,
            "patch_size": len(patch) if patch else 0,
            "reason": reason,
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
    def _extract_patch(events: dict[str, list[Any]]) -> str | None:
        """Extract patch from the latest agent_output event."""
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "agent_output" and "patch" in evt.data:
                    return evt.data["patch"]
        return None

    @staticmethod
    def _find_artifact(events: dict[str, list[Any]], filename: str) -> str | None:
        """Find artifact file path by filename."""
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "artifact" and evt.data.get("filename") == filename:
                    return evt.data.get("path")
        return None
