"""SWE-bench hooks — task-specific logic for SWE-bench rollouts.

on_startup:  load pod spec template from SWE_POD_SPEC_TEMPLATE env var.
on_enqueue:  deep copy template, set per-instance image, inject env vars
             (AGL_TASK_INPUT, AGL_EVAL_SCRIPT, AGL_EVAL_META, etc.),
             set config.timeout from template.

on_succeeded / on_failed: post zero-reward fallback if container didn't post one.
  Grading is done in the container using official swebench tools.

Required env vars:
  SWE_POD_SPEC_TEMPLATE  path to the pod spec YAML (e.g. examples/swe_bench/job-template.yaml)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import yaml
from swebench.harness.test_spec.test_spec import make_test_spec

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig
from agl_lite.store.memory import InMemoryStore


class SWEBenchHooks(RolloutHooks):

    def on_startup(self, store: InMemoryStore) -> None:
        """Load pod spec template once from SWE_POD_SPEC_TEMPLATE env var."""
        path = os.environ["SWE_POD_SPEC_TEMPLATE"]
        self._pod_spec = yaml.safe_load(Path(path).read_text())

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        instance = request.input
        if not isinstance(instance, dict) or "instance_id" not in instance:
            raise ValueError("SWE-bench input must be a dict with 'instance_id'")

        instance_id = instance["instance_id"]
        safe_id = instance_id.lower().replace("__", "_1776_")

        # 1. Deep copy template and set per-instance image.
        pod_spec = self.copy_pod_spec()
        agent = self.get_container(pod_spec, "agent")
        agent["image"] = f"swebench/sweb.eval.x86_64.{safe_id}:latest"

        # 2. Generate eval script via swebench (pure CPU, ~ms).
        test_spec = make_test_spec(instance)

        # 3. Inject per-sample env vars directly into the agent container.
        agent.setdefault("env", [])
        per_sample_env = {
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
        }
        for name, value in per_sample_env.items():
            agent["env"].append({"name": name, "value": value})

        # 4. Hoist activeDeadlineSeconds from pod spec root → config.timeout.
        timeout = pod_spec.pop("activeDeadlineSeconds", None)

        if request.config is None:
            request.config = RolloutConfig()
        request.config.pod_spec = pod_spec
        if timeout is not None:
            request.config.timeout = int(timeout)

        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        """Post fallback reward if container didn't post one."""
        if self._has_reward_event(events):
            return
        attempt_id = rollout.succeeded_attempt_id or next(iter(events), "unknown")
        store.add_event(rollout.rollout_id, attempt_id, "reward", {"value": 0.0, "reason": "no_reward_posted"})

    def on_failed(self, rollout: Rollout, store: InMemoryStore) -> None:
        """Post zero reward on failure."""
        store.add_event(rollout.rollout_id, "failed", "reward", {"value": 0.0, "reason": "terminal_failed"})

    @staticmethod
    def _has_reward_event(events: dict[str, list[Any]]) -> bool:
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "reward":
                    return True
        return False
