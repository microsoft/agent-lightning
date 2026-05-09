"""SWE-bench hooks — task-specific logic for SWE-bench rollouts.

The base RolloutHooks.on_startup automatically loads the pod spec from
AGL_POD_SPEC_TEMPLATE (set in the project .env file).  This hook only needs
to customise on_enqueue (per-instance image + env vars) and on_succeeded /
on_failed (reward posting).

Required env vars:
  AGL_POD_SPEC_TEMPLATE  path to the pod spec YAML (e.g. examples/swe_bench/job-template.yaml)
"""

from __future__ import annotations

import json
import os
from typing import Any, cast

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
        image_namespace = os.environ.get("AGL_SWEBENCH_IMAGE_NAMESPACE", "swebench")

        # Generate eval script and image name from the same TestSpec so the hook
        # matches the images built by examples/swe_bench/build_images.py.
        test_spec = make_test_spec(cast(Any, instance), namespace=image_namespace)

        # 1. Deep copy template and set per-instance image.
        pod_spec = self.copy_pod_spec()
        agent = self.get_container(pod_spec, "agent")
        agent["image"] = test_spec.instance_image_key

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
