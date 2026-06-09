"""Math-poc hooks for mockai (echo) mode.

Mockai echoes the last user message, so we embed \\boxed{answer} in the question.
The agent's parser extracts it as the "model response".

Required env vars:
  AGL_POD_SPEC_TEMPLATE  path to examples/math-poc/job-template.yaml
"""

from __future__ import annotations

import json
from typing import Any

from agl_lite.hooks import RolloutHooks, TraceWriter
from agl_lite.schemas import Rollout, RolloutConfig, RolloutCreate


class MathMockHooks(RolloutHooks):
    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        raw = request.input if isinstance(request.input, dict) else {}
        question = raw.get("question", "")
        ground_truth = raw.get("answer", "")

        augmented = question + f"\n\\boxed{{{ground_truth}}}"

        # Build pod spec and inject per-sample env var.
        pod_spec = self.copy_pod_spec()
        agent = self.get_container(pod_spec, "agent")
        agent.setdefault("env", [])
        agent["env"].append({"name": "AGL_TASK_INPUT", "value": json.dumps(augmented)})

        if request.config is None:
            request.config = RolloutConfig()
        request.config.pod_spec = pod_spec
        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: TraceWriter) -> None:
        gt = ""
        if isinstance(rollout.input, dict):
            gt = rollout.input.get("answer", "")

        answer = self._extract_answer(events)
        reward = 1.0 if answer and answer.strip() == str(gt).strip() else 0.0

        attempt_id = rollout.last_attempt_id or "unknown"
        store.add_event(
            rollout.rollout_id,
            attempt_id,
            "reward",
            {
                "value": reward,
                "ground_truth": gt,
                "agent_answer": answer,
            },
        )

    def _extract_answer(self, events: dict[str, list[Any]]) -> str | None:
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "agent_output":
                    return evt.data.get("answer")
        return None
