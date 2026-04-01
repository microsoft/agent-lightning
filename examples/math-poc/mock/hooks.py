"""Math-poc hooks for mockai (echo) mode.

Mockai echoes the last user message, so we embed \\boxed{answer} in the question.
The agent's parser extracts it as the "model response". Alternating pattern:
even index → correct answer, odd index → wrong answer (deterministic rewards).

Required env vars:
  AGL_POD_SPEC_TEMPLATE  path to examples/math-poc/job-template.yaml
"""

from __future__ import annotations

import json
from typing import Any

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig
from agl_lite.store.memory import InMemoryStore

WRONG_ANSWER = "WRONG"


class MathMockHooks(RolloutHooks):

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        raw = request.input if isinstance(request.input, dict) else {}
        question = raw.get("question", "")
        ground_truth = raw.get("answer", "")

        # Read sample index from metadata for alternating pattern.
        meta = request.metadata
        idx = 0
        if isinstance(meta, dict):
            idx = meta.get("sample_idx_in_batch", 0) or 0
        elif meta is not None:
            idx = getattr(meta, "sample_idx_in_batch", 0) or 0

        # Alternating: even=correct, odd=wrong (deterministic mock rewards).
        boxed_value = ground_truth if idx % 2 == 0 else WRONG_ANSWER
        augmented = question + f"\n\\boxed{{{boxed_value}}}"

        # Build pod spec and inject per-sample env var.
        pod_spec = self.copy_pod_spec()
        agent = self.get_container(pod_spec, "agent")
        agent.setdefault("env", [])
        agent["env"].append({"name": "AGL_TASK_INPUT", "value": json.dumps(augmented)})

        if request.config is None:
            request.config = RolloutConfig()
        request.config.pod_spec = pod_spec
        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        gt = ""
        if isinstance(rollout.input, dict):
            gt = rollout.input.get("answer", "")

        answer = self._extract_answer(events)
        reward = 1.0 if answer and answer.strip() == str(gt).strip() else 0.0

        attempt_id = rollout.succeeded_attempt_id or "unknown"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {
            "value": reward,
            "ground_truth": gt,
            "agent_answer": answer,
        })

    def _extract_answer(self, events: dict[str, list[Any]]) -> str | None:
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "agent_output":
                    return evt.data.get("answer")
        return None
