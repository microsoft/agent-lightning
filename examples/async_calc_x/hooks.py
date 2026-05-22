"""Calc-X hooks for vLLM (real inference) mode.

Math problems sent to the model via AutoGen + MCP calculator agent.
Reward computed by numeric comparison (sympy-based) of the agent's
### ANSWER: <answer> ### with the ground truth.

Required env vars:
  AGL_POD_SPEC_TEMPLATE  path to examples/async_calc_x/job-template.yaml
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Add examples/async_calc_x to path so eval_utils is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_utils import scalar_are_results_same

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import RolloutConfig
from agl_lite.store.memory import InMemoryStore


class CalcXHooks(RolloutHooks):

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        raw = request.input if isinstance(request.input, dict) else {}
        question = raw.get("question", "")
        task_id = raw.get("id", "")

        # Build pod spec and inject per-sample env vars.
        pod_spec = self.copy_pod_spec()
        agent = self.get_container(pod_spec, "agent")
        agent.setdefault("env", [])

        # Pass question + id as JSON for the agent to parse.
        task_input = json.dumps({"question": question, "id": task_id})
        agent["env"].append({"name": "AGL_TASK_INPUT", "value": task_input})

        model_name = os.environ.get("AGL_MODEL_NAME", "")
        if model_name:
            agent["env"].append({"name": "AGL_MODEL_NAME", "value": model_name})

        if request.config is None:
            request.config = RolloutConfig()
        request.config.pod_spec = pod_spec
        return request

    def on_succeeded(self, rollout: "Rollout", events: dict[str, list[Any]], store: InMemoryStore) -> None:
        gt = ""
        if isinstance(rollout.input, dict):
            gt = rollout.input.get("result", "")

        answer = self._extract_answer(events)
        reward, reason = self._compute_reward(answer, str(gt))

        attempt_id = rollout.succeeded_attempt_id or "unknown"
        store.add_event(rollout.rollout_id, attempt_id, "reward", {
            "value": reward,
            "ground_truth": gt,
            "agent_answer": answer,
            "reason": reason,
        })

    def _extract_answer(self, events: dict[str, list[Any]]) -> str | None:
        for attempt_events in events.values():
            for evt in attempt_events:
                if evt.event_type == "agent_output":
                    return evt.data.get("answer")
        return None

    @classmethod
    def _compute_reward(cls, agent_answer: str | None, ground_truth: str) -> tuple[float, str]:
        if agent_answer is None:
            return 0.0, "no answer extracted"
        if scalar_are_results_same(agent_answer, ground_truth, 1e-2):
            return 1.0, "correct"
        return 0.0, f"wrong: {agent_answer!r} != {ground_truth!r}"
