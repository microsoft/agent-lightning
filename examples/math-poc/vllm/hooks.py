"""Math-poc hooks for vLLM (real inference) mode.

Plain questions sent to the model. Reward computed by numeric comparison
of the agent's \\boxed{answer} with the ground truth.
"""

from __future__ import annotations

import json
import os
from typing import Any

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig
from agl_lite.store.memory import InMemoryStore


class MathVllmHooks(RolloutHooks):

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        raw = request.input if isinstance(request.input, dict) else {}
        question = raw.get("question", "")

        # Set agent-facing task input (image, command, etc. come from job-template)
        if request.config is None:
            request.config = RolloutConfig(image="")
        request.config.environment_variables["AGL_TASK_INPUT"] = json.dumps(question)
        # Pass model name so agent knows which model to call
        model_name = os.environ.get("AGL_MODEL_NAME", "")
        if model_name:
            request.config.environment_variables.setdefault("AGL_MODEL_NAME", model_name)

        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        # Read ground_truth directly from rollout.input
        gt = ""
        if isinstance(rollout.input, dict):
            gt = rollout.input.get("answer", "")

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

    @staticmethod
    def _normalize_number(s: str) -> float | None:
        if not s:
            return None
        cleaned = s.strip().replace(",", "").replace("$", "").replace("%", "")
        try:
            return float(cleaned)
        except ValueError:
            return None

    @classmethod
    def _compute_reward(cls, agent_answer: str | None, ground_truth: str) -> tuple[float, str]:
        if agent_answer is None:
            return 0.0, "no answer extracted"
        agent_num = cls._normalize_number(agent_answer)
        gt_num = cls._normalize_number(ground_truth)
        if agent_num is None:
            return 0.0, f"agent answer not numeric: {agent_answer!r}"
        if gt_num is None:
            return 0.0, f"ground truth not numeric: {ground_truth!r}"
        if abs(agent_num - gt_num) < 1e-6:
            return 1.0, "correct"
        return 0.0, f"wrong: {agent_num} != {gt_num}"
