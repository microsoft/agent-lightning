"""Math-poc hooks for mockai (echo) mode.

Mockai echoes the last user message, so we embed \\boxed{answer} in the question.
The agent's parser extracts it as the "model response". Alternating pattern:
even index → correct answer, odd index → wrong answer (deterministic rewards).
"""

from __future__ import annotations

import json
from typing import Any

from agl_lite.hooks import RolloutHooks
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig
from agl_lite.store.memory import InMemoryStore

WRONG_ANSWER = "WRONG"
MODEL_NAME = "mock-llm"


class MathMockHooks(RolloutHooks):

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        raw = request.input if isinstance(request.input, dict) else {}
        question = raw.get("question", "")
        ground_truth = raw.get("answer", "")

        # Read sample index from metadata
        meta = request.metadata
        idx = 0
        if isinstance(meta, dict):
            idx = meta.get("sample_idx_in_batch", 0) or 0
        elif meta is not None:
            idx = meta.sample_idx_in_batch or 0

        # Alternating: even=correct, odd=wrong (deterministic mock rewards)
        correct = idx % 2 == 0
        boxed_value = ground_truth if correct else WRONG_ANSWER
        augmented = question + f"\n\\boxed{{{boxed_value}}}"

        # Stash grading context in metadata
        if meta is None:
            request.metadata = {"ground_truth": ground_truth, "expect_correct": correct, "boxed_value": boxed_value}
        elif isinstance(meta, dict):
            meta["ground_truth"] = ground_truth
            meta["expect_correct"] = correct
            meta["boxed_value"] = boxed_value
        else:
            meta.ground_truth = ground_truth  # type: ignore[attr-defined]
            meta.expect_correct = correct  # type: ignore[attr-defined]
            meta.boxed_value = boxed_value  # type: ignore[attr-defined]

        # Set agent-facing config
        request.config = request.config or RolloutConfig(image="")
        request.config.image = request.config.image or "math-agent:dev"
        request.config.environment_variables["AGL_TASK_INPUT"] = json.dumps(augmented)
        request.config.environment_variables.setdefault("AGL_MODEL_NAME", MODEL_NAME)

        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        gt = getattr(rollout.metadata, "ground_truth", "")
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
