"""Rollout hooks for the llm-in-sandbox example."""

from __future__ import annotations

import json
import os
from typing import Any

from agl_lite.hooks import RolloutHooks, TraceWriter
from agl_lite.schemas.api import EnqueueRolloutRequest
from agl_lite.schemas.rollout import Rollout, RolloutConfig

DATASET_DEFAULTS = {
    "instruct_pretrain": ("llm_sandbox_instruct_pretrain", "train_verl.json"),
    "chem_mini": ("llm_sandbox_chem_mini", "test_verl.json"),
    "math_mini": ("llm_sandbox_math_mini", "test_verl.json"),
}
DEFAULT_MAX_TOKENS_PER_CALL = "20000"


class LlmInSandboxHooks(RolloutHooks):
    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        sample: dict[str, Any] = request.input if isinstance(request.input, dict) else {}
        raw_extra_info = sample.get("extra_info")
        extra_info: dict[str, Any] = raw_extra_info if isinstance(raw_extra_info, dict) else {}
        data_source = str(sample.get("data_source", ""))
        folder_name, filename = DATASET_DEFAULTS.get(data_source, ("", ""))
        folder_name = str(extra_info.get("data_folder_name") or folder_name)
        filename = str(extra_info.get("data_filename") or filename)
        data_index = extra_info.get("data_index", extra_info.get("index", sample.get("index", 0)))

        pod_spec = self.copy_pod_spec()
        agent = self.get_container(pod_spec, "agent")
        agent.setdefault("env", [])
        metadata = request.metadata
        if isinstance(metadata, dict):
            temperature = metadata.get("temperature")
            is_train = metadata.get("is_train")
        else:
            temperature = getattr(metadata, "temperature", None)
            is_train = getattr(metadata, "is_train", None)
        if temperature is None and is_train is False:
            temperature = 0.0
        temperature_value = (
            str(temperature) if temperature is not None else os.environ.get("AGL_LLM_TEMPERATURE", "1.0")
        )

        per_sample_env = {
            "AGL_TASK_INPUT": json.dumps(sample),
            "AGL_MODEL_NAME": os.environ.get("AGL_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507"),
            "AGL_OPENAI_MODEL_PREFIX": os.environ.get("AGL_OPENAI_MODEL_PREFIX", "openai/"),
            "AGL_LLM_TEMPERATURE": temperature_value,
            "DATA_FOLDER_NAME": folder_name,
            "DATA_FILENAME": filename,
            "DATA_INDEX": str(data_index),
            "OPENAI_TIMEOUT": os.environ.get("OPENAI_TIMEOUT", "900"),
            "MAX_TOKENS_PER_CALL": os.environ.get("MAX_TOKENS_PER_CALL") or DEFAULT_MAX_TOKENS_PER_CALL,
        }
        for name, value in per_sample_env.items():
            agent["env"].append({"name": name, "value": value})

        if request.config is None:
            request.config = RolloutConfig()
        request.config.pod_spec = pod_spec
        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: TraceWriter) -> None:
        if self._has_reward_event(events):
            return
        attempt_id = rollout.succeeded_attempt_id or next(iter(events), "unknown")
        store.add_event(
            rollout.rollout_id,
            attempt_id,
            "reward",
            {"value": 0.0, "reason": "no_reward_posted_by_agent", "source": "fallback"},
        )

    def on_failed(self, rollout: Rollout, store: TraceWriter) -> None:
        store.add_event(
            rollout.rollout_id,
            "failed",
            "reward",
            {"value": 0.0, "reason": "terminal_failed", "source": "fallback"},
        )

    @staticmethod
    def _has_reward_event(events: dict[str, list[Any]]) -> bool:
        for attempt_events in events.values():
            for event in attempt_events:
                if event.event_type == "reward":
                    return True
        return False
