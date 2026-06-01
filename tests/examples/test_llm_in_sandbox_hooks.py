from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from agl_lite.schemas.api import EnqueueRolloutRequest

HOOKS_PATH = Path(__file__).resolve().parents[2] / "examples/llm-in-sandbox/hooks.py"
_spec = importlib.util.spec_from_file_location("llm_in_sandbox_hooks", HOOKS_PATH)
assert _spec is not None and _spec.loader is not None
hooks_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks_module)


def _hook() -> Any:
    hook = hooks_module.LlmInSandboxHooks()
    hook._pod_spec = {"containers": [{"name": "agent", "env": []}]}
    return hook


def _env_by_name(result: EnqueueRolloutRequest) -> dict[str, str]:
    assert result.config is not None
    assert result.config.pod_spec is not None
    container = result.config.pod_spec["containers"][0]
    return {entry["name"]: entry["value"] for entry in container["env"]}


def test_on_enqueue_uses_training_temperature_from_metadata() -> None:
    request = EnqueueRolloutRequest(
        input={"data_source": "instruct_pretrain", "extra_info": {"data_index": 0}},
        metadata={"temperature": 1.0, "is_train": True},
    )

    result = _hook().on_enqueue(request)

    assert _env_by_name(result)["AGL_LLM_TEMPERATURE"] == "1.0"


def test_on_enqueue_uses_validation_temperature_from_metadata() -> None:
    request = EnqueueRolloutRequest(
        input={"data_source": "math_mini", "extra_info": {"data_index": 0}},
        metadata={"temperature": 0.0, "is_train": False},
    )

    result = _hook().on_enqueue(request)

    assert _env_by_name(result)["AGL_LLM_TEMPERATURE"] == "0.0"




def test_on_enqueue_defaults_validation_temperature_to_zero() -> None:
    request = EnqueueRolloutRequest(
        input={"data_source": "chem_mini", "extra_info": {"data_index": 0}},
        metadata={"is_train": False},
    )

    result = _hook().on_enqueue(request)

    assert _env_by_name(result)["AGL_LLM_TEMPERATURE"] == "0.0"
