# Copyright (c) Microsoft. All rights reserved.

"""Environment-configured SHAPER bundle for VLABench."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from agentlightning.types import LLM, PromptTemplate
from contrib.recipes.shaper.reproduce import ReproductionBundle

from ..common import load_text
from .actor_contract import (
    CHECKPOINT_MANIFEST_SHA256,
    CHECKPOINT_REPOSITORY,
    CHECKPOINT_REVISION,
)
from .agent import VLABenchAgent, VLABenchRuntimeConfig
from .contracts import (
    HARNESS_CONTRACT,
    OPENPI_COMMIT,
    OPENPI_REPOSITORY,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    check_upstream_source,
    make_harness_validator,
    validate_skill,
)
from .dataset import TRACK_NAME, load_reported_protocol_datasets, task_ids
from .openpi_identity import REPORTED_THREE_CAMERA
from .roles import VLABenchRoleProtocol

PROMPT_DIR = Path(__file__).parent / "prompts"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before building the VLABench SHAPER bundle.")
    return value


def _planner_resource() -> LLM:
    api_key_env = os.environ.get("SHAPER_API_KEY_ENV", "OPENAI_API_KEY")
    sampling: dict[str, Any] = {
        "max_completion_tokens": int(os.environ.get("SHAPER_PLANNER_MAX_TOKENS", "32768")),
        "optimizer_max_completion_tokens": int(os.environ.get("SHAPER_OPTIMIZER_MAX_TOKENS", "65536")),
        "timeout": float(os.environ.get("SHAPER_PLANNER_TIMEOUT", "300")),
        "max_retries": int(os.environ.get("SHAPER_PLANNER_RETRIES", "2")),
        "temperature": float(os.environ.get("SHAPER_PLANNER_TEMPERATURE", "1.0")),
        "top_p": float(os.environ.get("SHAPER_PLANNER_TOP_P", "0.95")),
        "presence_penalty": float(os.environ.get("SHAPER_PLANNER_PRESENCE_PENALTY", "0.0")),
    }
    extra_body = os.environ.get("SHAPER_PLANNER_EXTRA_BODY", "").strip()
    if extra_body:
        parsed: object = json.loads(extra_body)
        if not isinstance(parsed, dict):
            raise ValueError("SHAPER_PLANNER_EXTRA_BODY must be a JSON object.")
        sampling["extra_body"] = parsed
    return LLM(
        endpoint=_required_env("SHAPER_PLANNER_ENDPOINT"),
        model=_required_env("SHAPER_MODEL"),
        api_key=os.environ.get(api_key_env),
        sampling_parameters=sampling,
    )


def build_bundle() -> ReproductionBundle[dict[str, Any]]:
    """Build the complete VLABench training bundle from environment variables."""

    runtime = VLABenchRuntimeConfig(
        vlabench_root=Path(_required_env("VLABENCH_ROOT")).expanduser().resolve(),
        vla_host=os.environ.get("VLABENCH_VLA_HOST", "127.0.0.1"),
        vla_port=int(os.environ.get("VLABENCH_VLA_PORT", "8000")),
        vla_replan_steps=int(os.environ.get("VLABENCH_VLA_REPLAN_STEPS", "5")),
        vla_inference_timeout_seconds=float(os.environ.get("VLABENCH_VLA_TIMEOUT", "300")),
        max_vlm_rounds=int(os.environ.get("VLABENCH_MAX_VLM_ROUNDS", "10")),
        default_round_steps=int(os.environ.get("VLABENCH_DEFAULT_ROUND_STEPS", "200")),
        min_round_steps=int(os.environ.get("VLABENCH_MIN_ROUND_STEPS", "1")),
        planner_max_completion_tokens=int(os.environ.get("SHAPER_PLANNER_MAX_TOKENS", "32768")),
        max_substeps=int(os.environ.get("VLABENCH_MAX_SUBSTEPS", "1")),
        joint_tolerance=float(os.environ.get("VLABENCH_JOINT_TOLERANCE", "0.01")),
        reset_wait_steps=int(os.environ.get("VLABENCH_RESET_WAIT_STEPS", "10")),
        harness_timeout_seconds=float(os.environ.get("SHAPER_HARNESS_TIMEOUT", "3")),
        harness_memory_limit_mb=int(os.environ.get("SHAPER_HARNESS_MEMORY_MB", "768")),
        harness_max_output_chars=int(os.environ.get("SHAPER_HARNESS_MAX_OUTPUT_CHARS", "32000000")),
        observation_schema=os.environ.get("VLABENCH_OBSERVATION_SCHEMA", REPORTED_THREE_CAMERA),
        expected_actor_id=_required_env("VLABENCH_ACTOR_ID"),
        expected_policy_config=_required_env("VLABENCH_OPENPI_POLICY_CONFIG"),
    )
    source_errors = check_upstream_source(runtime.vlabench_root)
    if source_errors:
        raise RuntimeError("Unsupported VLABench checkout: " + "; ".join(source_errors))
    track_name = os.environ.get("VLABENCH_TRACK", TRACK_NAME)
    train, validation = load_reported_protocol_datasets(
        runtime.vlabench_root,
        track_name=track_name,
        max_steps=int(os.environ.get("VLABENCH_MAX_STEPS", "400")),
    )
    validator = make_harness_validator(
        timeout_seconds=runtime.harness_timeout_seconds,
        memory_limit_mb=runtime.harness_memory_limit_mb,
        max_output_chars=runtime.harness_max_output_chars,
    )
    planner = _planner_resource()
    resources = {
        "planner_llm": planner,
        "skill": PromptTemplate(template=load_text(PROMPT_DIR, "seed_skill.txt"), engine="f-string"),
        "harness": PromptTemplate(template=load_text(PROMPT_DIR, "seed_harness.py"), engine="f-string"),
    }
    return ReproductionBundle(
        agent=VLABenchAgent(runtime),
        train_dataset=train,
        val_dataset=validation,
        initial_resources=resources,
        planner_resource_name="planner_llm",
        harness_contract=HARNESS_CONTRACT,
        skill_validator=validate_skill,
        harness_validator=validator,
        role_protocol=VLABenchRoleProtocol(PROMPT_DIR),
        provenance={
            "implementation_scope": "SHAPER method implementation with a benchmark-specific interface and prompt pack",
            "benchmark": "VLABench",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "openpi_repository": OPENPI_REPOSITORY,
            "openpi_commit": OPENPI_COMMIT,
            "checkpoint_repository": CHECKPOINT_REPOSITORY,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
            "split_status": "reported 15-episode optimization and 24-episode fixed-validation protocol",
            "track": track_name,
            "train_task_ids": task_ids(train),
            "validation_task_ids": task_ids(validation),
            "actor": {
                "type": "frozen OpenPI websocket policy",
                "replan_steps": runtime.vla_replan_steps,
                "inference_timeout_seconds": runtime.vla_inference_timeout_seconds,
                "observation_schema": runtime.observation_schema,
                "actor_id": runtime.expected_actor_id,
                "policy_config": runtime.expected_policy_config,
            },
            "reward": {
                "optimization": "official VLABench task progress",
                "terminal_success": "official timestep.last() or progress >= 1.0",
            },
            "prompt_pack": "contrib/recipes/shaper/vlabench/prompts",
        },
    )
