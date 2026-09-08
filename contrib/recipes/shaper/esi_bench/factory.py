# Copyright (c) Microsoft. All rights reserved.

"""Environment-configured SHAPER bundle for official ESI-Bench rollouts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from agentlightning.types import LLM, PromptTemplate
from contrib.recipes.shaper.reproduce import ReproductionBundle

from ..common import load_text
from .agent import ESIBenchAgent, ESIBenchRuntimeConfig
from .check_env import absolute_executable, check_map_generation_patch, check_worker_environment
from .contracts import (
    BEHAVIOR_ASSET_VERSION,
    BEHAVIOR_COMMIT,
    BEHAVIOR_REPOSITORY,
    HARNESS_CONTRACT,
    OMNIGIBSON_ROBOT_ASSET_VERSION,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    check_behavior_source,
    check_upstream_source,
    make_harness_validator,
    validate_skill,
)
from .dataset import load_datasets, task_ids
from .roles import ESIBenchRoleProtocol

RECIPE_DIR = Path(__file__).parent
PROMPT_DIR = RECIPE_DIR / "prompts"
SPLIT_DIR = RECIPE_DIR / "splits"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Set {name} before building the ESI-Bench SHAPER bundle.")
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
    """Build a complete official-runner ESI-Bench training bundle."""

    root = Path(_required_env("ESI_BENCH_ROOT")).expanduser().resolve()
    behavior_root = Path(_required_env("ESI_BEHAVIOR_ROOT")).expanduser().resolve()
    worker_python = absolute_executable(Path(os.environ.get("ESI_WORKER_PYTHON", sys.executable)))
    source_errors = check_upstream_source(root)
    if source_errors:
        raise RuntimeError("Unsupported ESI-Bench checkout: " + "; ".join(source_errors))
    behavior_errors = [*check_behavior_source(behavior_root), *check_worker_environment(worker_python, behavior_root)]
    if behavior_errors:
        raise RuntimeError("Unsupported BEHAVIOR/OmniGibson environment: " + "; ".join(behavior_errors))
    map_errors = check_map_generation_patch(Path(_required_env("ESI_MAKE_MAPS_PATH")))
    if map_errors:
        raise RuntimeError("Unsupported OmniGibson map setup: " + "; ".join(map_errors))
    output_root = Path(os.environ.get("ESI_OUTPUT_ROOT", "outputs/shaper/esi_runner")).expanduser().resolve()
    raw_data_root = os.environ.get("ESI_OMNIGIBSON_DATA_ROOT") or os.environ.get("OMNIGIBSON_DATA_PATH")
    if not raw_data_root:
        raise ValueError("Set ESI_OMNIGIBSON_DATA_ROOT or OMNIGIBSON_DATA_PATH before building the bundle.")
    omnigibson_data_root = Path(raw_data_root).expanduser().resolve()
    questions_jsonl = (
        Path(os.environ.get("ESI_QUESTIONS_JSONL", str(root / "hf_dataset" / "data" / "questions.jsonl")))
        .expanduser()
        .resolve()
    )
    runtime = ESIBenchRuntimeConfig(
        esi_bench_root=root,
        behavior_root=behavior_root,
        questions_jsonl=questions_jsonl,
        output_root=output_root,
        omnigibson_data_root=omnigibson_data_root,
        worker_python=worker_python,
        max_steps=int(os.environ.get("ESI_MAX_STEPS", "30")),
        min_steps=int(os.environ.get("ESI_MIN_STEPS", "3")),
        confidence_threshold=float(os.environ.get("ESI_CONFIDENCE_THRESHOLD", "0.85")),
        max_new_tokens=int(os.environ.get("ESI_MAX_NEW_TOKENS", "32768")),
        temperature=float(os.environ.get("ESI_TEMPERATURE", "1.0")),
        top_p=float(os.environ.get("ESI_TOP_P", "0.95")),
        robot=os.environ.get("ESI_ROBOT", "R1"),
        episode_timeout_seconds=float(os.environ.get("ESI_EPISODE_TIMEOUT", "1800")),
        environment_retries=int(os.environ.get("ESI_ENVIRONMENT_RETRIES", "1")),
        harness_timeout_seconds=float(os.environ.get("SHAPER_HARNESS_TIMEOUT", "3")),
        harness_memory_limit_mb=int(os.environ.get("SHAPER_HARNESS_MEMORY_MB", "768")),
        harness_max_output_chars=int(os.environ.get("SHAPER_HARNESS_MAX_OUTPUT_CHARS", "24000000")),
    )
    train_split = Path(os.environ.get("ESI_TRAIN_SPLIT", str(SPLIT_DIR / "recipe_train10.txt"))).expanduser().resolve()
    validation_split = (
        Path(os.environ.get("ESI_VALIDATION_SPLIT", str(SPLIT_DIR / "recipe_validation10.txt"))).expanduser().resolve()
    )
    train, validation = load_datasets(
        questions_jsonl,
        train_split,
        validation_split,
        max_steps=runtime.max_steps,
        canonical_root=root / "dataset" / "json_clean",
    )
    validator = make_harness_validator(
        timeout_seconds=runtime.harness_timeout_seconds,
        memory_limit_mb=runtime.harness_memory_limit_mb,
        max_output_chars=runtime.harness_max_output_chars,
    )
    resources = {
        "planner_llm": _planner_resource(),
        "skill": PromptTemplate(template=load_text(PROMPT_DIR, "seed_skill.txt"), engine="f-string"),
        "harness": PromptTemplate(template=load_text(PROMPT_DIR, "seed_harness.py"), engine="f-string"),
    }
    return ReproductionBundle(
        agent=ESIBenchAgent(runtime),
        train_dataset=train,
        val_dataset=validation,
        initial_resources=resources,
        planner_resource_name="planner_llm",
        harness_contract=HARNESS_CONTRACT,
        skill_validator=validate_skill,
        harness_validator=validator,
        role_protocol=ESIBenchRoleProtocol(PROMPT_DIR),
        provenance={
            "implementation_scope": "SHAPER method implementation with a benchmark-specific interface and prompt pack",
            "benchmark": "ESI-Bench",
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "behavior_repository": BEHAVIOR_REPOSITORY,
            "behavior_commit": BEHAVIOR_COMMIT,
            "behavior_asset_version": BEHAVIOR_ASSET_VERSION,
            "omnigibson_robot_asset_version": OMNIGIBSON_ROBOT_ASSET_VERSION,
            "split_status": (
                "deterministic contrib 10-question optimization and 10-question fixed-validation recipe; "
                "not claimed to be the unavailable original experiment manifest"
            ),
            "train_task_ids": task_ids(train),
            "validation_task_ids": task_ids(validation),
            "runner": "official active_explore.pipeline.run_one in a fresh process per episode",
            "worker_python": str(worker_python),
            "harness_interception_scope": (
                "every frozen-planner user context passes through the selected harness: official primary "
                "collect_contents calls and the audited inclined-plane post_action_query call"
            ),
            "official_auxiliary_model_calls": (
                "the pinned runner's sole task-specific call is inclined-plane post-action analysis; "
                "its official prompt, frame order, schema, and per-call token limit remain authoritative"
            ),
            "reward": "official task scorer exact-match result",
            "observable_context": (
                "official RGB/reference images, pixel-only derivatives, visible action/reasoning/confidence, "
                "and sanitized official action results"
            ),
            "prompt_pack": "contrib/recipes/shaper/esi_bench/prompts",
        },
    )
