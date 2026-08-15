# Copyright (c) Microsoft. All rights reserved.

"""VLABench-specific artifact contracts shared by optimization and rollout."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from agentlightning.contrib.shaper import PythonHarnessValidator

from ..common import check_python_api, git_revision, git_tracked_changes, validate_multimodal_harness_output
from .actor_contract import OPENPI_COMMIT, OPENPI_REPOSITORY, POLICY_CONFIG

UPSTREAM_REPOSITORY = "https://github.com/OpenMOSS/VLABench"
UPSTREAM_COMMIT = "cf588fe60c0c7282174fe979f5913170cfe69017"

_OPENPI_OBSERVATION_KEYS = frozenset(
    {
        "observation/image",
        "observation/second_image",
        "observation/wrist_image",
    }
)


HARNESS_CONTRACT = """Define exactly `def build_context(history)`.
history is a JSON list containing only observable VLABench planner/VLA records:
round_index, task_instruction, planner_response, command, execution_steps,
observation_before and observation_after (OpenAI text/image_url parts),
observable action_result, and runtime_errors. Return a JSON-serializable string
or bounded text/image_url list. Do not access files, network, simulator state,
depth, segmentation, poses, object metadata, rewards, ground truth, or task
IDs."""


def check_upstream_source(vlabench_root: Path) -> list[str]:
    """Validate the source revision and lightweight API used by this adapter."""

    errors: list[str] = []
    revision = git_revision(vlabench_root)
    if revision is None:
        errors.append(f"VLABench source is not inside a readable Git checkout: {vlabench_root}")
    elif revision != UPSTREAM_COMMIT:
        errors.append(f"VLABench revision {revision} does not match pinned {UPSTREAM_COMMIT}.")
    changes = git_tracked_changes(vlabench_root)
    if changes:
        errors.append("VLABench checkout has tracked modifications: " + ", ".join(changes) + ".")
    errors.extend(
        check_python_api(
            vlabench_root / "envs" / "__init__.py",
            functions={
                "load_env": {
                    "task",
                    "episode_config",
                    "random_init",
                    "reset_wait_step",
                }
            },
        )
    )
    return errors


def check_openpi_source(openpi_root: Path) -> list[str]:
    """Validate the separately deployed frozen VLA actor implementation."""

    errors: list[str] = []
    revision = git_revision(openpi_root)
    if revision is None:
        errors.append(f"OpenPI source is not inside a readable Git checkout: {openpi_root}")
    elif revision != OPENPI_COMMIT:
        errors.append(f"OpenPI revision {revision} does not match pinned {OPENPI_COMMIT}.")
    changes = git_tracked_changes(openpi_root)
    if changes:
        errors.append("OpenPI checkout has tracked modifications: " + ", ".join(changes) + ".")
    errors.extend(
        check_python_api(
            openpi_root / "src" / "openpi" / "policies" / "policy_config.py",
            functions={
                "create_trained_policy": {
                    "train_config",
                    "checkpoint_dir",
                    "default_prompt",
                }
            },
        )
    )
    errors.extend(
        check_python_api(
            openpi_root / "src" / "openpi" / "serving" / "websocket_policy_server.py",
            class_methods={
                "WebsocketPolicyServer": {
                    "__init__": {"policy", "host", "port", "metadata"},
                    "serve_forever": set(),
                }
            },
        )
    )
    transform_path = openpi_root / "src" / "openpi" / "policies" / "vlabench_policy.py"
    try:
        transform_tree = ast.parse(transform_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        errors.append(f"Cannot parse pinned OpenPI VLABench transform {transform_path}: {exc}")
    else:
        input_class = next(
            (node for node in transform_tree.body if isinstance(node, ast.ClassDef) and node.name == "VLABenchInputs"),
            None,
        )
        call_method = (
            next(
                (
                    node
                    for node in input_class.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__call__"
                ),
                None,
            )
            if input_class is not None
            else None
        )
        consumed_keys: set[str] = set()
        if call_method is not None:
            for node in ast.walk(call_method):
                if (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "data"
                    and isinstance(node.slice, ast.Constant)
                    and isinstance(node.slice.value, str)
                ):
                    consumed_keys.add(node.slice.value)
        missing_keys = sorted(_OPENPI_OBSERVATION_KEYS - consumed_keys)
        if missing_keys:
            errors.append(
                "Pinned OpenPI VLABenchInputs.__call__ does not consume required observation keys: "
                + ", ".join(missing_keys)
                + "."
            )
    config_path = openpi_root / "src" / "openpi" / "training" / "config.py"
    try:
        config_source = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"Cannot read pinned OpenPI training config {config_path}: {exc}")
    else:
        if f'name="{POLICY_CONFIG}"' not in config_source:
            errors.append(f"Pinned OpenPI config {POLICY_CONFIG!r} is missing from {config_path}.")
    return errors


def validate_skill(source: str) -> list[str]:
    """Enforce the planner output contract consumed by the VLABench adapter."""

    errors: list[str] = []
    if not source.strip():
        errors.append("Skill must not be empty.")
    if "Answer:" not in source:
        errors.append("Skill must require an `Answer:` line.")
    if "Steps:" not in source:
        errors.append("Skill must require a `Steps:` line.")
    if len(source) > 20_000:
        errors.append("Skill must remain below 20,000 characters.")
    return errors


def _image(label: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{label}"},
    }


def _one_round_probe() -> tuple[list[dict[str, Any]]]:
    return (
        [
            {
                "round_index": 0,
                "task_instruction": "Take the red mug and place it in the tray.",
                "planner_response": "The mug is visible; approach it.",
                "command": "Please take the red mug.",
                "execution_steps": 48,
                "observation_before": [
                    {"type": "text", "text": "Main camera (third-person)"},
                    _image("MAIN_BEFORE_0"),
                    {"type": "text", "text": "Wrist camera (gripper)"},
                    _image("WRIST_BEFORE_0"),
                ],
                "observation_after": [
                    {"type": "text", "text": "Main camera (third-person)"},
                    _image("MAIN_AFTER_0"),
                    {"type": "text", "text": "Wrist camera (gripper)"},
                    _image("WRIST_AFTER_0"),
                ],
                "action_result": {"ik_success": True},
                "runtime_errors": [],
            }
        ],
    )


def _two_round_probe() -> tuple[list[dict[str, Any]]]:
    history = list(_one_round_probe()[0])
    history.append(
        {
            "round_index": 1,
            "task_instruction": "Take the red mug and place it in the tray.",
            "planner_response": "Contact is visible, but placement is incomplete.",
            "command": "Please put the red mug into the tray.",
            "execution_steps": 72,
            "observation_before": history[-1]["observation_after"],
            "observation_after": [
                {"type": "text", "text": "Main camera (third-person)"},
                _image("MAIN_AFTER_1"),
                {"type": "text", "text": "Wrist camera (gripper)"},
                _image("WRIST_AFTER_1"),
            ],
            "action_result": {"ik_success": False},
            "runtime_errors": ["IK solver failed once before recovery."],
        }
    )
    return (history,)


def make_harness_validator(
    *,
    timeout_seconds: float = 3.0,
    memory_limit_mb: int = 768,
    max_output_chars: int = 32_000_000,
) -> PythonHarnessValidator:
    """Build the validator used both when admitting and executing artifacts."""

    return PythonHarnessValidator(
        smoke_args=_one_round_probe(),
        additional_smoke_args=(([],), _two_round_probe()),
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
        output_validator=validate_multimodal_harness_output,
    )


__all__ = [
    "HARNESS_CONTRACT",
    "OPENPI_COMMIT",
    "OPENPI_REPOSITORY",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "check_openpi_source",
    "check_upstream_source",
    "make_harness_validator",
    "validate_skill",
]
