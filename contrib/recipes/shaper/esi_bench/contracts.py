# Copyright (c) Microsoft. All rights reserved.

"""ESI-Bench-specific artifact contracts shared by optimization and rollout."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentlightning.contrib.shaper import PythonHarnessValidator

from ..common import (
    check_python_api,
    git_head_file,
    git_revision,
    git_tracked_changes,
    validate_multimodal_harness_output,
)

UPSTREAM_REPOSITORY = "https://github.com/ESI-Bench/ESI-Bench"
UPSTREAM_COMMIT = "3c1756396f32b1a90c1f72356a7fde45f418e179"
BEHAVIOR_REPOSITORY = "https://github.com/StanfordVL/BEHAVIOR-1K"
BEHAVIOR_COMMIT = "67ad490856dd465d4606663106f81673fc8bf4e8"
BEHAVIOR_MAP_PATH = "asset_pipeline/b1k_pipeline/usd_conversion/make_maps.py"
BEHAVIOR_ASSET_VERSION = "3.9.0rc7"
OMNIGIBSON_ROBOT_ASSET_VERSION = "3.8.2"

_ORIGINAL_MAP_SETTING = "NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES + WALL_CATEGORIES"
_ESI_MAP_SETTING = "NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES"
_GENERIC_JSON_INSTRUCTION = "Return exactly one valid JSON object and nothing else."
_PIPELINE_PATH = Path("src/active_explore/pipeline.py")
_INCLINED_PLANE_PATH = Path("src/active_explore/tasks/physical_dynamics/inclined_plane.py")


HARNESS_CONTRACT = """Define exactly `def build_context(records)`.
records is a bounded JSON list containing official observable ESI-Bench data.
Past records contain record_kind, step, visible action/answer/confidence/
reasoning, sanitized action_result and its official JSON text rendering, an
official RGB full_frame, pixel size/quality, an 8x8 grayscale visual signature,
deterministic pixel-only focus_crops, an optional GRID1000 overlay selected
from the visible task prompt, and optional official extra RGBs. The final
current record also contains max_steps, remaining_steps, the official task
prompt, current RGB derivatives, and official question reference RGB
derivatives. For the pinned inclined-plane post-action call, the final current
record instead has call_kind=auxiliary_post_action and an ordered
observable_sequence of official RGB observations and visible text; prior past
records remain available. Return a bounded JSON-serializable string or list of
OpenAI text/image_url parts. The adapter separately supplies the selected skill
and authoritative task prompt; do not duplicate either. Do not access files,
network, simulator state, camera poses, object metadata, AABBs, depth,
segmentation, rewards, ground-truth answers, task IDs, or mutable external
state."""


def _parse_source(path: Path) -> tuple[ast.Module | None, list[str]]:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), []
    except (OSError, SyntaxError) as exc:
        return None, [f"Cannot inspect ESI-Bench model-call contract in {path}: {exc}"]


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _generate_json_calls(node: ast.AST) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Attribute)
        and candidate.func.attr == "generate_json"
    ]


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _is_call_to(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def check_model_call_contract(esi_bench_root: Path) -> list[str]:
    """Require every official model call to follow one audited observable path."""

    errors: list[str] = []
    pipeline_path = esi_bench_root / _PIPELINE_PATH
    pipeline_tree, parse_errors = _parse_source(pipeline_path)
    errors.extend(parse_errors)
    if pipeline_tree is None:
        return errors

    pipeline_calls = _generate_json_calls(pipeline_tree)
    if len(pipeline_calls) != 2:
        errors.append(
            f"Pinned ESI-Bench pipeline must contain exactly two model generate_json calls; found {len(pipeline_calls)}."
        )
    for function_name in ("force_final_choice", "run_one"):
        function = _top_level_function(pipeline_tree, function_name)
        if function is None:
            errors.append(f"Pinned ESI-Bench pipeline is missing {function_name}().")
            continue
        calls = _generate_json_calls(function)
        if len(calls) != 1:
            errors.append(
                f"ESI-Bench {function_name}() must contain exactly one generate_json call; found {len(calls)}."
            )
            continue
        call = calls[0]
        if not _is_call_to(_keyword(call, "contents"), "collect_contents"):
            errors.append(
                f"ESI-Bench {function_name}() must route model contents through the replaceable collect_contents harness."
            )
        system_instruction = _keyword(call, "system_instruction")
        if not (
            isinstance(system_instruction, ast.Constant)
            and isinstance(system_instruction.value, str)
            and system_instruction.value == _GENERIC_JSON_INSTRUCTION
        ):
            errors.append(
                f"ESI-Bench {function_name}() must keep the audited generic system instruction so the task skill "
                "is injected exactly once by collect_contents."
            )

    task_root = esi_bench_root / "src" / "active_explore" / "tasks"
    task_calls: list[tuple[Path, ast.Module, list[ast.Call]]] = []
    if not task_root.is_dir():
        errors.append(f"Missing official ESI-Bench task source directory: {task_root}")
        return errors
    for task_path in sorted(task_root.rglob("*.py")):
        task_tree, task_parse_errors = _parse_source(task_path)
        errors.extend(task_parse_errors)
        if task_tree is None:
            continue
        calls = _generate_json_calls(task_tree)
        if calls:
            task_calls.append((task_path.relative_to(esi_bench_root), task_tree, calls))

    if len(task_calls) != 1 or task_calls[0][0] != _INCLINED_PLANE_PATH or len(task_calls[0][2]) != 1:
        locations = [f"{path} ({len(calls)} call(s))" for path, _tree, calls in task_calls]
        errors.append(
            "ESI-Bench task modules may contain only the audited inclined-plane post-action model call; found: "
            + (", ".join(locations) if locations else "none")
            + "."
        )
        return errors

    inclined_path, inclined_tree, inclined_calls = task_calls[0]
    post_action = _top_level_function(inclined_tree, "post_action_query")
    if post_action is None or inclined_calls[0] not in _generate_json_calls(post_action):
        errors.append(f"{inclined_path} must keep its sole generate_json call inside post_action_query().")
        return errors
    call = inclined_calls[0]
    contents = _keyword(call, "contents")
    if not isinstance(contents, ast.Name) or contents.id != "contents":
        errors.append("Inclined-plane post_action_query() must send its official observable frame contents.")
    if not _is_call_to(_keyword(call, "system_instruction"), "build_system_prompt"):
        errors.append("Inclined-plane post_action_query() must use its official task-specific system prompt.")
    return errors


def check_upstream_source(esi_bench_root: Path) -> list[str]:
    """Validate the pinned official runner contract without importing Isaac."""

    errors: list[str] = []
    revision = git_revision(esi_bench_root)
    if revision is None:
        errors.append(f"ESI-Bench source is not inside a readable Git checkout: {esi_bench_root}")
    elif revision != UPSTREAM_COMMIT:
        errors.append(f"ESI-Bench revision {revision} does not match pinned {UPSTREAM_COMMIT}.")
    changes = git_tracked_changes(esi_bench_root)
    if changes:
        errors.append("ESI-Bench checkout has tracked modifications: " + ", ".join(changes) + ".")
    errors.extend(
        check_python_api(
            esi_bench_root / _PIPELINE_PATH,
            functions={
                "build_model_client": {"provider", "api_key", "model"},
                "collect_contents": {
                    "image_path",
                    "history",
                    "prompt",
                    "reference_image_paths",
                    "reference_image_path",
                },
                "force_final_choice": {
                    "task_module",
                    "model_client",
                    "payload",
                    "camera_info",
                    "image_path",
                    "history",
                    "config",
                    "task_state",
                    "reference_image_paths",
                },
                "run_one": {"config"},
            },
            annotated_classes={
                "ActiveExploreConfig": {
                    "task",
                    "metadata",
                    "question_index",
                    "json_root",
                    "results_root",
                    "step_image_root",
                    "provider",
                    "model",
                    "api_key",
                    "max_steps",
                    "min_steps",
                    "threshold",
                    "max_new_tokens",
                    "temperature",
                    "top_p",
                    "robot",
                    "overwrite",
                }
            },
        )
    )
    errors.extend(check_model_call_contract(esi_bench_root))
    return errors


def check_behavior_source(behavior_root: Path) -> list[str]:
    """Validate the contrib deployment pin for BEHAVIOR/OmniGibson."""

    errors: list[str] = []
    revision = git_revision(behavior_root)
    if revision is None:
        errors.append(f"BEHAVIOR-1K source is not inside a readable Git checkout: {behavior_root}")
    elif revision != BEHAVIOR_COMMIT:
        errors.append(f"BEHAVIOR-1K revision {revision} does not match pinned {BEHAVIOR_COMMIT}.")
    changes = git_tracked_changes(behavior_root)
    unexpected = sorted(set(changes or ()) - {BEHAVIOR_MAP_PATH})
    if unexpected:
        errors.append(
            "BEHAVIOR-1K checkout has tracked modifications beyond the official map setting: "
            + ", ".join(unexpected)
            + "."
        )
    if changes and BEHAVIOR_MAP_PATH in changes:
        baseline = git_head_file(behavior_root, BEHAVIOR_MAP_PATH)
        map_path = behavior_root / BEHAVIOR_MAP_PATH
        try:
            current = map_path.read_text(encoding="utf-8")
        except OSError as exc:
            errors.append(f"Cannot read patched BEHAVIOR map generator {map_path}: {exc}")
        else:
            if baseline is None or baseline.count(_ORIGINAL_MAP_SETTING) != 1:
                errors.append("Cannot reconstruct the expected ESI map setting from pinned BEHAVIOR HEAD.")
            else:
                expected = baseline.replace(_ORIGINAL_MAP_SETTING, _ESI_MAP_SETTING, 1)
                if current != expected:
                    errors.append(
                        "BEHAVIOR map generator differs from pinned HEAD beyond the one ESI-required "
                        "FLOOR_CATEGORIES setting."
                    )
    required = (
        behavior_root / "setup.sh",
        behavior_root / "OmniGibson" / "omnigibson" / "__init__.py",
        behavior_root / BEHAVIOR_MAP_PATH,
    )
    errors.extend(f"Missing pinned BEHAVIOR-1K source path: {path}" for path in required if not path.is_file())
    return errors


def check_omnigibson_install(behavior_root: Path) -> list[str]:
    """Require the active environment to import OmniGibson from the pin."""

    try:
        spec = importlib.util.find_spec("omnigibson")
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return [f"Cannot inspect the omnigibson installation: {exc}"]
    if spec is None or spec.origin is None:
        return ["Python package omnigibson is not importable."]
    origin = Path(spec.origin).resolve()
    expected = (behavior_root / "OmniGibson").resolve()
    try:
        origin.relative_to(expected)
    except ValueError:
        return [f"omnigibson resolves to {origin}, outside pinned source {expected}."]
    return []


def validate_skill(source: str) -> list[str]:
    """Preserve the official prompt inside a plain planner-policy artifact."""

    errors: list[str] = []
    count = source.count("{task_prompt}")
    if count != 1:
        errors.append(f"Skill must contain {{task_prompt}} exactly once; found {count}.")
    if len(source) > 30_000:
        errors.append("Skill must remain below 30,000 characters.")
    code_prefixes = ("import ", "from ", "def ", "class ", "async def ")
    if any(line.lstrip().startswith(code_prefixes) for line in source.splitlines()):
        errors.append("Skill must be planner instructions, not Python source.")
    lowered = source.lower()
    if any(marker in lowered for marker in ("etl_definition", "create_etl_artifact", "artifact[")):
        errors.append("Skill must not contain artifact-framework or ETL wrappers.")
    if "\x00" in source:
        errors.append("Skill must not contain NUL bytes.")
    return errors


def _image(label: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{label}"},
    }


def _observation(label: str, *, include_grid: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "source": "official_rgb",
        "pixel_size": {"width": 640, "height": 480},
        "pixel_quality": {
            "contrast": 0.6,
            "edge_density": 0.15,
            "laplacian_sharpness": 0.25,
            "score": 1.4375,
        },
        "visual_signature": {
            "kind": "8x8_grayscale",
            "values": [index * 4 for index in range(64)],
        },
        "full_frame": _image(f"{label}_FULL"),
        "focus_crops": [
            {
                "source": "deterministic_pixel_crop",
                "region": "horizontal_band_3",
                "quality": {
                    "contrast": 0.4,
                    "edge_density": 0.2,
                    "laplacian_sharpness": 0.1,
                    "score": 1.25,
                },
                "image": _image(f"{label}_CENTER"),
            },
            {
                "source": "deterministic_pixel_crop",
                "region": "overlapping_tile_7",
                "quality": {
                    "contrast": 0.5,
                    "edge_density": 0.1,
                    "laplacian_sharpness": 0.2,
                    "score": 1.0,
                },
                "image": _image(f"{label}_LOWER"),
            },
        ],
    }
    if include_grid:
        value["grid_overlay"] = {
            "source": "deterministic_pixel_overlay",
            "coordinate_system": "GRID1000: x=0..1000 left-to-right, y=0..1000 top-to-bottom",
            "image": _image(f"{label}_GRID1000"),
        }
    return value


def _one_step_probe() -> tuple[list[dict[str, Any]]]:
    return (
        [
            {
                "record_kind": "past",
                "step": 1,
                "action": "turn_left",
                "answer": "not sure",
                "confidence": 0.1,
                "reasoning": "Seek another viewpoint.",
                "action_result": {"handled": True, "operation": "camera"},
                "action_result_text": '{"handled": true, "operation": "camera"}',
                "observation": _observation("PAST_1"),
                "extra_observations": [],
            },
            {
                "record_kind": "current",
                "step": 2,
                "max_steps": 30,
                "remaining_steps": 29,
                "task_instruction": "Which candidate matches the reflected object?",
                "observation": _observation("CURRENT_2"),
                "reference_observations": [
                    {
                        "label": "QUESTION REFERENCE IMAGE 1",
                        **_observation("REFERENCE_1"),
                    }
                ],
            },
        ],
    )


def _two_step_probe() -> tuple[list[dict[str, Any]]]:
    records = list(_one_step_probe()[0][:-1])
    records.append(
        {
            "record_kind": "past",
            "step": 2,
            "action": "move_closer",
            "answer": "not sure",
            "confidence": 0.4,
            "reasoning": "Compare the candidates with retained evidence.",
            "action_result": {"handled": True, "operation": "navigation"},
            "action_result_text": '{"handled": true, "operation": "navigation"}',
            "observation": _observation("PAST_2"),
            "extra_observations": [_observation("EXTRA_2")],
        }
    )
    records.append(
        {
            "record_kind": "current",
            "step": 3,
            "max_steps": 30,
            "remaining_steps": 28,
            "task_instruction": "Which candidate matches the reflected object?",
            "observation": _observation("CURRENT_3"),
            "reference_observations": [],
        }
    )
    return (records,)


def _geometry_probe() -> tuple[list[dict[str, Any]]]:
    records = list(_one_step_probe()[0][:-1])
    records.append(
        {
            "record_kind": "current",
            "step": 2,
            "max_steps": 30,
            "remaining_steps": 29,
            "task_instruction": "Do the three objects form a straight line?",
            "observation": _observation("GEOMETRY_CURRENT", include_grid=True),
            "reference_observations": [],
        }
    )
    return (records,)


def _auxiliary_probe() -> tuple[list[dict[str, Any]]]:
    records = list(_one_step_probe()[0][:-1])
    records.append(
        {
            "record_kind": "current",
            "call_kind": "auxiliary_post_action",
            "step": 2,
            "max_steps": 30,
            "remaining_steps": 29,
            "task_instruction": "Estimate the visible inclined-plane outcome from the frame sequence.",
            "observable_sequence": [
                {
                    "content_kind": "observation",
                    "sequence_index": 0,
                    "observation": _observation("AUXILIARY_FRAME_1"),
                },
                {
                    "content_kind": "observation",
                    "sequence_index": 1,
                    "observation": _observation("AUXILIARY_FRAME_2"),
                },
                {
                    "content_kind": "text",
                    "sequence_index": 2,
                    "text": "Use only the visible frame sequence.",
                },
            ],
            "reference_observations": [],
        }
    )
    return (records,)


def make_harness_validator(
    *,
    timeout_seconds: float = 3.0,
    memory_limit_mb: int = 768,
    max_output_chars: int = 24_000_000,
) -> PythonHarnessValidator:
    """Build the validator used both when admitting and executing artifacts."""

    from agentlightning.contrib.shaper import PythonHarnessValidator

    return PythonHarnessValidator(
        smoke_args=_one_step_probe(),
        additional_smoke_args=(([],), _two_step_probe(), _geometry_probe(), _auxiliary_probe()),
        timeout_seconds=timeout_seconds,
        memory_limit_mb=memory_limit_mb,
        max_output_chars=max_output_chars,
        output_validator=validate_multimodal_harness_output,
    )


__all__ = [
    "BEHAVIOR_ASSET_VERSION",
    "BEHAVIOR_COMMIT",
    "BEHAVIOR_MAP_PATH",
    "BEHAVIOR_REPOSITORY",
    "HARNESS_CONTRACT",
    "OMNIGIBSON_ROBOT_ASSET_VERSION",
    "UPSTREAM_COMMIT",
    "UPSTREAM_REPOSITORY",
    "check_behavior_source",
    "check_model_call_contract",
    "check_omnigibson_install",
    "check_upstream_source",
    "make_harness_validator",
    "validate_skill",
]
