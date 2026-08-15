# Copyright (c) Microsoft. All rights reserved.

"""Static ESI-Bench/OmniGibson checks that do not initialize a simulator."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Sequence, cast

from ..cli import endpoint_socket
from .contracts import (
    BEHAVIOR_ASSET_VERSION,
    OMNIGIBSON_ROBOT_ASSET_VERSION,
    check_behavior_source,
    check_upstream_source,
)
from .dataset import load_datasets

RECIPE_DIR = Path(__file__).parent

_RUNTIME_MODULES = {
    "cv2": "opencv-python",
    "google.genai": "google-genai",
    "numpy": "numpy",
    "openai": "openai",
    "scipy": "scipy",
    "torch": "torch",
    "yaml": "PyYAML",
}


def absolute_executable(path: Path) -> Path:
    """Make an interpreter path absolute without resolving its venv symlink."""

    return Path(os.path.abspath(str(path.expanduser())))


def check_runtime_modules() -> list[str]:
    """Check modules imported eagerly by the pinned official pipeline."""

    errors: list[str] = []
    for module, package in _RUNTIME_MODULES.items():
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ModuleNotFoundError, ValueError):
            spec = None
        if spec is None:
            errors.append(f"Python module {module} is not importable; install {package} in the behavior environment.")
    return errors


def check_worker_environment(worker_python: Path, behavior_root: Path) -> list[str]:
    """Validate imports in the isolated Isaac/OmniGibson worker interpreter."""

    executable = absolute_executable(worker_python)
    if not executable.is_file():
        return [f"ESI worker Python does not exist: {executable}"]
    script = """
import importlib
import importlib.util
import json
import pathlib
import sys

behavior_root = pathlib.Path(sys.argv[1]).resolve()
modules = {
    "cv2": "opencv-python",
    "google.genai": "google-genai",
    "numpy": "numpy",
    "openai": "openai",
    "scipy": "scipy",
    "torch": "torch",
    "yaml": "PyYAML",
}
errors = []
for module, package in modules.items():
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        errors.append(f"Cannot inspect {module}: {exc}")
        continue
    if spec is None:
        errors.append(f"Python module {module} is not importable; install {package} in the worker environment.")

try:
    spec = importlib.util.find_spec("omnigibson")
except (ImportError, ModuleNotFoundError, ValueError) as exc:
    spec = None
    errors.append(f"Cannot inspect the omnigibson installation: {exc}")
if spec is None or spec.origin is None:
    errors.append("Python package omnigibson is not importable in the worker environment.")
else:
    origin = pathlib.Path(spec.origin).resolve()
    expected = (behavior_root / "OmniGibson").resolve()
    try:
        origin.relative_to(expected)
    except ValueError:
        errors.append(f"omnigibson resolves to {origin}, outside pinned source {expected}.")

try:
    importlib.import_module("contrib.recipes.shaper.esi_bench.worker")
except Exception as exc:
    errors.append(f"SHAPER ESI worker bridge is not importable: {type(exc).__name__}: {exc}")

print(json.dumps({
    "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    "errors": errors,
}))
"""
    try:
        environment = os.environ.copy()
        repository_root = Path(__file__).resolve().parents[4]
        existing_pythonpath = environment.get("PYTHONPATH", "")
        worker_paths = [str(repository_root), str(behavior_root / "OmniGibson")]
        if existing_pythonpath:
            worker_paths.append(existing_pythonpath)
        environment["PYTHONPATH"] = os.pathsep.join(worker_paths)
        result = subprocess.run(
            [str(executable), "-c", script, str(behavior_root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"Cannot run ESI worker Python {executable}: {exc}"]
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return [f"ESI worker Python preflight failed at {executable}: {detail}"]
    try:
        payload_value: object = json.loads(result.stdout)
    except json.JSONDecodeError:
        return [f"ESI worker Python returned invalid preflight output: {result.stdout!r}"]
    if not isinstance(payload_value, dict):
        return [f"ESI worker Python returned an invalid preflight payload: {payload_value!r}"]
    payload = cast(dict[str, object], payload_value)
    raw_errors = payload.get("errors", [])
    if not isinstance(raw_errors, list):
        return [f"ESI worker Python returned invalid errors: {raw_errors!r}"]
    errors = [str(value) for value in cast(list[object], raw_errors)]
    python_version = payload.get("python")
    if python_version != "3.11":
        errors.append(f"ESI worker requires Python 3.11; found {python_version!r} at {executable}.")
    return errors


def check_omnigibson_assets(data_root: Path) -> list[str]:
    """Check the datasets installed by the pinned BEHAVIOR setup."""

    required = (
        (data_root / "behavior-1k-assets", BEHAVIOR_ASSET_VERSION),
        (data_root / "omnigibson-robot-assets", OMNIGIBSON_ROBOT_ASSET_VERSION),
    )
    errors: list[str] = []
    for path, expected_version in required:
        if not path.is_dir():
            errors.append(f"Missing OmniGibson dataset payload: {path}")
            continue
        version_path = path / "VERSION"
        try:
            version = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            errors.append(f"Missing OmniGibson dataset version marker: {version_path}")
            continue
        if version != expected_version:
            errors.append(
                f"OmniGibson dataset {path.name} version {version!r} does not match pinned {expected_version!r}."
            )
    return errors


def check_map_generation_patch(path: Path) -> list[str]:
    """Verify the exact wall-removal setting required by ESI-Bench."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return [f"Missing OmniGibson map generator: {resolved}"]
    try:
        tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
    except (OSError, SyntaxError) as exc:
        return [f"Cannot inspect OmniGibson map generator {resolved}: {exc}"]
    assignments: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "NEEDED_STRUCTURE_CATEGORIES" for target in node.targets
        ):
            assignments.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "NEEDED_STRUCTURE_CATEGORIES"
            and node.value is not None
        ):
            assignments.append(node.value)
    if not assignments:
        return [f"{resolved} does not define NEEDED_STRUCTURE_CATEGORIES."]
    final = assignments[-1]
    if not isinstance(final, ast.Name) or final.id != "FLOOR_CATEGORIES":
        return [
            "ESI-Bench requires NEEDED_STRUCTURE_CATEGORIES = FLOOR_CATEGORIES in "
            f"{resolved}; the final assignment is {ast.unparse(final)!r}."
        ]
    return []


def _gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout.strip() else None


def check_environment(
    *,
    root: Path,
    behavior_root: Path,
    omnigibson_data_root: Path,
    questions_jsonl: Path,
    train_split: Path,
    validation_split: Path,
    make_maps_path: Path,
    worker_python: Path,
    planner_endpoint: str | None,
    require_planner: bool,
) -> list[str]:
    """Return missing or unsupported prerequisites without launching Isaac."""

    errors: list[str] = []
    errors.extend(check_upstream_source(root))
    errors.extend(check_behavior_source(behavior_root))
    errors.extend(check_worker_environment(worker_python, behavior_root))
    errors.extend(check_omnigibson_assets(omnigibson_data_root))
    errors.extend(check_map_generation_patch(make_maps_path))
    expected_map_path = behavior_root / "asset_pipeline" / "b1k_pipeline" / "usd_conversion" / "make_maps.py"
    if make_maps_path.resolve() != expected_map_path.resolve():
        errors.append(
            f"ESI_MAKE_MAPS_PATH must point into the pinned BEHAVIOR checkout: expected {expected_map_path}, "
            f"got {make_maps_path}."
        )
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        errors.append(f"ESI-Bench requires Linux x86_64; found {platform.system()} {platform.machine()}.")
    pipeline = root / "src" / "active_explore" / "pipeline.py"
    if not pipeline.is_file():
        errors.append(f"Missing official ESI-Bench runner: {pipeline}")
    gpu = _gpu_name()
    if gpu is None:
        errors.append("No NVIDIA GPU was reported by nvidia-smi.")
    elif any(marker in gpu.upper() for marker in ("RTX 50", "B100", "B200", "BLACKWELL")):
        errors.append(
            f"Official ESI-Bench documents poor rendering on 50-series/Blackwell GPUs; found {gpu}. Use a 20/30/40-series GPU."
        )
    try:
        train, validation = load_datasets(
            questions_jsonl,
            train_split,
            validation_split,
            canonical_root=root / "dataset" / "json_clean",
        )
        if not train or not validation:
            errors.append("ESI-Bench train and validation splits must both be non-empty.")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
    if planner_endpoint:
        endpoint_host, endpoint_port = endpoint_socket(planner_endpoint)
        if endpoint_host is None or endpoint_port is None:
            errors.append(f"SHAPER planner endpoint is not a valid HTTP(S) URL: {planner_endpoint!r}")
        else:
            try:
                with socket.create_connection((endpoint_host, endpoint_port), timeout=2.0):
                    pass
            except OSError as exc:
                errors.append(f"Planner endpoint {endpoint_host}:{endpoint_port} is unreachable: {exc}")
    elif require_planner:
        errors.append("Set SHAPER_PLANNER_ENDPOINT to an OpenAI-compatible chat-completions base URL.")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("ESI_BENCH_ROOT", ".")))
    parser.add_argument(
        "--behavior-root",
        type=Path,
        default=Path(os.environ.get("ESI_BEHAVIOR_ROOT", "missing-behavior-root")),
    )
    parser.add_argument(
        "--omnigibson-data-root",
        type=Path,
        default=Path(
            os.environ.get(
                "ESI_OMNIGIBSON_DATA_ROOT",
                os.environ.get("OMNIGIBSON_DATA_PATH", "missing-omnigibson-data"),
            )
        ),
    )
    parser.add_argument("--questions-jsonl", type=Path)
    parser.add_argument(
        "--train-split",
        type=Path,
        default=Path(os.environ.get("ESI_TRAIN_SPLIT", RECIPE_DIR / "splits" / "recipe_train10.txt")),
    )
    parser.add_argument(
        "--validation-split",
        type=Path,
        default=Path(os.environ.get("ESI_VALIDATION_SPLIT", RECIPE_DIR / "splits" / "recipe_validation10.txt")),
    )
    parser.add_argument("--planner-endpoint", default=os.environ.get("SHAPER_PLANNER_ENDPOINT"))
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=Path(os.environ.get("ESI_WORKER_PYTHON", sys.executable)),
    )
    parser.add_argument("--skip-planner-connect", action="store_true")
    parser.add_argument(
        "--make-maps-path",
        type=Path,
        default=Path(os.environ.get("ESI_MAKE_MAPS_PATH", "missing-make-maps.py")),
        help="Path to BEHAVIOR-1K asset_pipeline/b1k_pipeline/usd_conversion/make_maps.py.",
    )
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    questions = (
        args.questions_jsonl.expanduser().resolve()
        if args.questions_jsonl is not None
        else root / "hf_dataset" / "data" / "questions.jsonl"
    )
    errors = check_environment(
        root=root,
        behavior_root=args.behavior_root.expanduser().resolve(),
        omnigibson_data_root=args.omnigibson_data_root.expanduser().resolve(),
        questions_jsonl=questions,
        train_split=args.train_split.expanduser().resolve(),
        validation_split=args.validation_split.expanduser().resolve(),
        make_maps_path=args.make_maps_path.expanduser().resolve(),
        worker_python=absolute_executable(args.worker_python),
        planner_endpoint=str(args.planner_endpoint) if args.planner_endpoint else None,
        require_planner=not bool(args.skip_planner_connect),
    )
    if errors:
        for error in errors:
            print(f"[missing] {error}")
        return 2
    print("ESI-Bench SHAPER prerequisites passed without launching the simulator.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
