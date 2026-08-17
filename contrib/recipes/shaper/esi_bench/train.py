# Copyright (c) Microsoft. All rights reserved.

"""Preflight and run SHAPER training on the included ESI-Bench adapter."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

from ..cli import cli_arguments, print_preflight_errors, requests_help, required_environment
from ..reproduce import main as reproduce_main
from .check_env import absolute_executable, check_environment

FACTORY = "contrib.recipes.shaper.esi_bench.factory:build_bundle"
RECIPE_DIR = Path(__file__).parent


def preflight_environment() -> list[str]:
    root = Path(required_environment("ESI_BENCH_ROOT")).expanduser().resolve()
    behavior_root = Path(required_environment("ESI_BEHAVIOR_ROOT")).expanduser().resolve()
    raw_data_root = os.environ.get("ESI_OMNIGIBSON_DATA_ROOT") or os.environ.get("OMNIGIBSON_DATA_PATH")
    if not raw_data_root:
        raise ValueError("Set ESI_OMNIGIBSON_DATA_ROOT or OMNIGIBSON_DATA_PATH before running this command.")
    omnigibson_data_root = Path(raw_data_root).expanduser().resolve()
    questions = (
        Path(os.environ.get("ESI_QUESTIONS_JSONL", root / "hf_dataset" / "data" / "questions.jsonl"))
        .expanduser()
        .resolve()
    )
    train_split = (
        Path(os.environ.get("ESI_TRAIN_SPLIT", RECIPE_DIR / "splits" / "recipe_train10.txt")).expanduser().resolve()
    )
    validation_split = (
        Path(os.environ.get("ESI_VALIDATION_SPLIT", RECIPE_DIR / "splits" / "recipe_validation10.txt"))
        .expanduser()
        .resolve()
    )
    make_maps_path = Path(required_environment("ESI_MAKE_MAPS_PATH")).expanduser().resolve()
    worker_python = absolute_executable(Path(os.environ.get("ESI_WORKER_PYTHON", sys.executable)))
    return check_environment(
        root=root,
        behavior_root=behavior_root,
        omnigibson_data_root=omnigibson_data_root,
        questions_jsonl=questions,
        train_split=train_split,
        validation_split=validation_split,
        make_maps_path=make_maps_path,
        worker_python=worker_python,
        planner_endpoint=required_environment("SHAPER_PLANNER_ENDPOINT"),
        require_planner=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = cli_arguments(argv)
    if requests_help(arguments):
        reproduce_main(["--factory", FACTORY, *arguments])
        return 0
    if print_preflight_errors(preflight_environment()):
        return 2
    reproduce_main(["--factory", FACTORY, *arguments])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
