# Copyright (c) Microsoft. All rights reserved.

"""Preflight and run SHAPER training on the included VLABench adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from ..cli import cli_arguments, print_preflight_errors, requests_help, required_environment
from ..reproduce import main as reproduce_main
from .check_env import check_environment
from .dataset import TRACK_NAME
from .openpi_identity import REPORTED_THREE_CAMERA

FACTORY = "contrib.recipes.shaper.vlabench.factory:build_bundle"


def preflight_environment() -> list[str]:
    """Validate the pinned VLABench source, actor identity, and planner."""

    root = Path(required_environment("VLABENCH_ROOT")).expanduser().resolve()
    host = os.environ.get("VLABENCH_VLA_HOST", "127.0.0.1")
    port = int(os.environ.get("VLABENCH_VLA_PORT", "8000"))
    return check_environment(
        root=root,
        track_name=os.environ.get("VLABENCH_TRACK", TRACK_NAME),
        host=host,
        port=port,
        require_vla=True,
        expected_actor_id=required_environment("VLABENCH_ACTOR_ID"),
        expected_policy_config=required_environment("VLABENCH_OPENPI_POLICY_CONFIG"),
        expected_observation_schema=os.environ.get("VLABENCH_OBSERVATION_SCHEMA", REPORTED_THREE_CAMERA),
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
