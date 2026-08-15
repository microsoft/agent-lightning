# Copyright (c) Microsoft. All rights reserved.

"""Evaluate a SHAPER artifact pair on VLABench."""

from __future__ import annotations

from typing import Sequence

from ..cli import cli_arguments, print_preflight_errors, requests_help
from ..evaluate import main as evaluate_main
from .train import FACTORY, preflight_environment


def main(argv: Sequence[str] | None = None) -> int:
    arguments = cli_arguments(argv)
    if requests_help(arguments):
        return evaluate_main(["--factory", FACTORY, *arguments])
    if print_preflight_errors(preflight_environment()):
        return 2
    return evaluate_main(["--factory", FACTORY, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
