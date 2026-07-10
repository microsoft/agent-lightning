# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Iterable, cast


def main(argv: Iterable[str] | None = None) -> int:
    import sys

    from agentlightning.instrumentation.vllm import instrument_vllm

    vllm_main = cast(Callable[[], Any], getattr(import_module("vllm.entrypoints.cli.main"), "main"))
    instrument_vllm()
    if argv is not None:
        original_argv = sys.argv
        sys.argv = [original_argv[0], *list(argv)]
        try:
            vllm_main()
        finally:
            sys.argv = original_argv
    else:
        vllm_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
