# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Iterable


def main(argv: Iterable[str] | None = None) -> int:
    from agentlightning.instrumentation.vllm import instrument_vllm
    from vllm.entrypoints.cli.main import main as vllm_main
    import sys

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
