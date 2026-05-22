"""Per-rollout worker for local runner mode.

Invoked as: ``python -m agl_lite.controller.local_worker``

Reads everything from environment variables, loads the configured agent class,
calls run() once, then exits. The process exit code is the only signal the
LocalReconciler uses to decide the rollout's terminal state:
  - 0           → SUCCEEDED
  - non-zero    → TERMINAL_FAILED (or CANCELLED if cancel_requested was set)
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
import traceback


def _load_class(path: str) -> type:
    """Load ``pkg.mod:ClassName`` or ``pkg.mod.ClassName``."""
    if ":" in path:
        module_name, class_name = path.split(":", 1)
    else:
        module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    loaded = getattr(module, class_name)
    if not isinstance(loaded, type):
        raise TypeError(f"{path} is not a class")
    return loaded


def _run_maybe_async(result: object) -> None:
    if inspect.isawaitable(result):
        asyncio.run(result)  # type: ignore[arg-type]


def main() -> int:
    agent_class_path = os.environ["AGL_LOCAL_AGENT_CLASS"]
    task_input = json.loads(os.environ["AGL_TASK_INPUT"])

    try:
        agent_cls = _load_class(agent_class_path)
        agent = agent_cls()
        result = agent.run(task_input)
        _run_maybe_async(result)
        return 0
    except Exception:
        # stderr is inherited from the controller. Non-zero exit signals
        # TERMINAL_FAILED; LocalReconciler picks it up on the next tick.
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
