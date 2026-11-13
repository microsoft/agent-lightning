# Copyright (c) Microsoft. All rights reserved.

"""Rollout lifecycle hooks used by enqueue and fit flows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from agentlightning.schemas import RolloutCreate

if TYPE_CHECKING:
    from agentlightning.schemas import Rollout


class TraceWriter(Protocol):
    def add_event(self, rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> Any: ...


class RolloutHooks:
    """Base class for synchronous rollout lifecycle hooks."""

    def on_startup(self, store: Any | None = None) -> None:
        """Initialize hook state once after startup."""

    def on_enqueue(self, request: RolloutCreate) -> RolloutCreate:
        """Transform a rollout request before it is persisted."""
        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: TraceWriter) -> None:
        """Run after a rollout transitions to SUCCEEDED."""

    def on_failed(self, rollout: Rollout, store: TraceWriter) -> None:
        """Run after a rollout transitions to FAILED."""


def load_hooks(path: str) -> RolloutHooks:
    """Load the single ``RolloutHooks`` subclass from a Python file."""
    import hashlib
    import importlib.util
    import inspect
    import sys
    from pathlib import Path

    module_path = Path(path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Hooks module not found: {module_path}")

    module_name = "_agl_hooks_" + hashlib.sha256(str(module_path).encode()).hexdigest()
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)

        hook_classes = [
            obj
            for _, obj in inspect.getmembers(module, inspect.isclass)
            if issubclass(obj, RolloutHooks) and obj is not RolloutHooks
        ]

        if len(hook_classes) == 0:
            raise ValueError(f"No RolloutHooks subclass found in {path}")
        if len(hook_classes) > 1:
            names = [cls.__name__ for cls in hook_classes]
            raise ValueError(f"Multiple RolloutHooks subclasses found in {path}: {names}")

        return hook_classes[0]()
    except BaseException:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
