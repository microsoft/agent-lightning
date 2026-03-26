"""Rollout lifecycle hooks — task-specific logic injected into the store.

Hooks run synchronously inside store methods. Since the store is single-threaded
(plain ``def``, called from ``async def`` route handlers on one event loop),
hooks execute atomically — no reader can see intermediate state.

Users subclass ``RolloutHooks`` and override the methods they need. The server
loads the module at startup via ``--hooks path/to/hooks.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agl_lite.schemas.api import EnqueueRolloutRequest

if TYPE_CHECKING:
    from agl_lite.schemas.rollout import Rollout
    from agl_lite.store.memory import InMemoryStore


class RolloutHooks:
    """Base class for rollout lifecycle hooks.

    All methods have default no-op implementations so subclasses only need
    to override what they care about.

    Constraints:
      - Hooks must be **fast and synchronous** (no ``await``, no blocking I/O).
      - Volume reads (local disk, ~μs) and CPU-bound parsing (~ms) are fine.
      - Network calls to external APIs will block the event loop — avoid them.
    """

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        """Pre-processor: transform a rollout request before it enters the store.

        Called for each request in ``enqueue_rollouts()``, **before** the rollout
        is persisted. If this raises, the rollout is never created and the API
        returns an error to the caller.

        Typical uses:
          - Move raw dataset fields from ``input`` into ``metadata``
          - Set ``config.image`` based on task type (e.g., SWE-bench instance → Docker image)
          - Generate eval scripts and inject them as environment variables
          - Prepare ``input`` so it contains only what the agent needs
        """
        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: InMemoryStore) -> None:
        """Post-transition hook: called when a rollout transitions to SUCCEEDED.

        Runs synchronously inside ``update_rollout()``, after the transition is
        committed. Since the store is single-threaded, no reader can interleave —
        the transition and this hook are atomic from any external observer.

        ``events`` is the raw events dict for this rollout:
        ``{attempt_id: [Event, ...]}``.

        Typical uses:
          - Read test output from a shared volume
          - Grade using official evaluation tools (e.g., ``swebench.harness.grading``)
          - Post a reward event via ``store.add_event(rollout.rollout_id, ...)``
        """

    def on_failed(self, rollout: Rollout, store: InMemoryStore) -> None:
        """Post-transition hook: called when a rollout transitions to TERMINAL_FAILED.

        Typical uses:
          - Post a zero reward event
          - Log failure details
        """


def load_hooks(path: str) -> RolloutHooks:
    """Load a ``RolloutHooks`` subclass from a Python file.

    The module must define exactly one subclass of ``RolloutHooks``.
    It is instantiated with no arguments.

    Args:
        path: Filesystem path to the Python module (e.g., ``/app/hooks/hooks.py``).

    Returns:
        An instance of the ``RolloutHooks`` subclass found in the module.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the module contains zero or multiple ``RolloutHooks`` subclasses.
    """
    import importlib.util
    import inspect
    from pathlib import Path

    module_path = Path(path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Hooks module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("_agl_hooks", str(module_path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find all RolloutHooks subclasses (excluding the base class itself).
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
