"""Rollout lifecycle hooks — task-specific logic used by enqueue/fit flows.

Hooks are synchronous extension points used by the algorithm path (for example,
VERL fit/bridge) to transform enqueue requests and to post rewards after
terminal rollout states.

Users subclass ``RolloutHooks`` and override the methods they need. The server
loads the module at startup via ``--hooks path/to/hooks.py``.

Typical pattern::

    class MyHooks(RolloutHooks):
        # on_startup is optional — if AGL_POD_SPEC_TEMPLATE is set in the
        # environment the base implementation loads the pod spec automatically.
        # Override only when you need extra setup beyond file loading.

        def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
            pod_spec = self.copy_pod_spec()  # deep copy of the loaded template
            agent = self.get_container(pod_spec, "agent")
            agent["image"] = f"my-image:{request.input['version']}"
            request.config.pod_spec = pod_spec
            return request
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import yaml

from agl_lite.schemas.api import EnqueueRolloutRequest

if TYPE_CHECKING:
    from agl_lite.schemas.rollout import Rollout
    from agl_lite.store.memory import InMemoryStore


class TraceWriter(Protocol):
    def add_event(self, rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> Any: ...


class RolloutHooks:
    """Base class for rollout lifecycle hooks.

    All methods have default no-op implementations so subclasses only need
    to override what they care about.

    Constraints:
      - Hooks must be **fast and synchronous** (no ``await``, no blocking I/O).
      - Volume reads (local disk, ~μs) and CPU-bound parsing (~ms) are fine.
      - Network calls to external APIs will block the event loop — avoid them.

    Pod spec template convention:
      Set ``self._pod_spec`` in ``on_startup()`` and use ``copy_pod_spec()``
      in ``on_enqueue()`` to get a per-request mutable copy.
    """

    _pod_spec: dict[str, Any] | None = None

    def on_startup(self, store: InMemoryStore) -> None:
        """Called once by the server after startup and store initialisation.

        The base implementation reads ``AGL_POD_SPEC_TEMPLATE`` from the
        environment and, if set, loads that YAML file into ``self._pod_spec``.
        This covers the common case where all instances use the same container
        image base and only differ in per-sample env vars.

        ``AGL_POD_SPEC_TEMPLATE`` — path to a plain YAML file that describes the
        pod spec fragment: ``containers``, optional ``volumes``, ``nodeSelector``,
        ``tolerations``, ``activeDeadlineSeconds``, etc.  Typically
        ``examples/<project>/job-template.yaml``.

        Override only when you need setup beyond file loading, e.g. loading a
        dataset index or connecting to an external registry.  When overriding,
        call ``super().on_startup(store)`` first so the base pod spec load still
        happens::

            def on_startup(self, store: InMemoryStore) -> None:
                super().on_startup(store)      # loads AGL_POD_SPEC_TEMPLATE
                self._index = load_index(os.environ["MY_INDEX"])
        """
        template_path = os.environ.get("AGL_POD_SPEC_TEMPLATE")
        if template_path:
            self._pod_spec = yaml.safe_load(Path(template_path).read_text())

    def copy_pod_spec(self) -> dict[str, Any]:
        """Return a deep copy of the stored pod spec template.

        Call this in ``on_enqueue`` to get a per-request mutable copy.
        Raises ``RuntimeError`` if ``self._pod_spec`` has not been set in ``on_startup``.
        """
        if self._pod_spec is None:
            raise RuntimeError("no pod spec loaded — set self._pod_spec in on_startup()")
        return copy.deepcopy(self._pod_spec)

    @staticmethod
    def get_container(pod_spec: dict[str, Any], name: str) -> dict[str, Any]:
        """Return a container dict by name from a pod spec dict.

        Raises ``KeyError`` if no container with that name exists.
        """
        for c in pod_spec.get("containers", []):
            if c.get("name") == name:
                return c
        raise KeyError(f"container {name!r} not found in pod spec")

    def on_enqueue(self, request: EnqueueRolloutRequest) -> EnqueueRolloutRequest:
        """Pre-processor: transform a rollout request before it enters the store.

        Called for each request in ``enqueue_rollouts()``, **before** the rollout
        is persisted. If this raises, the rollout is never created and the API
        returns an error to the caller.

        Typical uses:
          - Deep copy ``self._pod_spec``, apply per-sample modifications
            (image, env vars, command), assign to ``request.config.pod_spec``
          - Move raw dataset fields from ``input`` into ``metadata``
          - Set ``config.timeout`` / ``config.max_retries``
        """
        return request

    def on_succeeded(self, rollout: Rollout, events: dict[str, list[Any]], store: TraceWriter) -> None:
        """Post-transition hook: called when a rollout transitions to SUCCEEDED.

        ``events`` is the raw events dict for this rollout: ``{attempt_id: [Event, ...]}``.

        Typical uses:
          - Grade output using official evaluation tools
          - Post a reward event via ``store.add_event(rollout.rollout_id, ...)``
        """

    def on_failed(self, rollout: Rollout, store: TraceWriter) -> None:
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
