"""In-memory store — single-threaded, no locks, plain dict/list.

All methods are plain `def` (synchronous). Called from `async def` route
handlers on the event loop thread. See docs/dev_guidelines.md § Concurrency Model.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agl_lite.schemas.api import (
    ArchiveBackend,
    ArchiveResult,
    EnqueueRolloutRequest,
    PatchRolloutRequest,
    RegisterModelRequest,
)
from agl_lite.schemas.errors import InvalidTransitionError, NotFoundError
from agl_lite.schemas.event import Event
from agl_lite.schemas.model_server import ModelServer
from agl_lite.schemas.resources import ResourcesUpdate
from agl_lite.schemas.rollout import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    Rollout,
    RolloutConfig,
    RolloutMetadata,
    RolloutStatus,
)

if TYPE_CHECKING:
    from agl_lite.hooks import RolloutHooks

log = logging.getLogger(__name__)


class InMemoryStore:
    """In-memory implementation of the agl-lite Store.

    Data structures:
      rollouts:  dict[rollout_id, Rollout]
      events:    dict[rollout_id, dict[attempt_id, list[Event]]]  (nested)
      resources: dict[resources_id, ResourcesUpdate]
      models:    dict[model, dict[endpoint, ModelServer]]  (nested)
    """

    def __init__(self, hooks: RolloutHooks | None = None) -> None:
        self._rollouts: dict[str, Rollout] = {}
        self._events: dict[str, dict[str, list[Event]]] = {}
        self._resources: dict[str, ResourcesUpdate] = {}
        self._latest_resources_id: str | None = None
        self._models: dict[str, dict[str, ModelServer]] = {}
        self._hooks = hooks

    # ── Rollout management ───────────────────────────────────────────

    def enqueue_rollouts(self, requests: list[EnqueueRolloutRequest]) -> list[Rollout]:
        """Create new rollouts in QUEUING status."""
        results: list[Rollout] = []
        for req in requests:
            # Pre-processor hook: transform request before persist.
            if self._hooks:
                req = self._hooks.on_enqueue(req)

            now = time.time()
            rollout_id = uuid.uuid4().hex
            rollout = Rollout(
                rollout_id=rollout_id,
                input=req.input,
                config=req.config or RolloutConfig(image=""),
                metadata=RolloutMetadata(**(req.metadata if isinstance(req.metadata, dict) else req.metadata.model_dump() if req.metadata else {})),
                resources_id=req.resources_id,
                created_at=now,
                updated_at=now,
            )
            self._rollouts[rollout_id] = rollout
            self._events[rollout_id] = {}
            results.append(rollout)
        return results

    def get_rollout(self, rollout_id: str) -> Rollout:
        """Get a single rollout by ID. Raises NotFoundError."""
        try:
            return self._rollouts[rollout_id]
        except KeyError:
            raise NotFoundError("Rollout", rollout_id) from None

    def rollout_exists(self, rollout_id: str) -> bool:
        """Check if a rollout exists. Used by gateway for fast validation (~100ns)."""
        return rollout_id in self._rollouts

    def update_rollout(self, rollout_id: str, req: PatchRolloutRequest) -> Rollout:
        """Partial update of a rollout. Only fields explicitly set in req are applied.

        If `status` is being changed, validates the state transition.
        Bumps `version` and `updated_at` on every successful update.

        Raises:
            NotFoundError: rollout doesn't exist
            InvalidTransitionError: illegal state transition
        """
        rollout = self.get_rollout(rollout_id)
        updates = req.model_dump(exclude_unset=True)

        if not updates:
            return rollout  # no-op

        # Validate state transition if status is changing.
        if "status" in updates:
            new_status = updates["status"]
            if new_status not in VALID_TRANSITIONS[rollout.status]:
                raise InvalidTransitionError(rollout_id, rollout.status, new_status)

        # Apply update.
        now = time.time()
        updated = rollout.model_copy(
            update={
                **updates,
                "version": rollout.version + 1,
                "updated_at": now,
            }
        )
        self._rollouts[rollout_id] = updated

        # Post-transition hooks — still inside the sync method,
        # no reader can interleave (single-threaded event loop).
        if self._hooks and "status" in updates and updated.status != rollout.status:
            try:
                if updated.status == RolloutStatus.SUCCEEDED:
                    events = self._events.get(rollout_id, {})
                    self._hooks.on_succeeded(updated, events, self)
                elif updated.status == RolloutStatus.TERMINAL_FAILED:
                    self._hooks.on_failed(updated, self)
            except Exception:
                log.exception("Hook error for rollout %s", rollout_id)

        return updated

    def cancel_rollout(self, rollout_id: str) -> Rollout:
        """Set cancel_requested=True. Rejects if already terminal.

        Raises:
            NotFoundError: rollout doesn't exist
            InvalidTransitionError: rollout is in a terminal state
        """
        rollout = self.get_rollout(rollout_id)

        if rollout.status in TERMINAL_STATUSES:
            raise InvalidTransitionError(rollout_id, rollout.status, "cancel_requested")

        if rollout.cancel_requested:
            return rollout  # idempotent

        now = time.time()
        updated = rollout.model_copy(
            update={
                "cancel_requested": True,
                "version": rollout.version + 1,
                "updated_at": now,
            }
        )
        self._rollouts[rollout_id] = updated
        return updated

    def query_rollouts(
        self,
        *,
        ids: list[str] | None = None,
        status_in: list[RolloutStatus] | None = None,
        cancel_requested: bool | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Rollout]:
        """Query rollouts with optional filters.

        When `ids` is provided, returns exactly those rollouts (batch fetch).
        Other filters combine with `ids` or work standalone.
        """
        if ids is not None:
            candidates = [self._rollouts[rid] for rid in ids if rid in self._rollouts]
        else:
            candidates = list(self._rollouts.values())

        # Apply filters.
        if status_in is not None:
            status_set = set(status_in)
            candidates = [r for r in candidates if r.status in status_set]
        if cancel_requested is not None:
            candidates = [r for r in candidates if r.cancel_requested == cancel_requested]

        # Pagination.
        candidates = candidates[offset:]
        if limit is not None:
            candidates = candidates[:limit]

        return candidates

    # ── Event storage ────────────────────────────────────────────────

    def add_event(self, rollout_id: str, attempt_id: str, event_type: str, data: dict[str, Any]) -> Event:
        """Append a single event. Raises NotFoundError if rollout doesn't exist."""
        if rollout_id not in self._rollouts:
            raise NotFoundError("Rollout", rollout_id)

        event = Event(
            event_type=event_type,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            timestamp=time.time(),
            data=data,
        )

        rid_events = self._events[rollout_id]
        if attempt_id not in rid_events:
            rid_events[attempt_id] = []
        rid_events[attempt_id].append(event)
        return event



    def query_events(
        self,
        rollout_id: str,
        *,
        attempt_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Event]:
        """Query events for a rollout.

        Smart attempt_id resolution when omitted:
          1. If rollout.succeeded_attempt_id is set → use it
          2. Otherwise → attempt with latest first event timestamp
          3. No events → return []
        """
        if rollout_id not in self._rollouts:
            raise NotFoundError("Rollout", rollout_id)

        rid_events = self._events.get(rollout_id, {})

        # Resolve attempt_id.
        if attempt_id is None:
            attempt_id = self._resolve_attempt_id(rollout_id, rid_events)
            if attempt_id is None:
                return []

        events = rid_events.get(attempt_id, [])

        # Filter by event_type.
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        # Pagination.
        events = events[offset:]
        if limit is not None:
            events = events[:limit]

        return events

    def list_attempts(self, rollout_id: str) -> list[str]:
        """Return attempt IDs for a rollout, ordered by first event timestamp (earliest first)."""
        if rollout_id not in self._rollouts:
            raise NotFoundError("Rollout", rollout_id)

        rid_events = self._events.get(rollout_id, {})
        if not rid_events:
            return []

        # Sort by first event timestamp.
        attempts = sorted(
            rid_events.keys(),
            key=lambda aid: rid_events[aid][0].timestamp if rid_events[aid] else float("inf"),
        )
        return attempts

    def _resolve_attempt_id(self, rollout_id: str, rid_events: dict[str, list[Event]]) -> str | None:
        """Smart attempt_id resolution.

        1. succeeded_attempt_id if set
        2. attempt with latest first-event timestamp (most recent attempt)
        3. None if no events
        """
        rollout = self._rollouts[rollout_id]

        # 1. Succeeded attempt.
        if rollout.succeeded_attempt_id and rollout.succeeded_attempt_id in rid_events:
            return rollout.succeeded_attempt_id

        # 2. Latest attempt (by first event timestamp — most recently started).
        if not rid_events:
            return None

        return max(
            rid_events.keys(),
            key=lambda aid: rid_events[aid][0].timestamp if rid_events[aid] else 0.0,
        )

    # ── Resource management ──────────────────────────────────────────

    def add_resources(self, resources: dict[str, Any]) -> ResourcesUpdate:
        """Add a new immutable resource snapshot."""
        resources_id = uuid.uuid4().hex
        now = time.time()
        update = ResourcesUpdate(
            resources_id=resources_id,
            resources=resources,
            created_at=now,
        )
        self._resources[resources_id] = update
        self._latest_resources_id = resources_id
        return update

    def get_resources(self, resources_id: str) -> ResourcesUpdate:
        """Get a resource snapshot by ID. Raises NotFoundError."""
        try:
            return self._resources[resources_id]
        except KeyError:
            raise NotFoundError("ResourcesUpdate", resources_id) from None

    def get_latest_resources(self) -> ResourcesUpdate | None:
        """Get the most recently added resource snapshot, or None."""
        if self._latest_resources_id is None:
            return None
        return self._resources[self._latest_resources_id]

    # ── Model server management ──────────────────────────────────────

    def register_models(self, requests: list[RegisterModelRequest]) -> list[ModelServer]:
        """Register (or update) model inference servers. Upsert by (model, endpoint)."""
        results: list[ModelServer] = []
        for req in requests:
            now = time.time()
            server = ModelServer(
                model=req.model,
                endpoint=req.endpoint,
                version=req.version,
                token=req.token,
                created_at=now,
            )
            if req.model not in self._models:
                self._models[req.model] = {}
            self._models[req.model][req.endpoint] = server
            results.append(server)
        return results

    def list_models(self) -> list[ModelServer]:
        """List all registered model servers (flat list)."""
        return [server for pool in self._models.values() for server in pool.values()]

    def get_model_pool(self, model: str) -> list[ModelServer]:
        """Get all servers for a model. Returns empty list if model not found."""
        pool = self._models.get(model, {})
        return list(pool.values())

    def remove_model_servers(self, model: str, endpoints: list[str] | None = None) -> None:
        """Remove servers for a model.

        If endpoints is None or empty, remove the entire model pool.
        Otherwise, remove only the specified endpoints.
        Auto-deletes the model entry if the pool becomes empty.
        Raises NotFoundError if model not found.
        """
        if model not in self._models:
            raise NotFoundError("Model", model)

        if not endpoints:
            # Remove entire pool.
            del self._models[model]
            return

        pool = self._models[model]
        for ep in endpoints:
            pool.pop(ep, None)  # silently skip missing endpoints

        # Auto-delete empty pool.
        if not pool:
            del self._models[model]

    def remove_all_models(self) -> None:
        """Remove all model servers. Gateway enters unavailable state (503)."""
        self._models.clear()

    # ── Data lifecycle (archive + purge) ─────────────────────────────

    def archive_rollouts(self, rollout_ids: list[str], backend: ArchiveBackend | None = None) -> ArchiveResult:
        """Archive and purge rollouts from hot store.

        1. Reject if any rollout is non-terminal (ValueError)
        2. If backend specified: persist to JSONL file (append if exists, create if not)
        3. Purge rollout records and all events from hot store
        """
        # Validate all rollouts exist and are terminal.
        rollouts_to_archive: list[Rollout] = []
        for rid in rollout_ids:
            rollout = self.get_rollout(rid)  # raises NotFoundError
            if rollout.status not in TERMINAL_STATUSES:
                raise ValueError(f"Cannot archive non-terminal rollout {rid} (status={rollout.status})")
            rollouts_to_archive.append(rollout)

        # Persist if backend specified.
        path: str | None = None
        if backend is not None:
            path = backend.path
            self._write_jsonl(rollouts_to_archive, Path(path))

        # Purge from hot store.
        for rid in rollout_ids:
            del self._rollouts[rid]
            self._events.pop(rid, None)

        return ArchiveResult(archived=len(rollouts_to_archive), purged=len(rollouts_to_archive), path=path)

    def _write_jsonl(self, rollouts: list[Rollout], path: Path) -> None:
        """Write rollouts + events + resources to JSONL file. Append if exists."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for rollout in rollouts:
                # Collect all events for this rollout (all attempts).
                all_events: list[Event] = []
                for attempt_events in self._events.get(rollout.rollout_id, {}).values():
                    all_events.extend(attempt_events)
                all_events.sort(key=lambda e: e.timestamp)

                # Include referenced resources snapshot.
                resources_data = None
                if rollout.resources_id and rollout.resources_id in self._resources:
                    resources_data = self._resources[rollout.resources_id].model_dump()

                line = {
                    "rollout": rollout.model_dump(),
                    "events": [e.model_dump() for e in all_events],
                    "resources": resources_data,
                }
                f.write(json.dumps(line, default=str) + "\n")
