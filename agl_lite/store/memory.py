"""In-memory store — single-threaded, no locks, plain dict/list.

All methods are plain `def` (synchronous). Called from `async def` route
handlers on the event loop thread. See docs/dev_guidelines.md § Concurrency Model.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from agl_lite.schemas.api import ArchiveBackend, ArchiveResult
from agl_lite.schemas.errors import ConflictError, InvalidTransitionError, NotFoundError
from agl_lite.schemas.event import Event
from agl_lite.schemas.model_server import ModelServer
from agl_lite.schemas.resources import ResourcesUpdate
from agl_lite.schemas.rollout import (
    TERMINAL_STATUSES,
    VALID_TRANSITIONS,
    Rollout,
    RolloutConfig,
    RolloutStatus,
)


class InMemoryStore:
    """In-memory implementation of the agl-lite Store.

    Data structures:
      rollouts:  dict[rollout_id, Rollout]
      events:    dict[rollout_id, dict[attempt_id, list[Event]]]  (nested)
      resources: dict[resources_id, ResourcesUpdate]
      models:    dict[endpoint, ModelServer]
    """

    def __init__(self) -> None:
        self._rollouts: dict[str, Rollout] = {}
        self._events: dict[str, dict[str, list[Event]]] = {}
        self._resources: dict[str, ResourcesUpdate] = {}
        self._latest_resources_id: str | None = None
        self._models: dict[str, ModelServer] = {}

    # ── Rollout management ───────────────────────────────────────────

    def enqueue_rollout(
        self,
        input: dict[str, Any],
        config: RolloutConfig,
        resources_id: str | None = None,
    ) -> Rollout:
        """Create a new rollout in QUEUING status."""
        now = time.time()
        rollout_id = uuid.uuid4().hex
        rollout = Rollout(
            rollout_id=rollout_id,
            input=input,
            config=config,
            resources_id=resources_id,
            created_at=now,
            updated_at=now,
        )
        self._rollouts[rollout_id] = rollout
        self._events[rollout_id] = {}
        return rollout

    def get_rollout(self, rollout_id: str) -> Rollout:
        """Get a single rollout by ID. Raises NotFoundError."""
        try:
            return self._rollouts[rollout_id]
        except KeyError:
            raise NotFoundError("Rollout", rollout_id) from None

    def rollout_exists(self, rollout_id: str) -> bool:
        """Check if a rollout exists. Used by gateway for fast validation (~100ns)."""
        return rollout_id in self._rollouts

    def update_rollout(
        self,
        rollout_id: str,
        status: RolloutStatus,
        expected_version: int,
        *,
        job_name: str | None = None,
        succeeded_attempt_id: str | None = None,
        error_message: str | None = None,
    ) -> Rollout:
        """Update rollout status with optimistic locking and transition validation.

        Raises:
            NotFoundError: rollout doesn't exist
            ConflictError: version mismatch
            InvalidTransitionError: illegal state transition
        """
        rollout = self.get_rollout(rollout_id)

        # Optimistic locking.
        if rollout.version != expected_version:
            raise ConflictError("Rollout", rollout_id, expected_version, rollout.version)

        # State transition validation.
        if status not in VALID_TRANSITIONS[rollout.status]:
            raise InvalidTransitionError(rollout_id, rollout.status, status)

        # Apply update.
        now = time.time()
        updated = rollout.model_copy(
            update={
                "status": status,
                "version": rollout.version + 1,
                "updated_at": now,
                **({"job_name": job_name} if job_name is not None else {}),
                **({"succeeded_attempt_id": succeeded_attempt_id} if succeeded_attempt_id is not None else {}),
                **({"error_message": error_message} if error_message is not None else {}),
            }
        )
        self._rollouts[rollout_id] = updated
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

    def add_events(self, events: list[dict[str, Any]]) -> list[Event]:
        """Append multiple events. Each dict must have rollout_id, attempt_id, event_type, data."""
        return [
            self.add_event(
                rollout_id=e["rollout_id"],
                attempt_id=e["attempt_id"],
                event_type=e["event_type"],
                data=e["data"],
            )
            for e in events
        ]

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

    def register_model(self, endpoint: str, version: int = 0) -> ModelServer:
        """Register (or update) a model inference server. Keyed by endpoint — upsert semantics."""
        now = time.time()
        model = ModelServer(
            endpoint=endpoint,
            version=version,
            created_at=now,
        )
        self._models[endpoint] = model
        return model

    def register_models(self, models: list[dict[str, Any]]) -> list[ModelServer]:
        """Register multiple model servers."""
        return [self.register_model(endpoint=m["endpoint"], version=m.get("version", 0)) for m in models]

    def list_models(self) -> list[ModelServer]:
        """List all registered model servers."""
        return list(self._models.values())

    def remove_model(self, endpoint: str) -> None:
        """Remove a single model server by endpoint. Raises NotFoundError."""
        try:
            del self._models[endpoint]
        except KeyError:
            raise NotFoundError("ModelServer", endpoint) from None

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
