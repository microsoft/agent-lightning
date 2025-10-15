# Copyright (c) Microsoft. All rights reserved.

"""SQLite-backed LightningStore implementation with transactional safety."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pickle
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    RolloutConfig,
    RolloutStatus,
    RolloutV2,
    Span,
    TaskInput,
    TraceStatus,
)

from .base import UNSET, LightningStore, Unset, is_finished, is_queuing

logger = logging.getLogger(__name__)

T = TypeVar("T")


TCoroutine = TypeVar(
    "TCoroutine",
    bound=Callable[..., Coroutine[Any, Any, Any]],
)


def _healthcheck_wrapper(func: TCoroutine) -> TCoroutine:
    """Decorator ensuring a health check runs before ``func``.

    The health check keeps attempts and rollouts in sync. It is skipped when the
    decorated function is invoked from inside :meth:`SqliteLightningStore._healthcheck`
    to avoid recursion.
    """

    async def wrapper(self: "SqliteLightningStore", *args: Any, **kwargs: Any) -> Any:
        if getattr(self, "_healthcheck_running", False):
            return await func(self, *args, **kwargs)

        self._healthcheck_running = True  # type: ignore[assignment]
        try:
            await self._healthcheck()  # pyright: ignore[reportPrivateUsage]
        finally:
            self._healthcheck_running = False  # type: ignore[assignment]

        return await func(self, *args, **kwargs)

    return cast(TCoroutine, wrapper)


def _serialize_pickle(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_pickle(buffer: Optional[bytes]) -> Any:
    if buffer is None:
        return None
    return pickle.loads(buffer)


def _serialize_optional(value: Any) -> Optional[bytes]:
    if value is None:
        return None
    return _serialize_pickle(value)


def _json_encode(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"__type__": "bytes", "data": base64.b64encode(value).decode("ascii")}
    if hasattr(value, "model_dump"):
        return _json_encode(value.model_dump())
    if isinstance(value, dict):
        items = cast(Dict[Any, Any], value)
        return {str(key): _json_encode(item) for key, item in items.items()}
    if isinstance(value, (list, tuple, set)):
        iterable = cast(Iterable[Any], value)
        return [_json_encode(item) for item in iterable]
    return value


def _json_decode(value: Any) -> Any:
    if isinstance(value, dict):
        mapping = cast(Dict[str, Any], value)
        marker = mapping.get("__type__")
        if marker == "bytes":
            return base64.b64decode(cast(str, mapping["data"]))
        return {key: _json_decode(item) for key, item in mapping.items()}
    if isinstance(value, list):
        iterable = cast(List[Any], value)
        return [_json_decode(item) for item in iterable]
    return value


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_encode(value), separators=(",", ":"))


def _json_loads(value: Optional[str]) -> Any:
    if value is None:
        return None
    return _json_decode(json.loads(value))


@dataclass
class _SpanRow:
    rollout_id: str
    attempt_id: str
    sequence_id: int
    trace_id: str
    span_id: str
    parent_id: Optional[str]
    name: str
    status_json: str
    attributes_json: str
    events_json: str
    links_json: str
    start_time: Optional[float]
    end_time: Optional[float]
    context_json: Optional[str]
    parent_context_json: Optional[str]
    resource_json: str
    extras_json: Optional[str]


class SqliteLightningStore(LightningStore):
    """SQLite-backed persistent :class:`LightningStore` implementation.

    The store persists rollouts, attempts, and spans inside a single SQLite
    database. Connections are opened per thread/process and guarded with
    coarse-grained asyncio and threading locks so that callers can safely use
    the instance from async tasks, threads, or forked workers.

    The public APIs mirror the in-memory store but every write happens inside
    an explicit transaction (``BEGIN IMMEDIATE``) to ensure that related rows
    are committed atomically. Read operations are executed on background
    threads so that synchronous SQLite calls do not block the event loop.
    """

    def __init__(self, path: str = ":memory:") -> None:
        """Create a store backed by the SQLite database at ``path``.

        Args:
            path: Path to the SQLite database file. Passing ``":memory:"``
                creates an isolated shared-memory database, while using a file
                path persists state across restarts.
        """
        if path == ":memory":
            identifier = uuid.uuid4()
            self._database = f"file:agentlightning-{identifier}?mode=memory&cache=shared"
            self._uri = True
        else:
            self._database = path
            self._uri = path.startswith("file:")

        self._pid = os.getpid()
        self._thread_local = threading.local()
        self._connections: set[sqlite3.Connection] = set()
        self._connections_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._lock = asyncio.Lock()
        self._completion_events: Dict[str, asyncio.Event] = {}
        self._healthcheck_running = False
        self._latest_resources_id: Optional[str] = None

        self._initialize_schema()

    def close(self) -> None:
        """Close all SQLite connections owned by this store."""

        with self._connections_lock:
            for connection in list(self._connections):
                try:
                    connection.close()
                except Exception:  # pragma: no cover - best effort cleanup
                    logger.exception("Failed to close SQLite connection")
                finally:
                    self._connections.discard(connection)

    def _ensure_process(self) -> None:
        """Reset connection state if the store is accessed after a ``fork``.

        SQLite connections cannot be shared across processes. When a child
        process touches the store we drop any inherited connections so that the
        child opens its own handle lazily on demand.
        """
        if os.getpid() == self._pid:
            return
        # Reset state when accessed from a forked process.
        self._pid = os.getpid()
        self._thread_local = threading.local()
        with self._connections_lock:
            self._connections = set()

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection configured for concurrency.

        Connections are opened in autocommit mode so that we can explicitly
        manage transactions. ``check_same_thread=False`` is required because the
        same connection object might be used from different asyncio tasks in
        the same thread.
        """
        connection = sqlite3.connect(
            self._database,
            uri=self._uri,
            detect_types=sqlite3.PARSE_DECLTYPES,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        with self._connections_lock:
            self._connections.add(connection)
        return connection

    def _get_connection(self) -> sqlite3.Connection:
        """Return a cached per-thread connection, recreating it if needed."""
        self._ensure_process()
        connection = getattr(self._thread_local, "connection", None)
        if connection is None:
            connection = self._create_connection()
            self._thread_local.connection = connection
        else:
            try:
                connection.execute("SELECT 1")
            except sqlite3.ProgrammingError:
                connection = self._create_connection()
                self._thread_local.connection = connection
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Context manager that executes a single ``BEGIN IMMEDIATE`` block.

        ``BEGIN IMMEDIATE`` upgrades to a write transaction immediately and
        therefore serialises writes across threads/processes. The SQLite
        connection is protected with an ``RLock`` so nested write operations can
        reuse the same transaction safely.
        """
        connection = self._get_connection()
        with self._write_lock:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except Exception:
                if connection.in_transaction:
                    connection.rollback()
                raise
            else:
                if connection.in_transaction:
                    connection.commit()

    def _initialize_schema(self) -> None:
        with self._transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rollouts (
                    rollout_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    mode TEXT,
                    resources_id TEXT,
                    input_blob BLOB NOT NULL,
                    config_json TEXT NOT NULL,
                    metadata_blob BLOB,
                    kind TEXT NOT NULL DEFAULT 'rollout'
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    rollout_id TEXT NOT NULL,
                    sequence_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    worker_id TEXT,
                    last_heartbeat_time REAL,
                    metadata_blob BLOB,
                    span_counter INTEGER NOT NULL DEFAULT 0,
                    UNIQUE (rollout_id, sequence_id),
                    FOREIGN KEY (rollout_id) REFERENCES rollouts(rollout_id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS spans (
                    rollout_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    sequence_id INTEGER NOT NULL,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_id TEXT,
                    name TEXT NOT NULL,
                    status_json TEXT NOT NULL,
                    attributes_json TEXT NOT NULL,
                    events_json TEXT NOT NULL,
                    links_json TEXT NOT NULL,
                    start_time REAL,
                    end_time REAL,
                    context_json TEXT,
                    parent_context_json TEXT,
                    resource_json TEXT NOT NULL,
                    extras_json TEXT,
                    PRIMARY KEY (rollout_id, attempt_id, sequence_id),
                    FOREIGN KEY (rollout_id) REFERENCES rollouts(rollout_id) ON DELETE CASCADE,
                    FOREIGN KEY (attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
                );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO rollouts (
                    rollout_id, status, start_time, end_time, mode, resources_id,
                    input_blob, config_json, metadata_blob, kind
                ) VALUES (?, 'internal', 0.0, NULL, NULL, NULL, ?, ?, NULL, 'resources')
                """,
                (
                    "__resources__",
                    _serialize_pickle({}),
                    _json_dumps({"latest_id": None}),
                ),
            )

    async def _execute_read(self, func: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``func`` against the thread's connection on a worker thread."""

        def runner() -> T:
            connection = self._get_connection()
            return func(connection)

        return await asyncio.to_thread(runner)

    async def _execute_write(self, func: Callable[[sqlite3.Connection], T]) -> T:
        """Execute ``func`` inside a transactional worker-thread context."""

        def runner() -> T:
            with self._transaction() as connection:
                return func(connection)

        return await asyncio.to_thread(runner)

    @staticmethod
    def _row_to_rollout(row: sqlite3.Row) -> RolloutV2:
        rollout = RolloutV2(
            rollout_id=row["rollout_id"],
            input=_deserialize_pickle(row["input_blob"]),
            start_time=row["start_time"],
            end_time=row["end_time"],
            mode=row["mode"],
            resources_id=row["resources_id"],
            status=row["status"],
            config=RolloutConfig.model_validate(_json_loads(row["config_json"])),
            metadata=_deserialize_pickle(row["metadata_blob"]),
        )
        return rollout

    @staticmethod
    def _row_to_attempt(row: sqlite3.Row) -> Attempt:
        attempt = Attempt(
            rollout_id=row["rollout_id"],
            attempt_id=row["attempt_id"],
            sequence_id=row["sequence_id"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            status=row["status"],
            worker_id=row["worker_id"],
            last_heartbeat_time=row["last_heartbeat_time"],
            metadata=_deserialize_pickle(row["metadata_blob"]),
        )
        return attempt

    @staticmethod
    def _row_to_span(row: sqlite3.Row) -> Span:
        """Rehydrate a :class:`Span` from the stored SQLite row.

        Each span is stored with its OpenTelemetry fields JSON-encoded. This
        helper performs the decoding while preserving binary payloads and
        custom extras so the reconstructed object mirrors the original span
        created by the worker.
        """
        payload: Dict[str, Any] = {
            "rollout_id": row["rollout_id"],
            "attempt_id": row["attempt_id"],
            "sequence_id": row["sequence_id"],
            "trace_id": row["trace_id"],
            "span_id": row["span_id"],
            "parent_id": row["parent_id"],
            "name": row["name"],
            "status": _json_loads(row["status_json"]),
            "attributes": _json_loads(row["attributes_json"]),
            "events": _json_loads(row["events_json"]),
            "links": _json_loads(row["links_json"]),
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "context": _json_loads(row["context_json"]),
            "parent": _json_loads(row["parent_context_json"]),
            "resource": _json_loads(row["resource_json"]),
        }
        extras = cast(Optional[Dict[str, Any]], _json_loads(row["extras_json"]))
        if isinstance(extras, dict):
            payload.update(extras)
        payload["status"] = TraceStatus.model_validate(payload["status"])
        resource_payload = _json_loads(row["resource_json"])
        payload["resource"] = resource_payload
        span = Span.model_validate(payload)
        # Restore direct references to dicts for attributes/events/links so the
        # dataclasses remain mutable and efficient, matching the behaviour of
        # OpenTelemetry's native exporters.
        object.__setattr__(span, "attributes", _json_loads(row["attributes_json"]))

        raw_events_any = _json_loads(row["events_json"])
        if isinstance(raw_events_any, list):
            raw_events = cast(List[Any], raw_events_any)
            for event_obj, event_raw in zip(span.events, raw_events):
                if isinstance(event_raw, dict):
                    event_dict = cast(Dict[str, Any], event_raw)
                    attributes_raw = cast(Optional[Dict[str, Any]], event_dict.get("attributes"))
                    if attributes_raw is not None:
                        object.__setattr__(event_obj, "attributes", attributes_raw)

        raw_links_any = _json_loads(row["links_json"])
        if isinstance(raw_links_any, list):
            raw_links = cast(List[Any], raw_links_any)
            for link_obj, link_raw in zip(span.links, raw_links):
                if isinstance(link_raw, dict):
                    link_dict = cast(Dict[str, Any], link_raw)
                    attributes_raw = cast(Optional[Dict[str, Any]], link_dict.get("attributes"))
                    if attributes_raw is not None:
                        object.__setattr__(link_obj, "attributes", attributes_raw)

        if isinstance(resource_payload, dict) and "attributes" in resource_payload:
            object.__setattr__(span.resource, "attributes", resource_payload["attributes"])

        return span

    @staticmethod
    def _span_to_row(span: Span) -> _SpanRow:
        """Serialise ``span`` into the structured row representation."""
        span_dict = span.model_dump()
        base_fields = {
            "rollout_id",
            "attempt_id",
            "sequence_id",
            "trace_id",
            "span_id",
            "parent_id",
            "name",
            "status",
            "attributes",
            "events",
            "links",
            "start_time",
            "end_time",
            "context",
            "parent",
            "resource",
        }
        # Preserve any custom attributes stored on the span by including them in
        # ``extras_json``. When round-tripping we merge this payload back into
        # the reconstructed model.
        extras = {key: value for key, value in span_dict.items() if key not in base_fields}
        return _SpanRow(
            rollout_id=span.rollout_id,
            attempt_id=span.attempt_id,
            sequence_id=span.sequence_id,
            trace_id=span.trace_id,
            span_id=span.span_id,
            parent_id=span.parent_id,
            name=span.name,
            status_json=_json_dumps(span.status.model_dump()),
            attributes_json=_json_dumps(span.attributes),
            events_json=_json_dumps(
                [event.model_dump() if hasattr(event, "model_dump") else event for event in span.events]
            ),
            links_json=_json_dumps([link.model_dump() if hasattr(link, "model_dump") else link for link in span.links]),
            start_time=span.start_time,
            end_time=span.end_time,
            context_json=_json_dumps(span.context.model_dump()) if span.context is not None else None,
            parent_context_json=_json_dumps(span.parent.model_dump()) if span.parent is not None else None,
            resource_json=_json_dumps(span.resource.model_dump()),
            extras_json=_json_dumps(extras) if extras else None,
        )

    @staticmethod
    def _serialize_resources_update(update: ResourcesUpdate) -> bytes:
        """Convert a :class:`ResourcesUpdate` payload into bytes."""
        return _serialize_pickle(update.model_dump())

    @staticmethod
    def _deserialize_resources_update(buffer: bytes) -> ResourcesUpdate:
        """Convert the stored byte payload back into ``ResourcesUpdate``."""
        payload = _deserialize_pickle(buffer)
        return ResourcesUpdate.model_validate(payload)

    def _load_resources_store(self, connection: sqlite3.Connection) -> Tuple[Optional[str], Dict[str, ResourcesUpdate]]:
        """Return the persisted resources map and the latest snapshot id."""
        row = connection.execute(
            "SELECT input_blob, config_json FROM rollouts WHERE rollout_id=?",
            ("__resources__",),
        ).fetchone()
        if row is None:
            return None, {}
        serialized_map: Dict[str, bytes] = _deserialize_pickle(row["input_blob"]) or {}
        metadata = cast(Dict[str, Any], _json_loads(row["config_json"]) or {})
        latest_id = cast(Optional[str], metadata.get("latest_id"))
        # Each resources payload is stored individually so we only deserialize
        # the map entries that are required by callers.
        resources: Dict[str, ResourcesUpdate] = {}
        for resource_id, buffer in serialized_map.items():
            resources[resource_id] = self._deserialize_resources_update(buffer)
        return latest_id, resources

    def _persist_resources_store(
        self,
        connection: sqlite3.Connection,
        *,
        latest_id: Optional[str],
        resources: Dict[str, ResourcesUpdate],
    ) -> None:
        """Persist the resources store and update the latest snapshot pointer."""
        serialized_map = {
            resource_id: self._serialize_resources_update(update) for resource_id, update in resources.items()
        }
        connection.execute(
            "UPDATE rollouts SET input_blob=?, config_json=? WHERE rollout_id=?",
            (
                _serialize_pickle(serialized_map),
                _json_dumps({"latest_id": latest_id}),
                "__resources__",
            ),
        )

    def _set_completion_event(self, rollout_id: str) -> None:
        """Signal any waiters that ``rollout_id`` has finished processing."""
        event = self._completion_events.get(rollout_id)
        if event:
            event.set()

    @staticmethod
    def _ensure_rollout_row(row: Optional[sqlite3.Row], rollout_id: str) -> sqlite3.Row:
        """Ensure that a rollout row exists for ``rollout_id``."""
        if row is None:
            raise ValueError(f"Rollout {rollout_id} not found")
        return row

    @staticmethod
    def _ensure_attempt_row(row: Optional[sqlite3.Row], attempt_id: str) -> sqlite3.Row:
        """Ensure that an attempt row exists for ``attempt_id``."""
        if row is None:
            raise ValueError(f"Attempt {attempt_id} not found")
        return row

    def _get_rollout_row(self, connection: sqlite3.Connection, rollout_id: str) -> sqlite3.Row:
        """Fetch the rollout row and raise when no record is found."""
        row = connection.execute(
            "SELECT * FROM rollouts WHERE rollout_id=? AND kind='rollout'",
            (rollout_id,),
        ).fetchone()
        return self._ensure_rollout_row(row, rollout_id)

    def _get_rollout_row_optional(self, connection: sqlite3.Connection, rollout_id: str) -> Optional[sqlite3.Row]:
        """Fetch the rollout row returning ``None`` when it is absent."""
        return connection.execute(
            "SELECT * FROM rollouts WHERE rollout_id=? AND kind='rollout'",
            (rollout_id,),
        ).fetchone()

    def _get_attempt_row(self, connection: sqlite3.Connection, attempt_id: str) -> sqlite3.Row:
        """Fetch the attempt row and raise when the id is unknown."""
        row = connection.execute(
            "SELECT * FROM attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        return self._ensure_attempt_row(row, attempt_id)

    def _get_latest_attempt_row(self, connection: sqlite3.Connection, rollout_id: str) -> sqlite3.Row:
        """Return the newest attempt row associated with ``rollout_id``."""
        row = connection.execute(
            """
            SELECT * FROM attempts
            WHERE rollout_id=?
            ORDER BY sequence_id DESC
            LIMIT 1
            """,
            (rollout_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"No attempts found for rollout {rollout_id}")
        return row

    @_healthcheck_wrapper
    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        """Persist a rollout and immediately create its first attempt.

        Args:
            input: User supplied payload for the rollout.
            mode: Optional execution mode (``"train"``, ``"val"``, ``"test"``).
            resources_id: Identifier of the resources snapshot to bind to the
                rollout. If omitted, the latest known resources are used.
            metadata: Arbitrary JSON-serialisable metadata to persist alongside
                the rollout.

        Returns:
            AttemptedRollout: The created rollout including its first attempt
            in ``preparing`` state.
        """
        async with self._lock:
            rollout_id = f"rollout-{uuid.uuid4()}"
            attempt_id = f"attempt-{uuid.uuid4()}"
            now = time.time()
            if resources_id is None:
                resources_id = self._latest_resources_id

            rollout = RolloutV2(
                rollout_id=rollout_id,
                input=input,
                start_time=now,
                end_time=None,
                mode=mode,
                resources_id=resources_id,
                status="preparing",
                metadata=metadata,
            )
            attempt = Attempt(
                rollout_id=rollout_id,
                attempt_id=attempt_id,
                sequence_id=1,
                start_time=now,
                status="preparing",
            )

            def writer(connection: sqlite3.Connection) -> None:
                connection.execute(
                    """
                    INSERT INTO rollouts (
                        rollout_id, status, start_time, end_time, mode, resources_id,
                        input_blob, config_json, metadata_blob, kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'rollout')
                    """,
                    (
                        rollout.rollout_id,
                        rollout.status,
                        rollout.start_time,
                        rollout.end_time,
                        rollout.mode,
                        rollout.resources_id,
                        _serialize_pickle(rollout.input),
                        _json_dumps(rollout.config.model_dump()),
                        _serialize_optional(rollout.metadata),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO attempts (
                        attempt_id, rollout_id, sequence_id, status, start_time, end_time,
                        worker_id, last_heartbeat_time, metadata_blob, span_counter
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        attempt.attempt_id,
                        attempt.rollout_id,
                        attempt.sequence_id,
                        attempt.status,
                        attempt.start_time,
                        attempt.end_time,
                        attempt.worker_id,
                        attempt.last_heartbeat_time,
                        _serialize_optional(attempt.metadata),
                    ),
                )

            await self._execute_write(writer)
            self._completion_events.setdefault(rollout_id, asyncio.Event())
            return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

    @_healthcheck_wrapper
    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> RolloutV2:
        """Persist a rollout in the queue without creating an attempt.

        Args:
            input: User supplied payload for the rollout.
            mode: Optional execution mode value.
            resources_id: Identifier of the resources snapshot to bind to the
                rollout. Defaults to the latest snapshot if omitted.
            metadata: Additional metadata to persist.

        Returns:
            RolloutV2: The queued rollout in ``queuing`` state.
        """
        async with self._lock:
            rollout_id = f"rollout-{uuid.uuid4()}"
            now = time.time()
            if resources_id is None:
                resources_id = self._latest_resources_id

            rollout = RolloutV2(
                rollout_id=rollout_id,
                input=input,
                start_time=now,
                status="queuing",
                mode=mode,
                resources_id=resources_id,
                metadata=metadata,
            )

            def writer(connection: sqlite3.Connection) -> None:
                connection.execute(
                    """
                    INSERT INTO rollouts (
                        rollout_id, status, start_time, end_time, mode, resources_id,
                        input_blob, config_json, metadata_blob, kind
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'rollout')
                    """,
                    (
                        rollout.rollout_id,
                        rollout.status,
                        rollout.start_time,
                        rollout.end_time,
                        rollout.mode,
                        rollout.resources_id,
                        _serialize_pickle(rollout.input),
                        _json_dumps(rollout.config.model_dump()),
                        _serialize_optional(rollout.metadata),
                    ),
                )

            await self._execute_write(writer)
            self._completion_events.setdefault(rollout_id, asyncio.Event())
            return rollout

    @_healthcheck_wrapper
    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """Pop the next queued rollout and create a new attempt for it.

        Returns:
            Optional[AttemptedRollout]: ``None`` if the queue is empty, otherwise
            the dequeued rollout and its freshly created attempt.
        """
        async with self._lock:

            def writer(connection: sqlite3.Connection) -> Optional[AttemptedRollout]:
                row = connection.execute(
                    """
                    SELECT * FROM rollouts
                    WHERE kind='rollout' AND status IN ('queuing', 'requeuing')
                    ORDER BY start_time ASC
                    LIMIT 1
                    """,
                ).fetchone()
                if row is None:
                    return None
                rollout = self._row_to_rollout(row)
                rollout.status = "preparing"
                rollout.end_time = None

                sequence_row = connection.execute(
                    "SELECT COALESCE(MAX(sequence_id), 0) FROM attempts WHERE rollout_id=?",
                    (rollout.rollout_id,),
                ).fetchone()
                next_sequence = int(sequence_row[0]) + 1
                attempt = Attempt(
                    rollout_id=rollout.rollout_id,
                    attempt_id=f"attempt-{uuid.uuid4()}",
                    sequence_id=next_sequence,
                    start_time=time.time(),
                    status="preparing",
                )

                connection.execute(
                    "UPDATE rollouts SET status=?, end_time=? WHERE rollout_id=?",
                    (rollout.status, rollout.end_time, rollout.rollout_id),
                )
                connection.execute(
                    """
                    INSERT INTO attempts (
                        attempt_id, rollout_id, sequence_id, status, start_time, end_time,
                        worker_id, last_heartbeat_time, metadata_blob, span_counter
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        attempt.attempt_id,
                        attempt.rollout_id,
                        attempt.sequence_id,
                        attempt.status,
                        attempt.start_time,
                        attempt.end_time,
                        attempt.worker_id,
                        attempt.last_heartbeat_time,
                        _serialize_optional(attempt.metadata),
                    ),
                )
                return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

            return await self._execute_write(writer)

    @_healthcheck_wrapper
    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """Create a new attempt for ``rollout_id`` and mark it ``preparing``.

        Args:
            rollout_id: Identifier of the rollout receiving the attempt.

        Returns:
            AttemptedRollout: The rollout along with the newly inserted attempt.
        """
        async with self._lock:

            def writer(connection: sqlite3.Connection) -> AttemptedRollout:
                rollout_row = self._get_rollout_row(connection, rollout_id)
                rollout = self._row_to_rollout(rollout_row)

                latest_sequence_row = connection.execute(
                    "SELECT COALESCE(MAX(sequence_id), 0) FROM attempts WHERE rollout_id=?",
                    (rollout_id,),
                ).fetchone()
                next_sequence = int(latest_sequence_row[0]) + 1

                attempt = Attempt(
                    rollout_id=rollout_id,
                    attempt_id=f"attempt-{uuid.uuid4()}",
                    sequence_id=next_sequence,
                    start_time=time.time(),
                    status="preparing",
                )

                rollout.status = "preparing"
                rollout.end_time = None

                connection.execute(
                    "UPDATE rollouts SET status=?, end_time=? WHERE rollout_id=?",
                    (rollout.status, rollout.end_time, rollout.rollout_id),
                )
                connection.execute(
                    """
                    INSERT INTO attempts (
                        attempt_id, rollout_id, sequence_id, status, start_time, end_time,
                        worker_id, last_heartbeat_time, metadata_blob, span_counter
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        attempt.attempt_id,
                        attempt.rollout_id,
                        attempt.sequence_id,
                        attempt.status,
                        attempt.start_time,
                        attempt.end_time,
                        attempt.worker_id,
                        attempt.last_heartbeat_time,
                        _serialize_optional(attempt.metadata),
                    ),
                )
                return AttemptedRollout(**rollout.model_dump(), attempt=attempt)

            return await self._execute_write(writer)

    @_healthcheck_wrapper
    async def query_rollouts(
        self,
        *,
        status: Optional[Sequence[RolloutStatus]] = None,
        rollout_ids: Optional[Sequence[str]] = None,
    ) -> List[RolloutV2]:
        """Return rollouts filtered by status and/or identifiers.

        Args:
            status: Optional list of rollout statuses to include.
            rollout_ids: Optional list of rollout IDs to include regardless of
                status.

        Returns:
            List[RolloutV2]: All rollouts that satisfy the provided filters,
            ordered by ``start_time``.
        """
        async with self._lock:
            if status is not None and len(status) == 0:
                return []
            if rollout_ids is not None and len(rollout_ids) == 0:
                return []

            def reader(connection: sqlite3.Connection) -> List[RolloutV2]:
                clauses = ["kind='rollout'"]
                params: List[Any] = []
                if status is not None:
                    placeholders = ",".join("?" for _ in status)
                    clauses.append(f"status IN ({placeholders})")
                    params.extend(status)
                if rollout_ids is not None:
                    placeholders = ",".join("?" for _ in rollout_ids)
                    clauses.append(f"rollout_id IN ({placeholders})")
                    params.extend(rollout_ids)
                where_clause = " AND ".join(clauses)
                query = f"SELECT * FROM rollouts WHERE {where_clause} ORDER BY start_time ASC"
                rows = connection.execute(query, tuple(params)).fetchall()
                return [self._row_to_rollout(row) for row in rows]

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """Return all attempts created for ``rollout_id`` in sequence order."""
        async with self._lock:

            def reader(connection: sqlite3.Connection) -> List[Attempt]:
                rows = connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE rollout_id=?
                    ORDER BY sequence_id ASC
                    """,
                    (rollout_id,),
                ).fetchall()
                return [self._row_to_attempt(row) for row in rows]

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def get_rollout_by_id(self, rollout_id: str) -> Optional[RolloutV2]:
        """Fetch a rollout by ID or ``None`` if it does not exist."""
        async with self._lock:

            def reader(connection: sqlite3.Connection) -> Optional[RolloutV2]:
                row = self._get_rollout_row_optional(connection, rollout_id)
                return self._row_to_rollout(row) if row is not None else None

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """Return the most recent attempt for ``rollout_id`` if present."""
        async with self._lock:

            def reader(connection: sqlite3.Connection) -> Optional[Attempt]:
                row = connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE rollout_id=?
                    ORDER BY sequence_id DESC
                    LIMIT 1
                    """,
                    (rollout_id,),
                ).fetchone()
                return self._row_to_attempt(row) if row is not None else None

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """Return the resources snapshot with ``resources_id`` if available."""
        async with self._lock:

            def reader(connection: sqlite3.Connection) -> Optional[ResourcesUpdate]:
                latest, resources = self._load_resources_store(connection)
                update = resources.get(resources_id)
                if update is None:
                    return None
                if latest is not None:
                    self._latest_resources_id = latest
                return update

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Return the latest resources snapshot or ``None`` if none exist."""
        async with self._lock:

            def reader(connection: sqlite3.Connection) -> Optional[ResourcesUpdate]:
                latest, resources = self._load_resources_store(connection)
                self._latest_resources_id = latest
                if latest is None:
                    return None
                return resources.get(latest)

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """Overwrite the resources snapshot ``resources_id`` with ``resources``.

        Args:
            resources_id: Identifier of the snapshot being updated.
            resources: Mapping of resource names to typed payloads.

        Returns:
            ResourcesUpdate: The persisted resources snapshot.
        """
        async with self._lock:
            update = ResourcesUpdate(resources_id=resources_id, resources=resources)

            def writer(connection: sqlite3.Connection) -> ResourcesUpdate:
                latest, store = self._load_resources_store(connection)
                store[resources_id] = update
                latest = resources_id
                self._persist_resources_store(connection, latest_id=latest, resources=store)
                self._latest_resources_id = latest
                return update

            return await self._execute_write(writer)

    @_healthcheck_wrapper
    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """Create a new resources snapshot and mark it as the latest version."""
        resources_id = f"resources-{uuid.uuid4()}"
        return await self.update_resources(resources_id, resources)

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Reserve and return the next span sequence ID for an attempt."""
        async with self._lock:

            def writer(connection: sqlite3.Connection) -> int:
                row = self._get_attempt_row(connection, attempt_id)
                if row["rollout_id"] != rollout_id:
                    raise ValueError("Attempt does not belong to rollout")
                next_value = int(row["span_counter"]) + 1
                connection.execute(
                    "UPDATE attempts SET span_counter=? WHERE attempt_id=?",
                    (next_value, attempt_id),
                )
                return next_value

            return await self._execute_write(writer)

    def _persist_span(self, connection: sqlite3.Connection, span: Span) -> Span:
        """Insert or update ``span`` and update attempt/rollout status.

        The method is synchronous so it expects to be called inside
        :meth:`_transaction`. It ensures that the attempt's span counter and
        status, as well as the rollout status, reflect the new span.
        """
        attempt_row = self._get_attempt_row(connection, span.attempt_id)
        rollout_row = self._get_rollout_row(connection, span.rollout_id)
        latest_attempt_row = self._get_latest_attempt_row(connection, span.rollout_id)

        if attempt_row["rollout_id"] != span.rollout_id:
            raise ValueError("Span attempt/rollout mismatch")

        # Persist the span using ``INSERT OR REPLACE`` so replays from workers
        # are idempotent. The structured JSON fields keep attributes/events
        # queryable while still supporting arbitrary nested data via
        # ``_json_dumps``.
        row = self._span_to_row(span)
        connection.execute(
            """
            INSERT OR REPLACE INTO spans (
                rollout_id, attempt_id, sequence_id, trace_id, span_id, parent_id,
                name, status_json, attributes_json, events_json, links_json,
                start_time, end_time, context_json, parent_context_json, resource_json, extras_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.rollout_id,
                row.attempt_id,
                row.sequence_id,
                row.trace_id,
                row.span_id,
                row.parent_id,
                row.name,
                row.status_json,
                row.attributes_json,
                row.events_json,
                row.links_json,
                row.start_time,
                row.end_time,
                row.context_json,
                row.parent_context_json,
                row.resource_json,
                row.extras_json,
            ),
        )

        new_counter = max(int(attempt_row["span_counter"]), span.sequence_id)
        now = time.time()
        new_status = attempt_row["status"]
        # Seeing a span means the worker is actively running. Ensure the
        # attempt transitions out of states that indicate it has not started or
        # has stalled and bump the counter we use to allocate sequence ids.
        if new_status in {"preparing", "unresponsive", "timeout"}:
            new_status = "running"
        connection.execute(
            "UPDATE attempts SET status=?, last_heartbeat_time=?, span_counter=? WHERE attempt_id=?",
            (new_status, now, new_counter, span.attempt_id),
        )

        if latest_attempt_row["attempt_id"] == span.attempt_id:
            rollout_status = rollout_row["status"]
            if rollout_status in {"preparing", "queuing", "requeuing"}:
                # Promote the rollout to running once the latest attempt starts
                # emitting spans. Earlier attempts do not affect the rollout
                # status because they already reached a terminal state.
                connection.execute(
                    "UPDATE rollouts SET status=?, end_time=NULL WHERE rollout_id=?",
                    ("running", span.rollout_id),
                )

        return span

    @_healthcheck_wrapper
    async def add_span(self, span: Span) -> Span:
        """Persist ``span`` and update the owning attempt/rollout state."""
        async with self._lock:

            def writer(connection: sqlite3.Connection) -> Span:
                return self._persist_span(connection, span)

            return await self._execute_write(writer)

    @_healthcheck_wrapper
    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        """Convert and persist an OpenTelemetry span for the given attempt."""
        if sequence_id is None:
            sequence_id = await self.get_next_span_sequence_id(rollout_id, attempt_id)

        span = Span.from_opentelemetry(
            readable_span,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
        )

        async with self._lock:

            def writer(connection: sqlite3.Connection) -> Span:
                return self._persist_span(connection, span)

            return await self._execute_write(writer)

    @_healthcheck_wrapper
    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        """List spans for ``rollout_id`` optionally scoped to an attempt."""
        async with self._lock:

            def reader(connection: sqlite3.Connection) -> List[Span]:
                attempt_filter: Optional[str]
                if attempt_id == "latest":
                    row = connection.execute(
                        """
                        SELECT attempt_id FROM attempts
                        WHERE rollout_id=?
                        ORDER BY sequence_id DESC
                        LIMIT 1
                        """,
                        (rollout_id,),
                    ).fetchone()
                    attempt_filter = row["attempt_id"] if row is not None else None
                else:
                    attempt_filter = attempt_id

                sql = "SELECT * FROM spans WHERE rollout_id=?"
                params: List[Any] = [rollout_id]
                if attempt_filter is not None:
                    sql += " AND attempt_id=?"
                    params.append(attempt_filter)
                sql += " ORDER BY sequence_id ASC"
                rows = connection.execute(sql, tuple(params)).fetchall()
                return [self._row_to_span(row) for row in rows]

            return await self._execute_read(reader)

    @_healthcheck_wrapper
    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[RolloutV2]:
        """Wait until every rollout in ``rollout_ids`` finishes or times out.

        Args:
            rollout_ids: Identifiers for the rollouts to wait on.
            timeout: Optional timeout applied to each rollout wait.

        Returns:
            List[RolloutV2]: Rollouts that completed before the timeout.
        """
        completed: List[RolloutV2] = []

        async def wait_for_single(rollout_id: str) -> None:
            async with self._lock:

                def reader(connection: sqlite3.Connection) -> Optional[RolloutV2]:
                    row = self._get_rollout_row_optional(connection, rollout_id)
                    return self._row_to_rollout(row) if row is not None else None

                rollout = await self._execute_read(reader)
                if rollout and is_finished(rollout):
                    completed.append(rollout)
                    return
                event = self._completion_events.setdefault(rollout_id, asyncio.Event())

            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return

            async with self._lock:

                def reader_after(connection: sqlite3.Connection) -> Optional[RolloutV2]:
                    row = self._get_rollout_row_optional(connection, rollout_id)
                    return self._row_to_rollout(row) if row is not None else None

                rollout = await self._execute_read(reader_after)
                if rollout and is_finished(rollout):
                    completed.append(rollout)

        await asyncio.gather(*(wait_for_single(rid) for rid in rollout_ids))
        return completed

    @staticmethod
    def _propagate_attempt_to_rollout(rollout: RolloutV2, attempt: Attempt) -> None:
        if attempt.status in {"preparing", "running", "succeeded"}:
            rollout.status = cast(RolloutStatus, attempt.status)
            if attempt.status == "succeeded":
                rollout.end_time = time.time()
            return

        if attempt.status in {"failed", "timeout", "unresponsive"}:
            if attempt.status in rollout.config.retry_condition and attempt.sequence_id < rollout.config.max_attempts:
                rollout.status = "requeuing"
                rollout.end_time = None
            else:
                rollout.status = "failed"
                rollout.end_time = time.time()
            return

        raise ValueError(f"Invalid attempt status: {attempt.status}")

    @_healthcheck_wrapper
    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> RolloutV2:
        """Update selected rollout fields inside a single transaction.

        Each argument defaults to :data:`UNSET` so that callers can update a
        subset of fields without overwriting the others.

        Args:
            rollout_id: Identifier of the rollout to update.
            input: New task input.
            mode: Updated execution mode.
            resources_id: Updated resources snapshot identifier.
            status: Updated rollout status.
            config: Updated rollout configuration.
            metadata: Updated metadata payload.

        Returns:
            RolloutV2: The rollout after applying the updates.
        """
        async with self._lock:

            def writer(connection: sqlite3.Connection) -> RolloutV2:
                row = self._get_rollout_row(connection, rollout_id)
                rollout = self._row_to_rollout(row)

                if not isinstance(input, Unset):
                    rollout.input = input
                if not isinstance(mode, Unset):
                    rollout.mode = mode
                if not isinstance(resources_id, Unset):
                    rollout.resources_id = resources_id
                if not isinstance(status, Unset):
                    rollout.status = status
                if not isinstance(config, Unset):
                    rollout.config = config
                if not isinstance(metadata, Unset):
                    rollout.metadata = metadata

                if status is not UNSET and is_finished(rollout):
                    rollout.end_time = time.time()
                elif status is not UNSET and not is_queuing(rollout):
                    rollout.end_time = None

                connection.execute(
                    """
                    UPDATE rollouts
                    SET status=?, start_time=?, end_time=?, mode=?, resources_id=?,
                        input_blob=?, config_json=?, metadata_blob=?
                    WHERE rollout_id=?
                    """,
                    (
                        rollout.status,
                        rollout.start_time,
                        rollout.end_time,
                        rollout.mode,
                        rollout.resources_id,
                        _serialize_pickle(rollout.input),
                        _json_dumps(rollout.config.model_dump()),
                        _serialize_optional(rollout.metadata),
                        rollout.rollout_id,
                    ),
                )

                if status is not UNSET and is_finished(rollout):
                    self._set_completion_event(rollout.rollout_id)

                return rollout

            return await self._execute_write(writer)

    @_healthcheck_wrapper
    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        """Update attempt fields and propagate any status changes.

        Args:
            rollout_id: Identifier of the parent rollout.
            attempt_id: Attempt identifier or ``"latest"`` to pick the newest
                attempt.
            status: New attempt status.
            worker_id: Worker identifier running the attempt.
            last_heartbeat_time: Timestamp of the latest heartbeat.
            metadata: Updated metadata payload.

        Returns:
            Attempt: The attempt after applying the updates.
        """
        async with self._lock:

            def writer(connection: sqlite3.Connection) -> Attempt:
                if attempt_id == "latest":
                    attempt_row = self._get_latest_attempt_row(connection, rollout_id)
                else:
                    attempt_row = self._get_attempt_row(connection, attempt_id)
                attempt = self._row_to_attempt(attempt_row)

                if not isinstance(status, Unset):
                    attempt.status = status
                    if status in {"failed", "succeeded"}:
                        attempt.end_time = time.time()
                if not isinstance(worker_id, Unset):
                    attempt.worker_id = worker_id
                if not isinstance(last_heartbeat_time, Unset):
                    attempt.last_heartbeat_time = last_heartbeat_time
                if not isinstance(metadata, Unset):
                    attempt.metadata = metadata

                connection.execute(
                    """
                    UPDATE attempts
                    SET status=?, start_time=?, end_time=?, worker_id=?, last_heartbeat_time=?, metadata_blob=?
                    WHERE attempt_id=?
                    """,
                    (
                        attempt.status,
                        attempt.start_time,
                        attempt.end_time,
                        attempt.worker_id,
                        attempt.last_heartbeat_time,
                        _serialize_optional(attempt.metadata),
                        attempt.attempt_id,
                    ),
                )

                latest_row = self._get_latest_attempt_row(connection, rollout_id)
                if latest_row["attempt_id"] == attempt.attempt_id and not isinstance(status, Unset):
                    rollout_row = self._get_rollout_row(connection, rollout_id)
                    rollout = self._row_to_rollout(rollout_row)
                    self._propagate_attempt_to_rollout(rollout, attempt)
                    connection.execute(
                        """
                        UPDATE rollouts
                        SET status=?, end_time=?
                        WHERE rollout_id=?
                        """,
                        (
                            rollout.status,
                            rollout.end_time,
                            rollout.rollout_id,
                        ),
                    )
                    if is_finished(rollout):
                        self._set_completion_event(rollout.rollout_id)

                return attempt

            return await self._execute_write(writer)

    async def _fetch_running_rollouts(self) -> List[AttemptedRollout]:
        """Return all rollouts that currently have in-flight attempts."""

        def reader(connection: sqlite3.Connection) -> List[AttemptedRollout]:
            rows = connection.execute(
                "SELECT * FROM rollouts WHERE kind='rollout' AND status IN ('preparing', 'running')",
            ).fetchall()
            results: List[AttemptedRollout] = []
            for row in rows:
                rollout = self._row_to_rollout(row)
                # Only the latest attempt can still change state or emit
                # heartbeats; older attempts are already terminal.
                attempt_row = connection.execute(
                    """
                    SELECT * FROM attempts
                    WHERE rollout_id=?
                    ORDER BY sequence_id DESC
                    LIMIT 1
                    """,
                    (rollout.rollout_id,),
                ).fetchone()
                if attempt_row is None:
                    continue
                attempt = self._row_to_attempt(attempt_row)
                results.append(AttemptedRollout(**rollout.model_dump(), attempt=attempt))
            return results

        return await self._execute_read(reader)

    async def _healthcheck(self) -> None:
        """Inspect in-flight attempts and update their health/status.

        The check runs before most public APIs via :func:`_healthcheck_wrapper`.
        It promotes attempts to ``running`` when we observe heartbeats, marks
        attempts as ``timeout``/``unresponsive`` based on the rollout config,
        and propagates terminal attempt states back to the owning rollout.
        """
        async with self._lock:
            running = await self._fetch_running_rollouts()

        current_time = time.time()

        for attempted in running:
            rollout = attempted
            attempt = attempted.attempt
            config = attempted.config

            # Mirror terminal attempt outcomes onto the parent rollout so
            # callers see consistent state even if the attempt completed on a
            # worker without touching the store again.
            if attempt.status in {"failed", "succeeded"}:
                await self.update_rollout(
                    rollout.rollout_id,
                    status=cast(RolloutStatus, attempt.status),
                )
                continue

            # Deadline enforcement based on the rollout configuration.
            if config.timeout_seconds is not None and current_time - attempt.start_time > config.timeout_seconds:
                await self.update_attempt(rollout.rollout_id, attempt.attempt_id, status="timeout")
                continue

            last_heartbeat = attempt.last_heartbeat_time
            if last_heartbeat is not None:
                # Heartbeats show the worker is alive—promote to running if it
                # has not already made that transition.
                if attempt.status == "preparing":
                    await self.update_attempt(rollout.rollout_id, attempt.attempt_id, status="running")
                    attempt = attempt.model_copy(update={"status": "running"})
                if (
                    config.unresponsive_seconds is not None
                    and current_time - last_heartbeat > config.unresponsive_seconds
                ):
                    # The worker went silent after previously heartbeat-ing.
                    await self.update_attempt(rollout.rollout_id, attempt.attempt_id, status="unresponsive")
                continue

            if (
                config.unresponsive_seconds is not None
                and current_time - attempt.start_time > config.unresponsive_seconds
            ):
                # Attempts without heartbeats eventually become ``unresponsive``
                # purely based on wall-clock time.
                await self.update_attempt(rollout.rollout_id, attempt.attempt_id, status="unresponsive")
