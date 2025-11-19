# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, TypedDict

from opentelemetry.sdk.trace import ReadableSpan

from agentlightning.types import (
    Attempt,
    AttemptedRollout,
    AttemptStatus,
    NamedResources,
    ResourcesUpdate,
    Rollout,
    RolloutConfig,
    RolloutStatus,
    Span,
    TaskInput,
    Worker,
    WorkerStatus,
)


def is_queuing(rollout: Rollout) -> bool:
    return rollout.status == "queuing" or rollout.status == "requeuing"


def is_running(rollout: Rollout) -> bool:
    return rollout.status == "preparing" or rollout.status == "running"


def is_finished(rollout: Rollout) -> bool:
    return rollout.status == "failed" or rollout.status == "succeeded" or rollout.status == "cancelled"


class _UnsetType:
    """A sentinel type to indicate an unset value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"

    def __reduce__(self):
        return (_get_unset, ())


def _get_unset() -> _UnsetType:
    return UNSET


UNSET = _UnsetType()
Unset = _UnsetType  # Alias for convenience


class LightningStoreCapabilities(TypedDict, total=False):
    """Capability of a LightningStore implementation.

    All keys are optional and false by default.
    """

    thread_safe: bool
    """Whether the store is thread-safe."""
    async_safe: bool
    """Whether the store is async-safe."""
    zero_copy: bool
    """Whether the store has only one copy across all threads/processes."""
    otlp_traces: bool
    """Whether the store supports OTLP/HTTP traces."""


class LightningStore:
    """Contract for the persistent control-plane that coordinates training rollouts.

    A `LightningStore` mediates every interaction between algorithms and runners:

    - **Rollout lifecycle:** accept new rollouts, queue them for execution, create attempts,
      and drive the rollout status machine (`"queuing"` → `"preparing"` → `"running"` →
      `{"succeeded","failed","cancelled"}` or `"requeuing"` when a retry is justified).
    - **Attempt tracking:** record each execution attempt, including progress heartbeats,
      retry sequencing, and terminal states such as `"timeout"` or `"unresponsive"`.
    - **Span ingest:** capture structured telemetry emitted by runners (either as native
      [`Span`][agentlightning.Span] objects or as `opentelemetry.sdk.trace.ReadableSpan`
      instances) so that algorithms can reconstruct trajectories and rewards.
    - **Resource versioning:** manage immutable snapshots of named resources
      (prompt templates, model checkpoints, proxy endpoints, …) and expose a single
      "latest" snapshot that runners can fetch just after claiming work.

    Implementations must provide thread-safe/async-safe semantics: each coroutine should
    appear atomic to callers even when multiple algorithms or runners call the API concurrently.
    Unless stated otherwise, missing identifiers should result in a `ValueError`.
    """

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            thread_safe=False,
            async_safe=False,
            zero_copy=False,
            otlp_traces=False,
        )

    def otlp_traces_endpoint(self) -> str:
        """Return the OTLP/HTTP traces endpoint of the store.

        The traces can have rollout ID and attempt ID (and optionally sequence ID)
        saved in the "resource" of the spans.
        The store, if it supports OTLP, should be able to receive the traces and save them
        via [`add_span`][agentlightning.LightningStore.add_span] or
        [`add_otel_span`][agentlightning.LightningStore.add_otel_span].

        The endpoint should be compatible with [OTLP HTTP protocol](https://opentelemetry.io/docs/specs/otlp/).
        It's not necessarily compatible with OTLP gRPC protocol.

        The returned endpoint will usually ends with `/v1/traces`.
        """
        raise NotImplementedError()

    async def start_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> AttemptedRollout:
        """Register a rollout and immediately create its first attempt.

        !!! note
            Use [`enqueue_rollout()`][agentlightning.LightningStore.enqueue_rollout] when the
            caller only wants to submit work for later scheduling.

        The rollout must be persisted with `status="preparing"` and an initial attempt
        with `sequence_id == 1` so the caller can begin execution without visiting the
        public queue. Implementations are expected to:

        1. Generate a unique `rollout_id` and `attempt_id`.
        2. Record `start_time` for both rollout and attempt based on the current clock.
        3. Copy `config` and `metadata` so later mutations do not leak shared references.
        4. Resolve `resources_id` to the latest resource snapshot when `None` is supplied.

        Args:
            input: Arbitrary task payload supplied by an algorithm.
            mode: Optional semantic mode for downstream analytics (`"train"`, `"val"`, `"test"`).
            resources_id: Concrete resource snapshot to execute against; defaults to the latest stored snapshot.
            config: Rollout retry/timeout policy. Should default to a fresh [`RolloutConfig`][agentlightning.RolloutConfig].
            metadata: Free-form metadata persisted verbatim with the rollout.

        Returns:
            The fully-populated [`AttemptedRollout`][agentlightning.AttemptedRollout] including
            the just-created attempt.

        Raises:
            NotImplementedError: Subclasses must provide durable storage for the rollout.
            ValueError: Implementations should raise when `resources_id` does not exist.
        """
        raise NotImplementedError()

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        """Persist a rollout in `queuing` state so runners can claim it later.

        !!! note
            Different from [`start_rollout()`][agentlightning.LightningStore.start_rollout],
            this method is called when the caller only wants to submit work for later scheduling.

        Implementations must generate a unique `rollout_id`, stamp `start_time` with
        the current time, default `config` to a fresh [`RolloutConfig`][agentlightning.RolloutConfig],
        and insert the rollout at the tail of the scheduling queue. No attempt is created yet.

        Args:
            input: Arbitrary task payload supplied by an algorithm.
            mode: Optional semantic mode indicator (`"train"`, `"val"`, `"test"`).
            resources_id: Resource snapshot used when a runner eventually executes the rollout.
            config: Fine-grained retry/timeout parameters to persist with the rollout.
            metadata: Free-form metadata stored verbatim with the rollout record.

        Returns:
            The stored [`Rollout`][agentlightning.Rollout] in `queuing` status.

        Raises:
            NotImplementedError: Subclasses must persist the rollout.
            ValueError: Implementations should raise when `resources_id` does not exist.
        """
        raise NotImplementedError()

    async def dequeue_rollout(self, worker_id: Optional[str] = None) -> Optional[AttemptedRollout]:
        """Claim the oldest queued rollout and transition it to `preparing`.

        This function do not block.

        Retrieval must be FIFO across rollouts that remain in `queuing` or `requeuing`
        state. When a rollout is claimed, implementations must:

        * Transition its status to `"preparing"`.
        * Create a new attempt with `status="preparing"` and `sequence_id` equal to
          the number of attempts already registered for the rollout plus one.
        * Return an [`AttemptedRollout`][agentlightning.AttemptedRollout] snapshot so the
          runner knows both rollout metadata and the attempt identifier.
        * Optionally refresh the caller's [`Worker`][agentlightning.Worker] telemetry
          (e.g., `last_dequeue_time`) when `worker_id` is provided.

        Returns:
            The next attempt to execute, or `None` when no eligible rollouts are queued.

        Raises:
            NotImplementedError: Subclasses must implement queue retrieval.
        """
        raise NotImplementedError()

    async def start_attempt(self, rollout_id: str) -> AttemptedRollout:
        """Create a manual retry attempt for an existing rollout.

        This is typically invoked by runners that wish to retry outside of the
        normal queue flow (for example in an online RL setup).
        Implementations must validate that the rollout exists, allocate a fresh `attempt_id`,
        increment the `sequence_id` monotonically, stamp the new attempt with `status="preparing"`,
        and return an up-to-date [`AttemptedRollout`][agentlightning.AttemptedRollout].

        Args:
            rollout_id: Unique identifier of the rollout receiving a new attempt.

        Returns:
            The rollout paired with its newly-created attempt.

        Raises:
            NotImplementedError: Subclasses must implement attempt creation.
            ValueError: Implementations must raise when `rollout_id` is unknown.
        """
        raise NotImplementedError()

    async def add_span(self, span: Span) -> Span:
        """Persist a pre-constructed span emitted during rollout execution.

        The provided [`Span`][agentlightning.Span] must already contain the `rollout_id`,
        `attempt_id`, and `sequence_id`. Implementations must:

        * Verify that both rollout and attempt exist.
        * Ensure span ordering remains strictly increasing per attempt (rejecting or keeping duplicates).
        * Treat the span arrival as a heartbeat: update the attempt's `last_heartbeat_time`
          and transition both attempt and rollout to `"running"` if they were still
          `"preparing"` or `"requeuing"`.

        Args:
            span: Fully populated span to persist.

        Returns:
            The stored span record (implementations may return a copy).

        Raises:
            NotImplementedError: Subclasses must implement span persistence.
            ValueError: Implementations must raise when the referenced rollout or attempt is missing.
        """
        raise NotImplementedError()

    async def add_otel_span(
        self,
        rollout_id: str,
        attempt_id: str,
        readable_span: ReadableSpan,
        sequence_id: int | None = None,
    ) -> Span:
        """Convert and persist an OpenTelemetry span for a particular attempt.

        Implementations must transform the `readable_span` into a [`Span`][agentlightning.Span]
        (typically via [`Span.from_opentelemetry()`][agentlightning.Span.from_opentelemetry]),
        assign a strictly increasing `sequence_id` when one is not provided, and persist it
        using the same semantics as [`add_span()`][agentlightning.LightningStore.add_span].

        Args:
            rollout_id: Identifier of the rollout that produced the span.
            attempt_id: Attempt identifier the span belongs to.
            readable_span: OpenTelemetry span in SDK form.
            sequence_id: Optional explicit ordering hint. When omitted, call
                [`get_next_span_sequence_id()`][agentlightning.LightningStore.get_next_span_sequence_id]
                automatically.

        Returns:
            The stored span record.

        Raises:
            NotImplementedError: Subclasses must implement span persistence.
            ValueError: Implementations must raise when the rollout or attempt is unknown.
        """
        raise NotImplementedError()

    async def query_rollouts(
        self,
        *,
        status_in: Optional[Sequence[RolloutStatus]] = None,
        rollout_id_in: Optional[Sequence[str]] = None,
        rollout_id_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
        # Deprecated fields
        status: Optional[Sequence[RolloutStatus]] = None,
        rollout_ids: Optional[Sequence[str]] = None,
    ) -> Sequence[Rollout]:
        """Retrieve rollouts filtered by status and/or explicit identifiers.

        This interface supports structured filtering, sorting, and pagination so
        callers can build simple dashboards without copying data out of the
        store. The legacy parameters `status` and `rollout_ids` remain valid and
        are treated as aliases for `status_in` and `rollout_id_in`
        respectively—when both the new and deprecated parameters are supplied
        the new parameters take precedence.

        Args:
            status_in: Optional whitelist of [`RolloutStatus`][agentlightning.RolloutStatus] values.
            rollout_id_in: Optional whitelist of rollout identifiers to include.
            rollout_id_contains: Optional substring match for rollout identifiers.
            filter_logic: Logical operator to combine filters.
            sort_by: Optional field to sort by. Must reference a numeric or string
                field on [`Rollout`][agentlightning.Rollout].
            sort_order: Direction to sort when `sort_by` is provided.
            limit: Maximum number of rows to return. Use `-1` for "no limit".
            offset: Number of rows to skip before returning results.
            status: Deprecated field. Use `status_in` instead.
            rollout_ids: Deprecated field. Use `rollout_id_in` instead.

        Returns:
            A sequence of matching rollouts (or [`AttemptedRollout`][agentlightning.AttemptedRollout]
            when attempts exist). Ordering is deterministic when `sort_by` is set.
            The return value is not guaranteed to be a list.

        Raises:
            NotImplementedError: Subclasses must implement the query.
        """
        raise NotImplementedError()

    async def query_attempts(
        self,
        rollout_id: str,
        *,
        sort_by: Optional[str] = "sequence_id",
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> Sequence[Attempt]:
        """Return every attempt ever created for `rollout_id` in ascending sequence order.

        The parameters allow callers to re-order or paginate the attempts so that
        large retry histories can be streamed lazily.

        Args:
            rollout_id: Identifier of the rollout being inspected.
            sort_by: Field to sort by. Must be a numeric or string field of
                [`Attempt`][agentlightning.Attempt]. Defaults to `sequence_id` (oldest first).
            sort_order: Order to sort by.
            limit: Limit on the number of results. `-1` for unlimited.
            offset: Offset into the results.

        Returns:
            Sequence of Attempts. Returns an empty sequence when none exist.
            The return value is not guaranteed to be a list.

        Raises:
            NotImplementedError: Subclasses must implement the query.
            ValueError: Implementations must raise when the rollout does not exist.
        """
        raise NotImplementedError()

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Rollout]:
        """Fetch a rollout by identifier without mutating its state.

        Args:
            rollout_id: Identifier to retrieve.

        Returns:
            The rollout when found, otherwise `None`.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_latest_attempt(self, rollout_id: str) -> Optional[Attempt]:
        """Fetch the attempt with the highest `sequence_id` for `rollout_id`.

        Args:
            rollout_id: Identifier to inspect.

        Returns:
            The most recent attempt or `None` when no attempts exist yet.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
            ValueError: Implementations must raise when the rollout does not exist.
        """
        raise NotImplementedError()

    async def query_resources(
        self,
        *,
        resources_id: Optional[str] = None,
        resources_id_contains: Optional[str] = None,
        # Filter logic is not supported here because I can't see why it's needed.
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> Sequence[ResourcesUpdate]:
        """List every stored resource snapshot in insertion order.

        Supports lightweight filtering, sorting, and pagination for embedding in
        dashboards.

        Args:
            resources_id: Optional identifier of the resources to include.
            resources_id_contains: Optional substring match for resources identifiers.
            sort_by: Optional field to sort by (must be numeric or string on
                [`ResourcesUpdate`][agentlightning.ResourcesUpdate]).
            sort_order: Order to sort by.
            limit: Limit on the number of results. `-1` for unlimited.
            offset: Offset into the results.

        Returns:
            [`ResourcesUpdate`][agentlightning.ResourcesUpdate] objects.
            By default, resources are sorted in a deterministic but undefined order.
            The return value is not guaranteed to be a list.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_resources_by_id(self, resources_id: str) -> Optional[ResourcesUpdate]:
        """Return a specific named resource snapshot by identifier.

        Args:
            resources_id: Identifier of the snapshot.

        Returns:
            The stored [`ResourcesUpdate`][agentlightning.ResourcesUpdate], or `None` when missing.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_latest_resources(self) -> Optional[ResourcesUpdate]:
        """Fetch the latest resource snapshot marked as the global default.

        Returns:
            The current latest [`ResourcesUpdate`][agentlightning.ResourcesUpdate], or `None` when
            no resources have been registered yet.

        Raises:
            NotImplementedError: Subclasses must implement retrieval.
        """
        raise NotImplementedError()

    async def get_next_span_sequence_id(self, rollout_id: str, attempt_id: str) -> int:
        """Allocate the next strictly increasing sequence number used to order spans.

        Implementations must retain counters so repeated calls return `1, 2, ...` without
        gaps unless spans were explicitly inserted with a custom `sequence_id`. The
        counter may be scoped per rollout or per attempt, but the sequence must be
        strictly increasing for spans emitted by the specified attempt so traces remain
        totally ordered.

        See [Distributed Tracing][distributed-tracing] for detailed motivations.

        Args:
            rollout_id: Identifier of the rollout emitting spans.
            attempt_id: Attempt identifier for the upcoming span.

        Returns:
            The next integer sequence identifier, unique within the attempt.

        Raises:
            NotImplementedError: Subclasses must provide the allocator.
            ValueError: Implementations must raise when the rollout or attempt does not exist.
        """
        raise NotImplementedError()

    async def wait_for_rollouts(self, *, rollout_ids: List[str], timeout: Optional[float] = None) -> List[Rollout]:
        """Block until the targeted rollouts reach a terminal status or the timeout expires.

        Terminal statuses are `"succeeded"`, `"failed"`, and `"cancelled"`. When the timeout
        elapses, implementations should return the subset of rollouts that are already terminal
        and omit the rest.

        !!! warning
            It's dangerous and might be event-loop blocking to call this function
            with a long timeout. It's a good idea to poll for the method to check
            if new completed rollouts can coming. Be careful in implementing the sleep logic
            to avoid busy-waiting.

        Args:
            rollout_ids: Identifiers of rollouts to watch.
            timeout: Maximum time in seconds to wait. `None` waits indefinitely.

        Returns:
            Rollouts that finished before the deadline, in arbitrary order.

        Raises:
            NotImplementedError: Subclasses must implement waiting semantics.
            ValueError: Implementations must raise when a rollout identifier is unknown.
        """
        raise NotImplementedError()

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
        *,
        # Filtering
        trace_id: Optional[str] = None,
        trace_id_contains: Optional[str] = None,
        span_id: Optional[str] = None,
        span_id_contains: Optional[str] = None,
        parent_id: Optional[str] = None,
        parent_id_contains: Optional[str] = None,
        name: Optional[str] = None,
        name_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        # Pagination
        limit: int = -1,
        offset: int = 0,
        # Sorting
        sort_by: Optional[str] = "sequence_id",
        sort_order: Literal["asc", "desc"] = "asc",
    ) -> Sequence[Span]:
        """Return the stored spans for a rollout, optionally scoped to one attempt.

        Supports a handful of filters that cover the most common debugging
        scenarios (matching `trace_id`/`span_id`/`parent_id` or substring
        matches on the span name). `attempt_id="latest"` acts as a convenience
        that resolves the most recent attempt before evaluating filters. When
        `attempt_id=None`, spans across every attempt are eligible. By default
        results are sorted by `sequence_id` (oldest first). Implementations may
        raise a `RuntimeError` when spans were evicted or expired.

        Args:
            rollout_id: Identifier of the rollout being inspected.
            attempt_id: Attempt identifier to filter by. Pass `"latest"` to retrieve only the
                most recent attempt, or `None` to return all spans across attempts.
            trace_id: Optional trace ID to filter by.
            trace_id_contains: Optional substring match for trace IDs.
            span_id: Optional span ID to filter by.
            span_id_contains: Optional substring match for span IDs.
            parent_id: Optional parent span ID to filter by.
            parent_id_contains: Optional substring match for parent span IDs.
            name: Optional span name to filter by.
            name_contains: Optional substring match for span names.
            filter_logic: Logical operator to combine the optional filters above.
                The `rollout_id` argument is always applied with AND semantics.
            limit: Limit on the number of results. `-1` for unlimited.
            offset: Offset into the results.
            sort_by: Field to sort by. Must be a numeric or string field of
                [`Span`][agentlightning.Span].
            sort_order: Order to sort by.

        Returns:
            An ordered list of spans (possibly empty).
            The return value is not guaranteed to be a list.

        Raises:
            NotImplementedError: Subclasses must implement the query.
            ValueError: Implementations must raise when the rollout or attempt is unknown.
        """
        raise NotImplementedError()

    async def add_resources(self, resources: NamedResources) -> ResourcesUpdate:
        """Persist a new immutable snapshot of named resources and mark it as latest.

        Implementations must assign a fresh `resources_id` and ensure subsequent calls to
        [`get_latest_resources()`][agentlightning.LightningStore.get_latest_resources] return the
        snapshot produced here.

        Args:
            resources: Mapping of resource names to their serialized payloads.

        Returns:
            The stored [`ResourcesUpdate`][agentlightning.ResourcesUpdate] including its generated id.

        Raises:
            NotImplementedError: Subclasses must implement resource persistence.
        """
        raise NotImplementedError()

    async def update_resources(self, resources_id: str, resources: NamedResources) -> ResourcesUpdate:
        """Overwrite or extend an existing resource snapshot and mark it as latest.

        This API is typically used by algorithms that maintain mutable resources (e.g., model
        checkpoints) under a stable identifier.

        Args:
            resources_id: Identifier of the snapshot to replace.
            resources: Updated mapping of resource names to payloads.

        Returns:
            The persisted [`ResourcesUpdate`][agentlightning.ResourcesUpdate].

        Raises:
            NotImplementedError: Subclasses must implement resource persistence.
            ValueError: Implementations must raise when `resources_id` does not exist.
        """
        raise NotImplementedError()

    async def update_rollout(
        self,
        rollout_id: str,
        input: TaskInput | Unset = UNSET,
        mode: Optional[Literal["train", "val", "test"]] | Unset = UNSET,
        resources_id: Optional[str] | Unset = UNSET,
        status: RolloutStatus | Unset = UNSET,
        config: RolloutConfig | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Rollout:
        """Update rollout metadata and, when provided, drive status transitions.

        Parameters default to the sentinel [`UNSET`][agentlightning.store.base.UNSET] to
        distinguish omitted fields from explicit `None` assignments. Implementations must:

        * Validate the rollout exists before mutating it.
        * Replace each property when a concrete value (including `None`) is supplied.
        * When the status switches into a terminal state, set `end_time` and signal any waiters.
        * When the status re-enters a queueing state, ensure the rollout is enqueued exactly once.

        Args:
            rollout_id: Identifier of the rollout to update.
            input: Replacement task payload; pass `None` to explicitly clear the input.
            mode: Replacement rollout mode.
            resources_id: Replacement resources snapshot reference.
            status: Target rollout status.
            config: Replacement retry/timeout configuration.
            metadata: Replacement metadata dictionary.

        Returns:
            The updated rollout record.

        Raises:
            NotImplementedError: Subclasses must implement mutation logic.
            ValueError: Implementations must raise when the rollout is unknown or the update is invalid.
        """
        raise NotImplementedError()

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        """Update attempt bookkeeping such as status, worker ownership, and heartbeats.

        When `attempt_id` is `"latest"` the update must target the attempt with the highest
        `sequence_id`; otherwise it must target the specific attempt. Implementations should
        propagate status changes to the rollout (for example via [`propagate_status()`][agentlightning.store.utils.propagate_status])
        once the latest attempt transitions to a terminal state.

        Similar to [`update_rollout()`][agentlightning.LightningStore.update_rollout],
        parameters also default to the sentinel [`UNSET`][agentlightning.store.base.UNSET].

        If `worker_id` is present, the worker status will be updated following the rules:

        1. If attempt status is "succeeded" or "failed", the corresponding worker status will be set to "idle".
        2. If attempt status is "unresponsive" or "timeout", the corresponding worker status will be set to "unknown".
        3. Otherwise, the worker status will be set to "busy".

        Args:
            rollout_id: Identifier of the rollout whose attempt will be updated.
            attempt_id: Attempt identifier or `"latest"` as a convenience.
            status: Replacement attempt status. Terminal statuses must set `end_time`.
            worker_id: Identifier for the worker currently processing the attempt.
            last_heartbeat_time: Wall-clock timestamp (seconds) of the latest heartbeat/span.
            metadata: Replacement metadata dictionary.

        Returns:
            The updated attempt record.

        Raises:
            NotImplementedError: Subclasses must implement mutation logic.
            ValueError: Implementations must raise when the rollout or attempt is unknown.
        """
        raise NotImplementedError()

    async def query_workers(
        self,
        *,
        status_in: Optional[Sequence[WorkerStatus]] = None,
        worker_id_contains: Optional[str] = None,
        filter_logic: Literal["and", "or"] = "and",
        sort_by: Optional[str] = None,
        sort_order: Literal["asc", "desc"] = "asc",
        limit: int = -1,
        offset: int = 0,
    ) -> Sequence[Worker]:
        """Query all workers in the system.

        Args:
            status_in: Optional whitelist of [`WorkerStatus`][agentlightning.WorkerStatus] values.
            worker_id_contains: Optional substring match for worker identifiers.
            filter_logic: Logical operator to combine the optional filters above.
            sort_by: Field to sort by. Must be a numeric or string field of [`Worker`][agentlightning.Worker].
            sort_order: Order to sort by.
            limit: Limit on the number of results. `-1` for unlimited.
            offset: Offset into the results.

        Returns:
            Sequence of Workers. Returns an empty sequence when none exist.
            The return value is not guaranteed to be a list.
        """
        raise NotImplementedError()

    async def get_worker_by_id(self, worker_id: str) -> Optional[Worker]:
        """Retrieve a single worker by identifier.

        Args:
            worker_id: Identifier of the worker.

        Returns:
            The worker record if it exists, otherwise `None`.

        Raises:
            NotImplementedError: Subclasses must implement lookup semantics.
        """
        raise NotImplementedError()

    async def update_worker(
        self,
        worker_id: str,
        heartbeat_stats: Dict[str, Any] | Unset = UNSET,
    ) -> Worker:
        """Record a heartbeat for `worker_id` and refresh telemetry.

        Implementations must treat this API as heartbeat-only: it should snapshot
        the latest stats when provided, stamp `last_heartbeat_time` with the
        current wall clock, and rely on other store mutations (`dequeue_rollout`,
        `update_attempt`, etc.) to drive the worker's busy/idle status,
        assignment, and activity timestamps.

        Args:
            worker_id: Identifier of the worker to update.
            heartbeat_stats: Replacement worker heartbeat statistics (non-null when provided).
        """
        raise NotImplementedError()
