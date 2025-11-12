# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union, TypedDict

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


class LightningStoreCapabilities(TypedDict):
    """Capability of a LightningStore implementation."""

    thread_safe: bool
    """Whether the store is thread-safe."""
    async_safe: bool
    """Whether the store is async-safe."""
    zero_copy: bool
    """Whether the store has only one copy across all threads/processes."""


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

    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            thread_safe=False,
            async_safe=False,
            zero_copy=False,
        )

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

    async def dequeue_rollout(self) -> Optional[AttemptedRollout]:
        """Claim the oldest queued rollout and transition it to `preparing`.

        This function do not block.

        Retrieval must be FIFO across rollouts that remain in `queuing` or `requeuing`
        state. When a rollout is claimed, implementations must:

        * Transition its status to `"preparing"`.
        * Create a new attempt with `status="preparing"` and `sequence_id` equal to
          the number of attempts already registered for the rollout plus one.
        * Return an [`AttemptedRollout`][agentlightning.AttemptedRollout] snapshot so the
          runner knows both rollout metadata and the attempt identifier.

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
        self, *, status: Optional[Sequence[RolloutStatus]] = None, rollout_ids: Optional[Sequence[str]] = None
    ) -> List[Union[Rollout, AttemptedRollout]]:
        """Retrieve rollouts filtered by status and/or explicit identifiers.

        Args:
            status: Optional whitelist of [`RolloutStatus`][agentlightning.RolloutStatus] values.
            rollout_ids: Optional whitelist of rollout identifiers to include.

        Returns:
            A list of matching rollouts. Ordering is backend-defined but must be deterministic.

        Raises:
            NotImplementedError: Subclasses must implement the query.
        """
        raise NotImplementedError()

    async def query_attempts(self, rollout_id: str) -> List[Attempt]:
        """Return every attempt ever created for `rollout_id` in ascending sequence order.

        Args:
            rollout_id: Identifier of the rollout being inspected.

        Returns:
            Attempts sorted by `sequence_id` (oldest first). Returns an empty list when none exist.

        Raises:
            NotImplementedError: Subclasses must implement the query.
            ValueError: Implementations must raise when the rollout does not exist.
        """
        raise NotImplementedError()

    async def get_rollout_by_id(self, rollout_id: str) -> Optional[Union[Rollout, AttemptedRollout]]:
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

    async def query_resources(self) -> List[ResourcesUpdate]:
        """List every stored resource snapshot in insertion order.

        Returns:
            A chronological list of [`ResourcesUpdate`][agentlightning.ResourcesUpdate] objects.

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

    async def query_spans(self, rollout_id: str, attempt_id: str | Literal["latest"] | None = None) -> List[Span]:
        """Return the stored spans for a rollout, optionally scoped to one attempt.

        Spans must be returned in ascending `sequence_id` order. Implementations may raise
        a `RuntimeError` when spans were evicted or expired.

        Args:
            rollout_id: Identifier of the rollout being inspected.
            attempt_id: Attempt identifier to filter by. Pass `"latest"` to retrieve only the
                most recent attempt, or `None` to return all spans across attempts.

        Returns:
            An ordered list of spans (possibly empty).

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

        If `resources_id` does not exist, implementations should add it as a new snapshot.

        Args:
            resources_id: Identifier of the snapshot to replace.
            resources: Updated mapping of resource names to payloads.

        Returns:
            The persisted [`ResourcesUpdate`][agentlightning.ResourcesUpdate].

        Raises:
            NotImplementedError: Subclasses must implement resource persistence.
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
