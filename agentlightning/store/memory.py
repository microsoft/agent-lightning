# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from collections.abc import Iterable
from collections.abc import Mapping as MappingABC
from typing import (
    Any,
    Callable,
    Counter,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

from pydantic import BaseModel

from agentlightning.types import AttemptedRollout, PaginatedResult, Rollout, Span

from .base import UNSET, LightningStoreCapabilities, Unset, is_finished, is_running
from .collection import InMemoryLightningCollections
from .collection_based import CollectionBasedLightningStore

T_callable = TypeVar("T_callable", bound=Callable[..., Any])

logger = logging.getLogger(__name__)


def estimate_model_size(obj: Any) -> int:
    """Rough recursive size estimate for Pydantic BaseModel instances."""

    if isinstance(obj, BaseModel):
        values = cast(Iterable[Any], obj.__dict__.values())
        return sum(estimate_model_size(value) for value in values) + sys.getsizeof(cast(object, obj))
    if isinstance(obj, MappingABC):
        mapping = cast(Mapping[Any, Any], obj)
        return sum(estimate_model_size(value) for value in mapping.values()) + sys.getsizeof(cast(object, obj))
    if isinstance(obj, (list, tuple, set)):
        iterable = cast(Iterable[Any], obj)
        return sum(estimate_model_size(value) for value in iterable) + sys.getsizeof(cast(object, obj))
    return sys.getsizeof(cast(object, obj))


def _detect_total_memory_bytes() -> int:
    """Best-effort detection of the total available system memory in bytes."""

    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except ImportError:
        # Fallback to 8GB if memory cannot be detected.
        logger.error("psutil is not installed. Falling back to 8GB of memory in total.")
        return 8 * 1024**3


class InMemoryLightningStore(CollectionBasedLightningStore[InMemoryLightningCollections]):
    """
    In-memory implementation of LightningStore using Python data structures.
    Thread-safe and async-compatible but data is not persistent.

    Args:
        eviction_memory_threshold: The threshold for evicting spans in bytes.
            By default, it's 70% of the total VRAM available.
        safe_memory_threshold: The threshold for safe memory usage in bytes.
            By default, it's 80% of the eviction threshold.
        span_size_estimator: A function to estimate the size of a span in bytes.
            By default, it's a simple size estimator that uses sys.getsizeof.
    """

    def __init__(
        self,
        *,
        eviction_memory_threshold: float | int | None = None,
        safe_memory_threshold: float | int | None = None,
        span_size_estimator: Callable[[Span], int] | None = None,
    ):
        super().__init__(collections=InMemoryLightningCollections())

        self._start_time_by_rollout: Dict[str, float] = {}
        self._span_bytes_by_rollout: Dict[str, int] = Counter()
        self._total_span_bytes: int = 0
        self._evicted_rollout_span_sets: Set[str] = set()

        self._memory_capacity_bytes = _detect_total_memory_bytes()
        if self._memory_capacity_bytes <= 0:
            raise ValueError("Detected memory capacity must be positive")

        self._eviction_threshold_bytes = self._resolve_memory_threshold(
            eviction_memory_threshold,
            default_ratio=0.7,
            capacity_bytes=self._memory_capacity_bytes,
            name="eviction_memory_threshold",
            minimum=1,
        )

        if safe_memory_threshold is None:
            safe_memory_threshold = max(int(self._eviction_threshold_bytes * 0.8), 0)

        self._safe_threshold_bytes = self._resolve_memory_threshold(
            safe_memory_threshold,
            default_ratio=self._eviction_threshold_bytes / self._memory_capacity_bytes,
            capacity_bytes=self._memory_capacity_bytes,
            name="safe_memory_threshold",
            minimum=0,
        )

        if not (0 <= self._safe_threshold_bytes < self._eviction_threshold_bytes):
            raise ValueError("safe_memory_threshold must be smaller than eviction_memory_threshold")
        self._custom_span_size_estimator = span_size_estimator

        # Completion tracking for wait_for_rollouts (cross-loop safe)
        self._completion_events: Dict[str, threading.Event] = {}

        # Running rollouts cache, including preparing and running rollouts
        self._running_rollout_ids: Set[str] = set()

        # Caches the latest resources ID.
        self._latest_resources_id: Union[str, None, Unset] = UNSET

    @property
    def capabilities(self) -> LightningStoreCapabilities:
        """Return the capabilities of the store."""
        return LightningStoreCapabilities(
            thread_safe=False,
            async_safe=True,
            zero_copy=False,
            otlp_traces=False,
        )

    async def wait_for_rollout(self, rollout_id: str, timeout: Optional[float] = None) -> Optional[Rollout]:
        """Wait for a specific rollout to complete with a timeout."""
        async with self.collections.atomic() as collections:
            rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
            if rollout and is_finished(rollout):
                return rollout

        if timeout is not None and timeout <= 0:
            return None

        # If not completed and we have an event, wait for completion
        if rollout_id in self._completion_events:
            evt = self._completion_events[rollout_id]

            # Wait for the event with proper timeout handling
            # evt.wait() returns True if event was set, False if timeout occurred
            if timeout is None:
                # Wait indefinitely by polling with finite timeouts
                # This allows threads to exit cleanly on shutdown
                while True:
                    result = await asyncio.to_thread(evt.wait, 10.0)  # Poll every 10 seconds
                    if result:  # Event was set
                        break
                    # Loop and check again (continues indefinitely since timeout=None)
            else:
                # Wait with the specified timeout
                result = await asyncio.to_thread(evt.wait, timeout)

            # If event was set (not timeout), check if rollout is finished
            if result:
                async with self.collections.atomic() as collections:
                    rollout = await collections.rollouts.get({"rollout_id": {"exact": rollout_id}})
                    if rollout and is_finished(rollout):
                        return rollout

        return None

    async def on_rollout_update(self, rollout: Rollout) -> None:
        """Update the running rollout ids set when the rollout updates."""
        if is_running(rollout):
            self._running_rollout_ids.add(rollout.rollout_id)
        else:
            self._running_rollout_ids.discard(rollout.rollout_id)

        if is_finished(rollout):
            self._completion_events.setdefault(rollout.rollout_id, threading.Event())
            self._completion_events[rollout.rollout_id].set()
        else:
            self._completion_events.setdefault(rollout.rollout_id, threading.Event())
        # Rollout status can never transition from finished to running (unlike attempt)
        # so we don't need to clear the completion event even in case of retrying.

        if rollout.rollout_id not in self._start_time_by_rollout:
            self._start_time_by_rollout[rollout.rollout_id] = rollout.start_time

    async def get_running_rollouts(self, collections: InMemoryLightningCollections) -> List[AttemptedRollout]:
        """Accelerated version of `get_running_rollouts` for in-memory store. Used for healthcheck."""
        rollouts = await collections.rollouts.query(filter={"rollout_id": {"within": list(self._running_rollout_ids)}})
        running_rollouts: List[AttemptedRollout] = []
        for rollout in rollouts.items:
            latest_attempt = await collections.attempts.get(
                filter={"rollout_id": {"exact": rollout.rollout_id}},
                sort={"name": "sequence_id", "order": "desc"},
            )
            if not latest_attempt:
                # The rollout is running but has no attempts, this should not happen
                logger.error(f"Rollout {rollout.rollout_id} is running but has no attempts")
                continue
            running_rollouts.append(AttemptedRollout(**rollout.model_dump(), attempt=latest_attempt))
        return running_rollouts

    async def query_spans(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"] | None = None,
        **kwargs: Any,
    ) -> PaginatedResult[Span]:
        if rollout_id in self._evicted_rollout_span_sets:
            raise RuntimeError(f"Spans for rollout {rollout_id} have been evicted")
        return await super().query_spans(rollout_id, attempt_id, **kwargs)

    async def _add_span_unlocked(self, collections: InMemoryLightningCollections, span: Span) -> Span:
        """In-memory store needs to maintain the span data in memory, and evict spans when memory is low."""

        await super()._add_span_unlocked(collections, span)
        self._account_span_size(span)
        await self._maybe_evict_spans(collections)

        return span

    async def _get_latest_resources_id(self, collections: InMemoryLightningCollections) -> Optional[str]:
        if isinstance(self._latest_resources_id, Unset):
            latest_resources = await collections.resources.get(sort={"name": "update_time", "order": "desc"})
            if latest_resources:
                self._latest_resources_id = latest_resources.resources_id
            else:
                self._latest_resources_id = None
        return self._latest_resources_id

    @staticmethod
    def _resolve_memory_threshold(
        value: float | int | None,
        *,
        default_ratio: float,
        capacity_bytes: int,
        name: str,
        minimum: int,
    ) -> int:
        if value is None:
            resolved = int(capacity_bytes * default_ratio)
        elif isinstance(value, float):
            if minimum == 0:
                if not (0 <= value <= 1):
                    raise ValueError(f"{name} ratio must be between 0 and 1 inclusive")
            else:
                if not (0 < value <= 1):
                    raise ValueError(f"{name} ratio must be greater than 0 and at most 1")
            resolved = int(capacity_bytes * value)
        else:
            value_int = value
            if value_int < 0:
                raise ValueError(f"{name} must be non-negative")
            resolved = value_int

        if resolved < minimum:
            raise ValueError(f"{name} must be at least {minimum} bytes")

        return resolved

    def _account_span_size(self, span: Span) -> int:
        if self._custom_span_size_estimator is not None:
            size = max(int(self._custom_span_size_estimator(span)), 0)
        else:
            size = estimate_model_size(span)

        self._span_bytes_by_rollout[span.rollout_id] += size
        self._total_span_bytes += size
        return size

    async def _maybe_evict_spans(self, collections: InMemoryLightningCollections) -> None:
        if self._total_span_bytes <= self._eviction_threshold_bytes:
            return

        logger.info(
            f"Total span bytes: {self._total_span_bytes}, eviction threshold: {self._eviction_threshold_bytes}, "
            f"safe threshold: {self._safe_threshold_bytes}. Evicting spans..."
        )
        candidates: List[tuple[float, str]] = [
            (start_time, rollout_id) for rollout_id, start_time in self._start_time_by_rollout.items()
        ]
        candidates.sort()

        logger.info(f"Evicting spans for {len(candidates)} rollouts to free up memory...")
        memory_consumed_before = self._total_span_bytes
        for _, rollout_id in candidates:
            if self._total_span_bytes <= self._safe_threshold_bytes:
                break
            logger.debug(f"Evicting spans for rollout {rollout_id} to free up memory...")
            await self._evict_spans_for_rollout(collections, rollout_id)
        logger.info(f"Freed up {memory_consumed_before - self._total_span_bytes} bytes of memory")

    async def _evict_spans_for_rollout(self, collections: InMemoryLightningCollections, rollout_id: str) -> None:
        await collections.evict_spans_for_rollout(rollout_id)
        removed_bytes = self._span_bytes_by_rollout.pop(rollout_id, 0)
        if removed_bytes > 0:
            # There is something removed for real
            self._total_span_bytes = max(self._total_span_bytes - removed_bytes, 0)
            self._evicted_rollout_span_sets.add(rollout_id)
