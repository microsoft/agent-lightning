# Copyright (c) Microsoft. All rights reserved.

"""Metrics abstraction with explicit registration and several backends.

It provides:

- MetricsBackend: Abstract interface for registering and recording metrics.
- ConsoleMetricsBackend: In-process backend with sliding-window
  aggregations (rate, P50, P95, P99) logged to stdout.
- PrometheusMetricsBackend: Thin wrapper around prometheus_client.
- MultiMetricsBackend: Fan-out backend that forwards calls to multiple underlying backends.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import aiologic

if TYPE_CHECKING:
    from prometheus_client import CollectorRegistry

LabelDict = Dict[str, str]
# Label metadata
LabelKey = Tuple[Tuple[str, str], ...]  # normalized (key, value) pairs in registration order

logger = logging.getLogger(__name__)


def _validate_labels(
    kind: str,
    metric_name: str,
    labels: LabelDict,
    expected_names: Tuple[str, ...],
) -> LabelKey:
    """Validates label keys against the metric definition.

    Args:
        kind: Metric kind for error messages ("counter" or "histogram").
        metric_name: Metric name.
        labels: Provided label dictionary.
        expected_names: Expected label names as a tuple.

    Returns:
        A tuple of (key, value) pairs honoring the registered label order.

    Raises:
        ValueError: If label keys do not match expected_names.
    """

    label_items: List[Tuple[str, str]] = []
    for label_name in expected_names:
        if label_name not in labels:
            raise ValueError(f"Label '{label_name}' is required for {kind.capitalize()} '{metric_name}'.")
        label_items.append((label_name, labels[label_name]))

    return tuple(label_items)


def _normalize_label_names(label_names: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Normalizes label names into a canonical tuple.

    Args:
        label_names: Iterable of label names or None.

    Returns:
        A tuple of label names preserving their original order.
    """
    if not label_names:
        return ()
    return tuple(label_names)


def _normalize_prometheus_metric_name(metric_name: str) -> str:
    """Normalizes Prometheus metric names by replacing unsupported characters."""

    return metric_name.replace(".", "_")


@dataclass(frozen=True)
class _CounterDef:
    """Definition of a registered counter metric."""

    name: str
    label_names: Tuple[str, ...]
    group_level: Optional[int] = None


@dataclass(frozen=True)
class _HistogramDef:
    """Definition of a registered histogram metric."""

    name: str
    label_names: Tuple[str, ...]
    buckets: Tuple[float, ...]
    group_level: Optional[int] = None


@dataclass
class _CounterState:
    """Runtime state of a counter metric group (for console backend)."""

    timestamps: List[float]
    amounts: List[float]


@dataclass
class _HistogramState:
    """Runtime state of a histogram metric group (for console backend)."""

    timestamps: List[float]
    values: List[float]


class MetricsBackend:
    """Abstract base class for metrics backends."""

    def has_prometheus(self) -> bool:
        """Check if the backend has prometheus support."""
        return False

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a counter metric.

        Args:
            name: Metric name.
            label_names: List of label names. Order determines the truncation
                priority for group-level logging.
            group_level: Optional per-metric grouping depth for backends that
                support label grouping (Console). Global backend settings take
                precedence when provided.

        Raises:
            ValueError: If the metric is already registered with a different
                type or label set.
        """
        raise NotImplementedError()

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a histogram metric.

        Args:
            name: Metric name.
            label_names: List of label names. Order determines the truncation
                priority for group-level logging.
            buckets: Bucket boundaries (exclusive upper bounds). If None, the
                backend may choose defaults.
            group_level: Optional per-metric grouping depth for backends that
                support label grouping (Console). Global backend settings take
                precedence when provided.

        Raises:
            ValueError: If the metric is already registered with a different
                type or label set.
        """
        raise NotImplementedError()

    async def inc_counter(
        self,
        name: str,
        amount: float = 1.0,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Increments a registered counter.

        Args:
            name: Metric name (must be registered as a counter).
            amount: Increment amount.
            labels: Label values.

        Raises:
            ValueError: If the metric is not registered, has the wrong type,
                or label keys do not match the registered label names.
        """
        raise NotImplementedError()

    async def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Records an observation for a registered histogram.

        Args:
            name: Metric name (must be registered as a histogram).
            value: Observed value.
            labels: Label values.

        Raises:
            ValueError: If the metric is not registered, has the wrong type,
                or label keys do not match the registered label names.
        """
        raise NotImplementedError()


class ConsoleMetricsBackend(MetricsBackend):
    """Console backend with sliding-window aggregations and label grouping.

    This backend:

    * Requires explicit metric registration.
    * Stores timestamped events per (metric_name, labels) key.
    * Computes rate and percentiles (P50, P95, P99) over a sliding time window.
    * Uses a single global logging decision: when logging is triggered, it
      logs all metric groups, not just the one being updated.

    Rate is always per second.

    Label grouping: When logging, label dictionaries are truncated to the first
    `group_level` label pairs (following the registered label order) and metrics
    with identical truncated labels are aggregated together. For example:

    ```python
    labels = {"method": "GET", "path": "/", "status": "200"}
    group_level = 2  # aggregated labels {"method": "GET", "path": "/"}
    ```

    If `group_level` is None or < 1, all label combinations for a metric are
    merged into a single log entry (equivalent to grouping by zero labels).
    Individual counters or histograms can set their own `group_level` during
    registration; those values apply only when the backend-level `group_level`
    is unset, allowing selective overrides.

    Thread-safety: Runtime updates and snapshotting use two aiologic locks: one for mutating
    shared state and another that serializes the global logging decision/snapshot capture so
    other tasks can continue writing. Metric registration happens during initialization,
    so it is intentionally left lock-free; this assumption is documented here to avoid
    blocking writes unnecessarily.
    """

    def __init__(
        self,
        window_seconds: Optional[float] = 60.0,
        log_interval_seconds: float = 10.0,
        group_level: Optional[int] = None,
    ) -> None:
        """Initializes ConsoleMetricsBackend.

        Args:
            window_seconds: Sliding window size (in seconds) used when computing
                rate and percentiles. If None, all in-memory events are used.
            log_interval_seconds: Minimum time (in seconds) between log bursts.
                When the interval elapses, the next metric event triggers a
                snapshot and logging of all metrics.
            group_level: Label grouping depth. When logging, only the first
                `group_level` labels (following registered order) are retained and metric
                events sharing those labels are aggregated. If None or < 1,
                all label combinations collapse into a single group per metric.
        """
        self.window_seconds = window_seconds
        self.log_interval_seconds = log_interval_seconds
        self.group_level = group_level

        self._counters: Dict[str, _CounterDef] = {}
        self._histograms: Dict[str, _HistogramDef] = {}

        # Runtime state keyed by (metric_name, label_key)
        self._counter_state: Dict[Tuple[str, LabelKey], _CounterState] = {}
        self._hist_state: Dict[Tuple[str, LabelKey], _HistogramState] = {}

        # Global last log time (for all metrics)
        self._last_log_time: Optional[float] = None

        self._write_lock = aiologic.Lock()
        self._snapshot_lock = aiologic.Lock()

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a counter metric.

        See base class for argument documentation.
        """
        label_tuple = _normalize_label_names(label_names)
        existing_counter = self._counters.get(name)
        existing_hist = self._histograms.get(name)

        if existing_hist is not None:
            raise ValueError(f"Metric '{name}' already registered as histogram.")

        if existing_counter is not None:
            if existing_counter.label_names != label_tuple:
                raise ValueError(
                    f"Counter '{name}' already registered with labels "
                    f"{existing_counter.label_names}, got {label_tuple}."
                )
            return

        self._counters[name] = _CounterDef(name=name, label_names=label_tuple, group_level=group_level)

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a histogram metric.

        See base class for argument documentation.
        """
        label_tuple = _normalize_label_names(label_names)
        if buckets is None:
            bucket_tuple: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
        else:
            bucket_tuple = tuple(buckets)

        existing_counter = self._counters.get(name)
        existing_hist = self._histograms.get(name)

        if existing_counter is not None:
            raise ValueError(f"Metric '{name}' already registered as counter.")

        if existing_hist is not None:
            if existing_hist.label_names != label_tuple or existing_hist.buckets != bucket_tuple:
                raise ValueError(
                    f"Histogram '{name}' already registered with "
                    f"labels={existing_hist.label_names}, "
                    f"buckets={existing_hist.buckets}."
                )
            return

        self._histograms[name] = _HistogramDef(
            name=name,
            label_names=label_tuple,
            buckets=bucket_tuple,
            group_level=group_level,
        )

    async def inc_counter(
        self,
        name: str,
        amount: float = 1.0,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Increments a registered counter metric.

        See base class for behavior and error conditions.
        """
        now = time.time()
        labels = labels or {}

        definition = self._counters.get(name)
        if definition is None:
            raise ValueError(f"Counter '{name}' is not registered.")

        label_key = _validate_labels("counter", name, labels, definition.label_names)
        state_key = (name, label_key)

        async with self._write_lock:
            state = self._counter_state.get(state_key)
            if state is None:
                state = _CounterState(timestamps=[], amounts=[])
                self._counter_state[state_key] = state

            state.timestamps.append(now)
            state.amounts.append(amount)
            self._prune_events(state.timestamps, state.amounts, now)

        counter_snaps: List[Tuple[str, LabelDict, List[float], List[float]]] = []
        hist_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]] = []
        should_log = False
        snapshot_time = now

        async with self._snapshot_lock:
            should_log = self._should_log_locked(now)
        if should_log:
            async with self._write_lock:
                counter_snaps, hist_snaps = self._snapshot_locked(now)
            self._log_snapshot(counter_snaps, hist_snaps, snapshot_time)

    async def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Records an observation for a registered histogram metric.

        See base class for behavior and error conditions.
        """
        now = time.time()
        labels = labels or {}

        definition = self._histograms.get(name)
        if definition is None:
            raise ValueError(f"Histogram '{name}' is not registered.")

        label_key = _validate_labels("histogram", name, labels, definition.label_names)
        state_key = (name, label_key)

        async with self._write_lock:
            state = self._hist_state.get(state_key)
            if state is None:
                state = _HistogramState(timestamps=[], values=[])
                self._hist_state[state_key] = state

            state.timestamps.append(now)
            state.values.append(value)
            self._prune_events(state.timestamps, state.values, now)

        counter_snaps: List[Tuple[str, LabelDict, List[float], List[float]]] = []
        hist_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]] = []
        should_log = False
        snapshot_time = now

        async with self._snapshot_lock:
            should_log = self._should_log_locked(now)

        if should_log:
            async with self._write_lock:
                counter_snaps, hist_snaps = self._snapshot_locked(now)
            self._log_snapshot(counter_snaps, hist_snaps, snapshot_time)

    def _prune_events(
        self,
        timestamps: List[float],
        values: List[float],
        now: float,
    ) -> None:
        """Prunes events older than the sliding window.

        Args:
            timestamps: List of event timestamps (ascending).
            values: List of corresponding values or amounts.
            now: Current time.
        """
        if self.window_seconds is None or not timestamps:
            return
        cutoff = now - self.window_seconds
        idx = 0
        for i, ts in enumerate(timestamps):
            if ts >= cutoff:
                idx = i
                break
        else:
            idx = len(timestamps)
        if idx > 0:
            del timestamps[:idx]
            del values[:idx]

    def _should_log_locked(self, now: float) -> bool:
        """Determines whether to emit a log snapshot (lock must be held).

        This decision is global: if it returns True, all metrics will be
        logged based on a snapshot taken at this time.

        Args:
            now: Current timestamp.

        Returns:
            True if enough time has elapsed since the last log; False otherwise.
        """
        last = self._last_log_time
        if last is None or now - last >= self.log_interval_seconds:
            self._last_log_time = now
            return True
        return False

    def _snapshot_locked(
        self,
        now: float,
    ) -> Tuple[
        List[Tuple[str, LabelDict, List[float], List[float]]],
        List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]],
    ]:
        """Creates a snapshot of all metric state (lock must be held).

        Args:
            now: Current timestamp.

        Returns:
            A tuple (counter_snapshots, histogram_snapshots) where:
              - counter_snapshots: list of (metric_name, labels, timestamps, amounts)
              - histogram_snapshots: list of (metric_name, labels, values, buckets)
        """
        counter_snaps: List[Tuple[str, LabelDict, List[float], List[float]]] = []
        hist_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]] = []

        # Prune and snapshot counters.
        for (name, label_key), state in self._counter_state.items():
            self._prune_events(state.timestamps, state.amounts, now)
            if not state.timestamps:
                continue
            labels = dict(label_key)
            counter_snaps.append(
                (
                    name,
                    labels,
                    list(state.timestamps),
                    list(state.amounts),
                )
            )

        # Prune and snapshot histograms.
        for (name, label_key), state in self._hist_state.items():
            self._prune_events(state.timestamps, state.values, now)
            if not state.values:
                continue
            labels = dict(label_key)
            buckets = self._histograms[name].buckets
            hist_snaps.append(
                (
                    name,
                    labels,
                    list(state.values),
                    buckets,
                )
            )

        return counter_snaps, hist_snaps

    def _truncate_labels_for_logging(self, labels: LabelDict, group_level: Optional[int]) -> LabelDict:
        """Returns a label dict truncated to the configured group depth.

        Args:
            labels: Original label dictionary.
            group_level: Effective grouping depth for this metric.

        Returns:
            A new dictionary containing at most `group_level` label pairs,
            chosen by registered label order. If group_level is None or < 1,
            returns an empty dict so that all label combinations collapse together.
        """
        if group_level is None or group_level < 1:
            return {}
        items = list(labels.items())
        return dict(items[:group_level])

    def _log(self, message: str) -> None:
        """Logs a message via the module logger."""
        logger.info(message)

    def _log_snapshot(
        self,
        counter_snaps: List[Tuple[str, LabelDict, List[float], List[float]]],
        hist_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]],
        snapshot_time: float,
    ) -> None:
        """Logs all metrics from a snapshot.

        Args:
            counter_snaps: Counter snapshot list.
            hist_snaps: Histogram snapshot list.
        """
        entries: List[str] = []
        for name, labels, timestamps, amounts in self._group_counter_snapshots(counter_snaps):
            line = self._log_counter(name, labels, timestamps, amounts, snapshot_time)
            if line:
                entries.append(line)

        for name, labels, values, buckets in self._group_histogram_snapshots(hist_snaps):
            line = self._log_histogram(name, labels, values, buckets, snapshot_time)
            if line:
                entries.append(line)

        if entries:
            entries.sort()
            self._log("  ".join(entries))

    def _effective_group_level(self, metric_name: str, *, is_histogram: bool) -> Optional[int]:
        """Returns the active group level for a metric, honoring per-metric overrides."""
        if self.group_level is not None:
            return self.group_level
        if is_histogram:
            definition = self._histograms.get(metric_name)
        else:
            definition = self._counters.get(metric_name)
        if definition is None:
            return None
        return definition.group_level

    def _group_counter_snapshots(
        self,
        counter_snaps: List[Tuple[str, LabelDict, List[float], List[float]]],
    ) -> List[Tuple[str, LabelDict, List[float], List[float]]]:
        grouped: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
        for name, labels, timestamps, amounts in counter_snaps:
            group_level = self._effective_group_level(name, is_histogram=False)
            truncated_labels = self._truncate_labels_for_logging(labels, group_level)
            key = (name, tuple(truncated_labels.items()))
            entry = grouped.setdefault(
                key,
                {"name": name, "labels": truncated_labels, "timestamps": [], "amounts": []},
            )
            entry["timestamps"].extend(timestamps)
            entry["amounts"].extend(amounts)

        grouped_snaps: List[Tuple[str, LabelDict, List[float], List[float]]] = []
        for entry in grouped.values():
            timestamps = entry["timestamps"]
            amounts = entry["amounts"]
            if not timestamps:
                continue
            combined = sorted(zip(timestamps, amounts), key=lambda item: item[0])
            ordered_timestamps = [ts for ts, _ in combined]
            ordered_amounts = [amt for _, amt in combined]
            grouped_snaps.append(
                (
                    entry["name"],
                    entry["labels"],
                    ordered_timestamps,
                    ordered_amounts,
                )
            )

        return grouped_snaps

    def _group_histogram_snapshots(
        self,
        hist_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]],
    ) -> List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]]:
        grouped: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Dict[str, Any]] = {}
        for name, labels, values, buckets in hist_snaps:
            group_level = self._effective_group_level(name, is_histogram=True)
            truncated_labels = self._truncate_labels_for_logging(labels, group_level)
            key = (name, tuple(truncated_labels.items()))
            entry = grouped.setdefault(
                key,
                {"name": name, "labels": truncated_labels, "values": [], "buckets": buckets},
            )
            if entry["buckets"] != buckets:
                raise ValueError(f"Histogram buckets mismatch for metric '{name}'.")
            entry["values"].extend(values)

        grouped_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]] = []
        for entry in grouped.values():
            values = entry["values"]
            if not values:
                continue
            grouped_snaps.append(
                (
                    entry["name"],
                    entry["labels"],
                    list(values),
                    entry["buckets"],
                )
            )

        return grouped_snaps

    def _log_counter(
        self,
        name: str,
        labels: LabelDict,
        timestamps: List[float],
        amounts: List[float],
        snapshot_time: float,
    ) -> Optional[str]:
        """Computes counter stats and returns formatted line."""
        if not timestamps:
            return None

        total = sum(amounts)
        window_start = timestamps[0]
        if self.window_seconds is not None:
            window_start = max(window_start, snapshot_time - self.window_seconds)
        min_duration = self.log_interval_seconds if self.log_interval_seconds > 0 else 1e-3
        duration = max(snapshot_time - window_start, min_duration)
        rate = total / duration

        label_str = _format_label_string(labels)
        return f"{name}{label_str}={rate:.2f}/s"

    def _log_histogram(
        self,
        name: str,
        labels: LabelDict,
        values: List[float],
        buckets: Tuple[float, ...],
        snapshot_time: float,
    ) -> Optional[str]:
        """Computes histogram stats and returns formatted line."""
        if not values:
            return None

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        def percentile(p: float) -> float:
            if n == 1:
                return sorted_vals[0]
            pos = (p / 100.0) * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            if lo == hi:
                return sorted_vals[lo]
            w = pos - lo
            return sorted_vals[lo] * (1 - w) + sorted_vals[hi] * w

        p50 = percentile(50.0)
        p95 = percentile(95.0)
        p99 = percentile(99.0)

        label_str = _format_label_string(labels)
        formatted = ",".join([_format_duration(p50), _format_duration(p95), _format_duration(p99)])
        return f"{name}{label_str}={formatted}"


def _format_label_string(labels: LabelDict) -> str:
    if not labels:
        return ""
    ordered = ",".join(f"{key}={value}" for key, value in labels.items())
    return f"{{{ordered}}}"


def _format_duration(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1.0:
        return f"{value:.2f}s"
    if abs_value >= 1e-3:
        return f"{value * 1_000:.2f}ms"
    if abs_value >= 1e-6:
        return f"{value * 1_000_000:.2f}µs"
    return f"{value * 1_000_000_000:.2f}ns"


class PrometheusMetricsBackend(MetricsBackend):
    """Metrics backend that forwards events to prometheus_client.

    All metrics must be registered before use. This backend does not compute
    any aggregations; it only updates Prometheus metrics.

    Thread-safety: Registration is protected by a lock. Metric updates assume metrics
    are registered during initialization and then remain stable.

    Due to the nature of Prometheus, this backend is only suitable for recording high-volume metrics.
    Low-volume metrics might be lost if the event has only appeared once.
    """

    def __init__(self) -> None:
        """Initializes PrometheusMetricsBackend.

        Raises:
            ImportError: If prometheus_client is not installed.
        """
        try:
            import prometheus_client  # type: ignore
        except ImportError:
            raise ImportError(
                "prometheus_client is not installed. Please either install it or use ConsoleMetricsBackend instead."
            )

        self._counters: Dict[str, _CounterDef] = {}
        self._histograms: Dict[str, _HistogramDef] = {}
        self._prom_counters: Dict[str, Any] = {}
        self._prom_histograms: Dict[str, Any] = {}
        self._prom_metric_names: Dict[str, str] = {}

    def has_prometheus(self) -> bool:
        """Check if the backend has prometheus support."""
        return True

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a Prometheus counter metric."""
        from prometheus_client import Counter as PromCounter

        label_tuple = _normalize_label_names(label_names)

        if name in self._histograms:
            raise ValueError(f"Metric '{name}' already registered as histogram.")

        existing = self._counters.get(name)
        if existing is not None:
            if existing.label_names != label_tuple:
                raise ValueError(
                    f"Counter '{name}' already registered with labels " f"{existing.label_names}, got {label_tuple}."
                )
            return

        prom_name = self._register_prometheus_metric_name(name)
        self._counters[name] = _CounterDef(name=name, label_names=label_tuple, group_level=group_level)

        prom_counter = PromCounter(
            prom_name,
            f"Counter {name}",
            labelnames=label_tuple,
        )
        self._prom_counters[name] = prom_counter

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a Prometheus histogram metric."""
        from prometheus_client import Histogram as PromHistogram

        label_tuple = _normalize_label_names(label_names)
        bucket_tuple = tuple(buckets) if buckets is not None else ()

        if name in self._counters:
            raise ValueError(f"Metric '{name}' already registered as counter.")

        existing = self._histograms.get(name)
        if existing is not None:
            if existing.label_names != label_tuple or existing.buckets != bucket_tuple:
                raise ValueError(
                    f"Histogram '{name}' already registered with "
                    f"labels={existing.label_names}, "
                    f"buckets={existing.buckets}."
                )
            return

        prom_name = self._register_prometheus_metric_name(name)
        self._histograms[name] = _HistogramDef(
            name=name,
            label_names=label_tuple,
            buckets=bucket_tuple,
            group_level=group_level,
        )

        if bucket_tuple:
            prom_hist = PromHistogram(
                prom_name,
                f"Histogram {name}",
                labelnames=label_tuple,
                buckets=bucket_tuple,
            )
        else:
            prom_hist = PromHistogram(
                prom_name,
                f"Histogram {name}",
                labelnames=label_tuple,
            )

        self._prom_histograms[name] = prom_hist

    async def inc_counter(
        self,
        name: str,
        amount: float = 1.0,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Increments a registered Prometheus counter."""
        labels = labels or {}
        definition = self._counters.get(name)
        if definition is None:
            raise ValueError(f"Counter '{name}' is not registered.")

        prom_counter = self._prom_counters[name]
        if definition.label_names:
            label_key = _validate_labels("counter", name, labels, definition.label_names)
            prom_counter.labels(**dict(label_key)).inc(amount)
        else:
            prom_counter.inc(amount)

    async def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Records an observation for a registered Prometheus histogram."""
        labels = labels or {}
        definition = self._histograms.get(name)
        if definition is None:
            raise ValueError(f"Histogram '{name}' is not registered.")

        prom_hist = self._prom_histograms[name]
        if definition.label_names:
            label_key = _validate_labels("histogram", name, labels, definition.label_names)
            prom_hist.labels(**dict(label_key)).observe(value)
        else:
            prom_hist.observe(value)

    def _register_prometheus_metric_name(self, name: str) -> str:
        """Registers the normalized Prometheus metric name and ensures uniqueness."""

        normalized = _normalize_prometheus_metric_name(name)
        existing = self._prom_metric_names.get(normalized)
        if existing is not None and existing != name:
            raise ValueError(
                f"Prometheus metric name conflict: '{name}' normalizes to '{normalized}', "
                f"which is already used by '{existing}'. Consider renaming one of the metrics."
            )
        self._prom_metric_names.setdefault(normalized, name)
        return normalized


class MultiMetricsBackend(MetricsBackend):
    """Metrics backend that forwards calls to multiple underlying backends."""

    def __init__(self, backends: Sequence[MetricsBackend]) -> None:
        """Initializes MultiMetricsBackend.

        Args:
            backends: Sequence of underlying backends.

        Raises:
            ValueError: If no backends are provided.
        """
        if not backends:
            raise ValueError("MultiMetricsBackend requires at least one backend.")
        self._backends = list(backends)

    def has_prometheus(self) -> bool:
        """Check if the backend has prometheus support."""
        return any(backend.has_prometheus() for backend in self._backends)

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a counter metric in all underlying backends."""
        for backend in self._backends:
            backend.register_counter(name, label_names=label_names, group_level=group_level)

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        """Registers a histogram metric in all underlying backends."""
        for backend in self._backends:
            backend.register_histogram(
                name,
                label_names=label_names,
                buckets=buckets,
                group_level=group_level,
            )

    async def inc_counter(
        self,
        name: str,
        amount: float = 1.0,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Increments a counter metric in all underlying backends."""
        for backend in self._backends:
            await backend.inc_counter(name, amount=amount, labels=labels)

    async def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Records a histogram observation in all underlying backends."""
        for backend in self._backends:
            await backend.observe_histogram(name, value=value, labels=labels)


# This variable should be carried into forked processes
_prometheus_multiproc_dir: tempfile.TemporaryDirectory[str] | None = None


def setup_multiprocess_prometheus():
    """Set up prometheus multiprocessing directory if not already configured."""

    global _prometheus_multiproc_dir

    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        # Make TemporaryDirectory for prometheus multiprocessing
        # Note: global TemporaryDirectory will be automatically
        # cleaned up upon exit.
        _prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir.name
        logger.debug("Created PROMETHEUS_MULTIPROC_DIR at %s", _prometheus_multiproc_dir.name)
    else:
        logger.warning(
            "Found PROMETHEUS_MULTIPROC_DIR was set by user. This directory must be wiped between multiple runs."
        )


def get_prometheus_registry() -> CollectorRegistry:
    """Get the appropriate prometheus registry based on multiprocessing configuration."""
    from prometheus_client import REGISTRY, CollectorRegistry, multiprocess

    if os.getenv("PROMETHEUS_MULTIPROC_DIR") is not None:
        logger.info("Using multiprocess registry for prometheus metrics: %s", os.getenv("PROMETHEUS_MULTIPROC_DIR"))
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry

    return REGISTRY


def shutdown_metrics(server: Any = None, worker: Any = None, *args: Any, **kwargs: Any) -> None:
    """Shutdown prometheus metrics."""

    if _prometheus_multiproc_dir is not None:
        from prometheus_client import multiprocess

        path = _prometheus_multiproc_dir
        try:
            if hasattr(worker, "pid"):
                pid = worker.pid
            else:
                pid = os.getpid()
            multiprocess.mark_process_dead(pid, path.name)  # type: ignore
            logger.debug("Marked Prometheus metrics for process %d as dead", pid)
        except Exception as e:
            logger.error("Error during metrics cleanup: %s", str(e))
