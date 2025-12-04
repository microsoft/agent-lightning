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

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

LabelDict = Dict[str, str]
LabelKey = Tuple[Tuple[str, str], ...]  # normalized, sorted (key, value) pairs


def _validate_labels(
    kind: str,
    name: str,
    labels: LabelDict,
    expected_names: Tuple[str, ...],
) -> LabelKey:
    """Validates label keys against the metric definition.

    Args:
        kind: Metric kind for error messages ("counter" or "histogram").
        name: Metric name.
        labels: Provided label dictionary.
        expected_names: Expected label names as a tuple.

    Returns:
        A tuple of (key, value) pairs sorted by registered label order.

    Raises:
        ValueError: If label keys do not match expected_names.
    """

    label_items: List[Tuple[str, str]] = []
    for name in expected_names:
        if name not in labels:
            raise ValueError(f"Label '{name}' is required for {kind.capitalize()} '{name}'.")
        label_items.append((name, labels[name]))

    return tuple(label_items)


def _normalize_label_names(label_names: Optional[Sequence[str]]) -> Tuple[str, ...]:
    """Normalizes label names into a canonical tuple.

    Not sorted. Order is the insertion order.

    Args:
        label_names: Iterable of label names or None.

    Returns:
        A tuple of label names.
    """
    if not label_names:
        return ()
    return tuple(sorted(label_names))


@dataclass(frozen=True)
class _CounterDef:
    """Definition of a registered counter metric."""

    name: str
    label_names: Tuple[str, ...]


@dataclass(frozen=True)
class _HistogramDef:
    """Definition of a registered histogram metric."""

    name: str
    label_names: Tuple[str, ...]
    buckets: Tuple[float, ...]


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

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Registers a counter metric.

        Args:
            name: Metric name.
            label_names: List of label names. Order is not important.

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
    ) -> None:
        """Registers a histogram metric.

        Args:
            name: Metric name.
            label_names: List of label names. Order is not important.
            buckets: Bucket boundaries (exclusive upper bounds). If None, the
                backend may choose defaults.

        Raises:
            ValueError: If the metric is already registered with a different
                type or label set.
        """
        raise NotImplementedError()

    def inc_counter(
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

    def observe_histogram(
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

    Label grouping: When logging, labels are truncated to the first `group_level` label
    pairs (according to sorted label key order). For example:

        labels = {"method": "GET", "path": "/", "status": "200"}
        group_level = 2 -> logged labels {"method": "GET", "path": "/"}

    If `group_level` is None or < 1, all labels are logged.

    Thread-safety: A single lock protects shared state mutation, pruning, and snapshotting.
    Percentile computation, formatting, and printing are done after releasing the lock.
    """

    def __init__(
        self,
        window_seconds: Optional[float] = 60.0,
        log_interval_seconds: float = 5.0,
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
                `group_level` labels (sorted by key) are included. If None or
                < 1, all labels are included.
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

        self._lock = threading.Lock()

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Registers a counter metric.

        See base class for argument documentation.
        """
        label_tuple = _normalize_label_names(label_names)
        with self._lock:
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

            self._counters[name] = _CounterDef(name=name, label_names=label_tuple)

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
    ) -> None:
        """Registers a histogram metric.

        See base class for argument documentation.
        """
        label_tuple = _normalize_label_names(label_names)
        if buckets is None:
            bucket_tuple: Tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0)
        else:
            bucket_tuple = tuple(buckets)

        with self._lock:
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
            )

    def inc_counter(
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

        with self._lock:
            state = self._counter_state.get(state_key)
            if state is None:
                state = _CounterState(timestamps=[], amounts=[])
                self._counter_state[state_key] = state

            state.timestamps.append(now)
            state.amounts.append(amount)
            self._prune_events(state.timestamps, state.amounts, now)

            should_log = self._should_log_locked(now)
            if should_log:
                counter_snaps, hist_snaps = self._snapshot_locked(now)
            else:
                counter_snaps = hist_snaps = None

        if should_log and (counter_snaps or hist_snaps):
            self._log_snapshot(counter_snaps, hist_snaps)

    def observe_histogram(
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

        with self._lock:
            state = self._hist_state.get(state_key)
            if state is None:
                state = _HistogramState(timestamps=[], values=[])
                self._hist_state[state_key] = state

            state.timestamps.append(now)
            state.values.append(value)
            self._prune_events(state.timestamps, state.values, now)

            should_log = self._should_log_locked(now)
            if should_log:
                counter_snaps, hist_snaps = self._snapshot_locked(now)
            else:
                counter_snaps = hist_snaps = None

        if should_log and (counter_snaps or hist_snaps):
            self._log_snapshot(counter_snaps, hist_snaps)

    # Helpers --------------------------------------------------------------

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

    def _truncate_labels_for_logging(self, labels: LabelDict) -> LabelDict:
        """Returns a label dict truncated to the configured group depth.

        Args:
            labels: Original label dictionary.

        Returns:
            A new dictionary containing at most `group_level` label pairs,
            chosen by sorted key order. If group_level is None or < 1, returns
            a shallow copy of the original labels.
        """
        if self.group_level is None or self.group_level < 1:
            return dict(labels)
        items = sorted(labels.items())
        return dict(items[: self.group_level])

    def _log(self, message: str) -> None:
        """Logs a message to stdout."""
        print(message)

    def _log_snapshot(
        self,
        counter_snaps: List[Tuple[str, LabelDict, List[float], List[float]]],
        hist_snaps: List[Tuple[str, LabelDict, List[float], Tuple[float, ...]]],
    ) -> None:
        """Logs all metrics from a snapshot.

        Args:
            counter_snaps: Counter snapshot list.
            hist_snaps: Histogram snapshot list.
        """
        for name, labels, timestamps, amounts in counter_snaps:
            truncated_labels = self._truncate_labels_for_logging(labels)
            self._log_counter(name, truncated_labels, timestamps, amounts)

        for name, labels, values, buckets in hist_snaps:
            truncated_labels = self._truncate_labels_for_logging(labels)
            self._log_histogram(name, truncated_labels, values, buckets)

    def _log_counter(
        self,
        name: str,
        labels: LabelDict,
        timestamps: List[float],
        amounts: List[float],
    ) -> None:
        """Computes and logs counter statistics for a single group."""
        if not timestamps:
            return

        total = sum(amounts)
        first_ts = timestamps[0]
        last_ts = timestamps[-1]
        duration = max(last_ts - first_ts, 1e-9)
        rate = total / duration

        window_description = f"last {self.window_seconds:.1f}s" if self.window_seconds is not None else "all time"

        self._log(
            f"[metrics][counter] {name}{labels} " f"total={total:.2f} rate={rate:.4f}/s " f"({window_description})"
        )

    def _log_histogram(
        self,
        name: str,
        labels: LabelDict,
        values: List[float],
        buckets: Tuple[float, ...],
    ) -> None:
        """Computes and logs histogram statistics for a single group."""
        if not values:
            return

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

        bucket_str = ""
        if buckets:
            counts = [0] * (len(buckets) + 1)
            for v in sorted_vals:
                placed = False
                for i, b in enumerate(buckets):
                    if v <= b:
                        counts[i] += 1
                        placed = True
                        break
                if not placed:
                    counts[-1] += 1

            parts = []
            for b, c in zip(buckets, counts[:-1]):
                parts.append(f"<= {b}: {c}")
            parts.append(f"> {buckets[-1]}: {counts[-1]}")
            bucket_str = " | buckets: " + ", ".join(parts)

        window_description = f"last {self.window_seconds:.1f}s" if self.window_seconds is not None else "all time"

        self._log(
            f"[metrics][histogram] {name}{labels} "
            f"P50={p50:.4f} P95={p95:.4f} P99={p99:.4f} "
            f"({window_description}){bucket_str}"
        )


class PrometheusMetricsBackend(MetricsBackend):
    """Metrics backend that forwards events to prometheus_client.

    All metrics must be registered before use. This backend does not compute
    any aggregations; it only updates Prometheus metrics.

    Thread-safety: Registration is protected by a lock. Metric updates assume metrics
    are registered during initialization and then remain stable.
    """

    def __init__(self) -> None:
        """Initializes PrometheusMetricsBackend.

        Raises:
            RuntimeError: If prometheus_client is not installed.
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

        self._lock = threading.Lock()

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Registers a Prometheus counter metric."""
        from prometheus_client import Counter as PromCounter

        label_tuple = _normalize_label_names(label_names)

        with self._lock:
            if name in self._histograms:
                raise ValueError(f"Metric '{name}' already registered as histogram.")

            existing = self._counters.get(name)
            if existing is not None:
                if existing.label_names != label_tuple:
                    raise ValueError(
                        f"Counter '{name}' already registered with labels "
                        f"{existing.label_names}, got {label_tuple}."
                    )
                return

            self._counters[name] = _CounterDef(name=name, label_names=label_tuple)

            prom_counter = PromCounter(
                name,
                f"Counter {name}",
                labelnames=label_tuple or None,
            )
            self._prom_counters[name] = prom_counter

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
    ) -> None:
        """Registers a Prometheus histogram metric."""
        from prometheus_client import Histogram as PromHistogram

        label_tuple = _normalize_label_names(label_names)
        bucket_tuple = tuple(buckets) if buckets is not None else ()

        with self._lock:
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

            self._histograms[name] = _HistogramDef(
                name=name,
                label_names=label_tuple,
                buckets=bucket_tuple,
            )

            if bucket_tuple:
                prom_hist = PromHistogram(
                    name,
                    f"Histogram {name}",
                    labelnames=label_tuple or None,
                    buckets=bucket_tuple,
                )
            else:
                prom_hist = PromHistogram(
                    name,
                    f"Histogram {name}",
                    labelnames=label_tuple or None,
                )

            self._prom_histograms[name] = prom_hist

    def inc_counter(
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

    def observe_histogram(
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

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
    ) -> None:
        """Registers a counter metric in all underlying backends."""
        for backend in self._backends:
            backend.register_counter(name, label_names=label_names)

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
    ) -> None:
        """Registers a histogram metric in all underlying backends."""
        for backend in self._backends:
            backend.register_histogram(
                name,
                label_names=label_names,
                buckets=buckets,
            )

    def inc_counter(
        self,
        name: str,
        amount: float = 1.0,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Increments a counter metric in all underlying backends."""
        for backend in self._backends:
            backend.inc_counter(name, amount=amount, labels=labels)

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[LabelDict] = None,
    ) -> None:
        """Records a histogram observation in all underlying backends."""
        for backend in self._backends:
            backend.observe_histogram(name, value=value, labels=labels)
