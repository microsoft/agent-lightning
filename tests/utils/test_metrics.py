# Copyright (c) Microsoft. All rights reserved.

# pyright: reportPrivateUsage=false

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

import agentlightning.utils.metrics as metrics_module
from agentlightning.utils.metrics import ConsoleMetricsBackend, MetricsBackend, MultiMetricsBackend


def test_validate_labels_reports_missing_label_with_metric_name() -> None:
    labels = {"method": "GET", "status": "200"}
    result = metrics_module._validate_labels("counter", "requests", labels, ("method", "status"))
    assert result == (("method", "GET"), ("status", "200"))

    with pytest.raises(ValueError) as excinfo:
        metrics_module._validate_labels("counter", "requests", {"method": "POST"}, ("method", "status"))

    message = str(excinfo.value)
    assert "Counter 'requests'" in message
    assert "'status' is required" in message


def test_normalize_label_names_is_sorted() -> None:
    assert metrics_module._normalize_label_names(["b", "a", "b"]) == ("a", "b", "b")
    assert metrics_module._normalize_label_names(None) == ()


def test_console_backend_logs_counters_with_sorted_labels() -> None:
    backend = ConsoleMetricsBackend()
    line = backend._log_counter(
        "rate",
        {"group2": "b", "group1": "a"},
        timestamps=[0.0, 1.0],
        amounts=[1.0, 1.0],
        snapshot_time=2.0,
    )

    assert line == "rate{group1=a,group2=b}=0.40/s"


def test_console_backend_logs_histograms_with_human_units() -> None:
    backend = ConsoleMetricsBackend()
    line = backend._log_histogram(
        "latency",
        {"group2": "b", "group1": "a"},
        values=[0.00395, 0.0168, 3.5],
        buckets=(0.5,),
        snapshot_time=1.0,
    )

    assert line is not None and line.startswith("latency{group1=a,group2=b}=")
    payload = line.rsplit("=", 1)[1]
    p50, p95, p99 = payload.split(",", 2)
    assert p50.endswith("ms")
    assert p95.endswith("s")
    assert p99.endswith("s")


def test_console_backend_respects_group_level_limit():
    backend = ConsoleMetricsBackend(group_level=2)
    truncated = backend._truncate_labels_for_logging({"group3": "c", "group1": "a", "group2": "b"})
    line = backend._log_counter("metric", truncated, [0.0, 2.0], [1.0, 1.0], snapshot_time=3.0)

    assert line == "metric{group1=a,group2=b}=0.40/s"


def test_console_backend_log_uses_logger(caplog: pytest.LogCaptureFixture) -> None:
    backend = ConsoleMetricsBackend()
    caplog.set_level(logging.INFO, logger="agentlightning.utils.metrics")
    backend._log("hello metrics")
    assert any(record.message == "hello metrics" for record in caplog.records)


class _RecordingBackend(MetricsBackend):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def register_counter(self, name: str, label_names: Optional[Sequence[str]] = None) -> None:
        self.calls.append(("register_counter", (name, tuple(label_names or ()))))

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
    ) -> None:
        self.calls.append(("register_histogram", (name, tuple(label_names or ()), tuple(buckets or ()))))

    def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        self.calls.append(("inc_counter", (name, amount, labels or {})))

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self.calls.append(("observe_histogram", (name, value, labels or {})))


def test_multi_metrics_backend_fans_out_calls() -> None:
    backend_a = _RecordingBackend()
    backend_b = _RecordingBackend()
    multi = MultiMetricsBackend([backend_a, backend_b])

    multi.register_counter("hits", label_names=["method"])
    multi.register_histogram("latency", label_names=["method"], buckets=[0.1, 1.0])
    multi.inc_counter("hits", amount=2.5, labels={"method": "GET"})
    multi.observe_histogram("latency", value=0.4, labels={"method": "GET"})

    expected_calls: List[Tuple[str, Tuple[Any, ...]]] = [
        ("register_counter", ("hits", ("method",))),
        ("register_histogram", ("latency", ("method",), (0.1, 1.0))),
        ("inc_counter", ("hits", 2.5, {"method": "GET"})),
        ("observe_histogram", ("latency", 0.4, {"method": "GET"})),
    ]

    assert backend_a.calls == expected_calls
    assert backend_b.calls == expected_calls


class _PrometheusStub(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("prometheus_client")
        self.counter_instances: List["_PromCounter"] = []
        self.histogram_instances: List["_PromHistogram"] = []

        self_owner = self

        class _CounterChild:
            def __init__(self) -> None:
                self.value = 0.0

            def inc(self, amount: float = 1.0) -> None:
                self.value += amount

        class _HistogramChild:
            def __init__(self) -> None:
                self.values: List[float] = []

            def observe(self, value: float) -> None:
                self.values.append(value)

        class _PromCounter:
            def __init__(self, name: str, doc: str, labelnames: Sequence[str]) -> None:
                self.name = name
                self.doc = doc
                self.labelnames = tuple(labelnames)
                self.default = _CounterChild()
                self.children: Dict[Tuple[Tuple[str, str], ...], _CounterChild] = {}
                self._register()

            def _register(self) -> None:
                self_owner.counter_instances.append(self)

            def labels(self, **kwargs: str) -> _CounterChild:
                key = tuple(sorted(kwargs.items()))
                return self.children.setdefault(key, _CounterChild())

            def inc(self, amount: float = 1.0) -> None:
                self.default.inc(amount)

        class _PromHistogram:
            def __init__(self, name: str, doc: str, labelnames: Sequence[str], buckets: Sequence[float]) -> None:
                self.name = name
                self.doc = doc
                self.labelnames = tuple(labelnames)
                self.buckets = tuple(buckets)
                self.default = _HistogramChild()
                self.children: Dict[Tuple[Tuple[str, str], ...], _HistogramChild] = {}
                self._register()

            def _register(self) -> None:
                self_owner.histogram_instances.append(self)

            def labels(self, **kwargs: str) -> _HistogramChild:
                key = tuple(sorted(kwargs.items()))
                return self.children.setdefault(key, _HistogramChild())

            def observe(self, value: float) -> None:
                self.default.observe(value)

        self.Counter = _PromCounter
        self.Histogram = _PromHistogram

        class CollectorRegistry:
            pass

        self.CollectorRegistry = CollectorRegistry
        self.REGISTRY = CollectorRegistry()

        class _Multiprocess:
            def __init__(self) -> None:
                self.registry: Optional[CollectorRegistry] = None

            def MultiProcessCollector(self, registry: CollectorRegistry) -> None:
                self.registry = registry

        self.multiprocess = _Multiprocess()


def _make_prometheus_stub() -> _PrometheusStub:
    return _PrometheusStub()


def test_prometheus_backend_binds_stubbed_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _make_prometheus_stub()
    monkeypatch.setitem(sys.modules, "prometheus_client", stub)

    backend = metrics_module.PrometheusMetricsBackend()
    backend.register_counter("hits", ["method"])
    backend.register_histogram("latency", ["method"], buckets=[0.5])

    backend.inc_counter("hits", amount=2.0, labels={"method": "GET"})
    backend.observe_histogram("latency", value=0.1, labels={"method": "GET"})

    counter_instance = stub.counter_instances[0]
    histogram_instance = stub.histogram_instances[0]
    assert counter_instance.children[(("method", "GET"),)].value == 2.0
    assert histogram_instance.children[(("method", "GET"),)].values == [0.1]


def _split_segments(log_lines: Sequence[str]) -> List[str]:
    segments: List[str] = []
    for line in log_lines:
        segments.extend(part.strip() for part in line.split("  "))
    return [seg for seg in segments if seg]


def test_console_backend_sliding_window_rate_and_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=5.0, log_interval_seconds=0.0)
    backend.register_counter("requests", ["group", "path"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0, 6.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    labels = {"group": "api", "path": "/list"}
    backend.inc_counter("requests", labels=labels)
    backend.inc_counter("requests", labels=labels)
    backend.inc_counter("requests", labels=labels)

    segments = [seg for seg in _split_segments(logged) if seg.startswith("requests{group=api,path=/list}")]
    assert len(segments) == 3

    latest = segments[-1]
    rate_str = latest.rsplit("=", 1)[1].rstrip("/s")
    rate = float(rate_str)
    assert abs(rate - 0.40) < 1e-2


def _duration_to_seconds(payload: str) -> float:
    if payload.endswith("ms"):
        return float(payload[:-2]) / 1_000
    if payload.endswith("µs"):
        return float(payload[:-2]) / 1_000_000
    if payload.endswith("ns"):
        return float(payload[:-2]) / 1_000_000_000
    if payload.endswith("s"):
        return float(payload[:-1])
    raise AssertionError(f"Unknown duration format: {payload}")


def test_console_backend_histogram_quantiles_and_group_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=10.0, log_interval_seconds=0.0, group_level=2)
    backend.register_histogram("latency", ["service", "endpoint", "status"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    svc_a_labels = {"service": "svcA", "endpoint": "/search", "status": "200"}
    svc_b_labels = {"service": "svcB", "endpoint": "/chat", "status": "500"}

    backend.observe_histogram("latency", value=0.01, labels=svc_a_labels)
    backend.observe_histogram("latency", value=0.02, labels=svc_a_labels)
    backend.observe_histogram("latency", value=0.03, labels=svc_a_labels)
    backend.observe_histogram("latency", value=2.0, labels=svc_b_labels)

    svc_a_segments = [
        seg for seg in _split_segments(logged) if seg.startswith("latency{endpoint=/search,service=svcA}")
    ]
    assert svc_a_segments, "expected log entries for service A"
    latest = svc_a_segments[-1]
    assert "status" not in latest

    payload = latest.rsplit("=", 1)[1]
    p50_str, p95_str, p99_str = payload.split(",", 2)
    p50 = _duration_to_seconds(p50_str)
    p95 = _duration_to_seconds(p95_str)
    p99 = _duration_to_seconds(p99_str)

    assert abs(p50 - 0.02) < 1e-3
    assert abs(p95 - 0.029) < 5e-3
    assert abs(p99 - 0.0298) < 5e-3


def test_console_backend_logs_all_metric_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=None, log_interval_seconds=0.0)
    backend.register_counter("requests", ["group"])
    backend.register_counter("errors", ["group"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    backend.inc_counter("requests", labels={"group": "api"})
    backend.inc_counter("errors", labels={"group": "api"})

    assert logged, "expected log output"
    last_line = logged[-1]
    assert "requests{group=api}" in last_line
    assert "errors{group=api}" in last_line


def test_console_backend_snapshot_logs_single_line() -> None:
    backend = ConsoleMetricsBackend()
    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    backend._log_snapshot(
        [
            ("counter", {"g": "1"}, [0.0, 1.0], [1.0, 1.0]),
        ],
        [
            ("latency", {"g": "1"}, [0.1, 0.2], (0.5,)),
        ],
        snapshot_time=1.5,
    )

    assert logged and "counter{g=1}" in logged[0] and "latency{g=1}" in logged[0]
