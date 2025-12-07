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


def test_normalize_label_names_preserves_original_order() -> None:
    assert metrics_module._normalize_label_names(["b", "a", "b"]) == ("b", "a", "b")
    assert metrics_module._normalize_label_names(None) == ()


def test_console_backend_logs_counters_with_registration_order() -> None:
    backend = ConsoleMetricsBackend(log_interval_seconds=0.0)
    line = backend._log_counter(
        "rate",
        {"group1": "a", "group2": "b"},
        timestamps=[0.0, 1.0],
        amounts=[1.0, 1.0],
        snapshot_time=5.0,
    )

    assert line == "rate{group1=a,group2=b}=0.40/s"


def test_console_backend_logs_histograms_with_human_units() -> None:
    backend = ConsoleMetricsBackend()
    line = backend._log_histogram(
        "latency",
        {"group1": "a", "group2": "b"},
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
    backend = ConsoleMetricsBackend(group_level=2, log_interval_seconds=0.0)
    truncated = backend._truncate_labels_for_logging({"group1": "a", "group2": "b", "group3": "c"}, backend.group_level)
    line = backend._log_counter("metric", truncated, [0.0, 2.0], [1.0, 1.0], snapshot_time=5.0)

    assert line == "metric{group1=a,group2=b}=0.40/s"


def test_console_backend_log_uses_logger(caplog: pytest.LogCaptureFixture) -> None:
    backend = ConsoleMetricsBackend()
    caplog.set_level(logging.INFO, logger="agentlightning.utils.metrics")
    backend._log("hello metrics")
    assert any(record.message == "hello metrics" for record in caplog.records)


class _RecordingBackend(MetricsBackend):
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Tuple[Any, ...]]] = []

    def register_counter(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        self.calls.append(("register_counter", (name, tuple(label_names or ()), group_level)))

    def register_histogram(
        self,
        name: str,
        label_names: Optional[Sequence[str]] = None,
        buckets: Optional[Sequence[float]] = None,
        group_level: Optional[int] = None,
    ) -> None:
        self.calls.append(("register_histogram", (name, tuple(label_names or ()), tuple(buckets or ()), group_level)))

    async def inc_counter(self, name: str, amount: float = 1.0, labels: Optional[Dict[str, str]] = None) -> None:
        self.calls.append(("inc_counter", (name, amount, labels or {})))

    async def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        self.calls.append(("observe_histogram", (name, value, labels or {})))


@pytest.mark.asyncio
async def test_multi_metrics_backend_fans_out_calls() -> None:
    backend_a = _RecordingBackend()
    backend_b = _RecordingBackend()
    multi = MultiMetricsBackend([backend_a, backend_b])

    multi.register_counter("hits", label_names=["method"])
    multi.register_histogram("latency", label_names=["method"], buckets=[0.1, 1.0])
    await multi.inc_counter("hits", amount=2.5, labels={"method": "GET"})
    await multi.observe_histogram("latency", value=0.4, labels={"method": "GET"})

    expected_calls: List[Tuple[str, Tuple[Any, ...]]] = [
        ("register_counter", ("hits", ("method",), None)),
        ("register_histogram", ("latency", ("method",), (0.1, 1.0), None)),
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


@pytest.mark.asyncio
async def test_prometheus_backend_binds_stubbed_prometheus(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = _make_prometheus_stub()
    monkeypatch.setitem(sys.modules, "prometheus_client", stub)

    backend = metrics_module.PrometheusMetricsBackend()
    backend.register_counter("hits", ["method"])
    backend.register_histogram("latency", ["method"], buckets=[0.5])

    await backend.inc_counter("hits", amount=2.0, labels={"method": "GET"})
    await backend.observe_histogram("latency", value=0.1, labels={"method": "GET"})

    counter_instance = stub.counter_instances[0]
    histogram_instance = stub.histogram_instances[0]
    assert counter_instance.children[(("method", "GET"),)].value == 2.0
    assert histogram_instance.children[(("method", "GET"),)].values == [0.1]


def _split_segments(log_lines: Sequence[str]) -> List[str]:
    segments: List[str] = []
    for line in log_lines:
        segments.extend(part.strip() for part in line.split("  "))
    return [seg for seg in segments if seg]


@pytest.mark.asyncio
async def test_console_backend_sliding_window_rate_and_eviction(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=5.0, log_interval_seconds=0.0, group_level=2)
    backend.register_counter("requests", ["group", "path"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0, 6.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    labels = {"group": "api", "path": "/list"}
    await backend.inc_counter("requests", labels=labels)
    await backend.inc_counter("requests", labels=labels)
    await backend.inc_counter("requests", labels=labels)

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


@pytest.mark.asyncio
async def test_console_backend_histogram_quantiles_and_group_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=10.0, log_interval_seconds=0.0, group_level=2)
    backend.register_histogram("latency", ["service", "endpoint", "status"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    svc_a_labels = {"service": "svcA", "endpoint": "/search", "status": "200"}
    svc_b_labels = {"service": "svcB", "endpoint": "/chat", "status": "500"}

    await backend.observe_histogram("latency", value=0.01, labels=svc_a_labels)
    await backend.observe_histogram("latency", value=0.02, labels=svc_a_labels)
    await backend.observe_histogram("latency", value=0.03, labels=svc_a_labels)
    await backend.observe_histogram("latency", value=2.0, labels=svc_b_labels)

    svc_a_segments = [
        seg for seg in _split_segments(logged) if seg.startswith("latency{service=svcA,endpoint=/search}")
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


@pytest.mark.asyncio
async def test_console_backend_logs_all_metric_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=None, log_interval_seconds=0.0)
    backend.register_counter("requests", ["group"])
    backend.register_counter("errors", ["group"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    await backend.inc_counter("requests", labels={"group": "api"})
    await backend.inc_counter("errors", labels={"group": "api"})

    assert logged, "expected log output"
    last_line = logged[-1]
    assert "requests=" in last_line
    assert "errors=" in last_line


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

    assert logged and "counter=" in logged[0] and "latency=" in logged[0]


def test_console_backend_snapshot_entries_are_sorted() -> None:
    backend = ConsoleMetricsBackend(group_level=1)
    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    backend._log_snapshot(
        [
            ("zeta", {"a": "1"}, [0.0], [1.0]),
            ("alpha", {"b": "2"}, [0.1], [2.0]),
        ],
        [],
        snapshot_time=1.0,
    )

    assert logged, "expected log output"
    line = logged[-1]
    assert line.startswith("alpha{b=2}")
    assert "  zeta{a=1}" in line


def test_console_backend_group_level_none_aggregates_all_label_groups() -> None:
    backend = ConsoleMetricsBackend(group_level=None)
    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    backend._log_snapshot(
        [
            ("requests", {"group": "api", "path": "/a"}, [0.0], [1.0]),
            ("requests", {"group": "api", "path": "/b"}, [0.5], [2.0]),
        ],
        [
            ("latency", {"group": "api", "path": "/a"}, [0.1], (0.5,)),
            ("latency", {"group": "api", "path": "/b"}, [0.2], (0.5,)),
        ],
        snapshot_time=1.0,
    )

    assert logged, "expected log output"
    line = logged[-1]
    assert line.count("requests=") == 1
    assert line.count("latency=") == 1


def test_console_backend_group_level_positive_only_aggregates_prefix() -> None:
    backend = ConsoleMetricsBackend(group_level=1)
    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    backend._log_snapshot(
        [
            ("requests", {"a": "x", "b": "y"}, [0.0], [1.0]),
            ("requests", {"a": "x", "b": "z"}, [0.1], [2.0]),
            ("requests", {"a": "w", "b": "y"}, [0.2], [3.0]),
        ],
        [],
        snapshot_time=1.0,
    )

    assert logged, "expected log output"
    line = logged[-1]
    # With group_level=1 we should see aggregation by the first declared label key.
    assert line.count("requests{a=w}") == 1
    assert line.count("requests{a=x}") == 1


@pytest.mark.asyncio
async def test_console_backend_logs_preserve_registration_label_order(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=None, log_interval_seconds=0.0, group_level=3)
    backend.register_counter("requests", ["path", "method", "status"])

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    await backend.inc_counter("requests", labels={"status": "200", "path": "/search", "method": "GET"})

    assert logged, "expected log output"
    last_line = logged[-1]
    assert "requests{path=/search,method=GET,status=200}" in last_line


@pytest.mark.asyncio
async def test_console_backend_metric_specific_group_level_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=None, log_interval_seconds=0.0)
    backend.register_counter("requests", ["service", "endpoint", "status"], group_level=2)

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    await backend.inc_counter("requests", labels={"status": "200", "endpoint": "/search", "service": "svcA"})

    assert logged, "expected log output"
    last_line = logged[-1]
    assert "requests{service=svcA,endpoint=/search}" in last_line
    assert "status" not in last_line


@pytest.mark.asyncio
async def test_console_backend_global_group_level_overrides_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=None, log_interval_seconds=0.0, group_level=1)
    backend.register_counter("requests", ["service", "endpoint"], group_level=2)

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    await backend.inc_counter("requests", labels={"service": "svcA", "endpoint": "/search"})

    assert logged, "expected log output"
    last_line = logged[-1]
    assert "requests{service=svcA}" in last_line
    assert "endpoint" not in last_line


@pytest.mark.asyncio
async def test_console_backend_histogram_metric_group_level(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = ConsoleMetricsBackend(window_seconds=None, log_interval_seconds=0.0)
    backend.register_histogram("latency", ["service", "endpoint"], group_level=1)

    logged: List[str] = []
    backend._log = logged.append  # type: ignore[assignment]

    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(metrics_module.time, "time", lambda: next(times))

    await backend.observe_histogram("latency", value=0.01, labels={"service": "svcA", "endpoint": "/search"})
    await backend.observe_histogram("latency", value=0.02, labels={"service": "svcA", "endpoint": "/chat"})

    assert logged, "expected log output"
    latest = logged[-1]
    assert "latency{service=svcA}" in latest
    assert "endpoint" not in latest
