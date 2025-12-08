# Copyright (c) Microsoft. All rights reserved.

"""Lightweight benchmark report for the Prometheus + Grafana stack shipped with Agent Lightning."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, cast
from urllib import error, parse, request

import aiohttp


class PrometheusQueryError(RuntimeError):
    """Raised when Prometheus returns an error payload."""


class PrometheusClient:
    """Tiny helper around the Prometheus HTTP API."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        default_time: Optional[dt.datetime] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_time = default_time

    def query_vector(self, expr: str, eval_time: Optional[dt.datetime] = None) -> List[Mapping[str, object]]:
        params: Dict[str, str] = {"query": expr}
        query_time = eval_time or self.default_time
        if query_time is not None:
            params["time"] = query_time.isoformat()
        payload = self._get("/api/v1/query", params)
        status = payload.get("status")
        if not isinstance(status, str) or status != "success":
            error_msg = payload.get("error", "unknown error")
            raise PrometheusQueryError(str(error_msg))
        data_obj = payload.get("data", {})
        if isinstance(data_obj, dict):
            data = cast(Dict[str, Any], data_obj)
        else:
            data = {}
        result_type_obj = data.get("resultType")
        result_type = result_type_obj if isinstance(result_type_obj, str) else None
        raw_result_obj = data.get("result", [])
        raw_result: List[object]
        if isinstance(raw_result_obj, list):
            raw_result = cast(List[object], raw_result_obj)
        else:
            raw_result = []
        if result_type == "scalar":
            if len(raw_result) >= 2:
                ts = raw_result[0]
                value = raw_result[1]
                return [{"metric": {}, "value": [ts, value]}]
            return []
        vector_result: List[Mapping[str, object]] = [
            cast(Mapping[str, object], item) for item in raw_result if isinstance(item, Mapping)
        ]
        if result_type == "matrix":
            collapsed: List[Dict[str, object]] = []
            for series in vector_result:
                values_obj = series.get("values")
                if isinstance(values_obj, list) and values_obj and isinstance(values_obj[-1], Sequence):
                    last = cast(Sequence[object], values_obj[-1])
                else:
                    continue
                metric_obj = series.get("metric")
                if isinstance(metric_obj, Mapping):
                    metric: Dict[str, object] = dict(cast(Mapping[str, object], metric_obj))
                else:
                    metric = {}
                collapsed.append({"metric": metric, "value": list(last)})
            return cast(List[Mapping[str, object]], collapsed)
        if result_type == "vector":
            return vector_result
        return []

    def query_scalar(self, expr: str, eval_time: Optional[dt.datetime] = None) -> Optional[float]:
        samples = self.query_vector(expr, eval_time=eval_time)
        if not samples:
            return None
        return _sample_value(samples[0])

    def _get(self, path: str, data: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
        encoded: Optional[bytes] = None
        if data is not None:
            encoded = parse.urlencode(data).encode()
        req = request.Request(f"{self.base_url}{path}", data=encoded)
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                loaded = json.loads(resp.read().decode())
                if isinstance(loaded, dict):
                    return cast(Dict[str, Any], loaded)
                return {}
        except error.URLError as exc:  # pragma: no cover - network/infra issues
            raise PrometheusQueryError(str(exc)) from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark metrics from Prometheus.")
    parser.add_argument("--prom-url", default="http://localhost:9090", help="Base URL for the Prometheus API.")
    parser.add_argument(
        "--store-url",
        default="http://localhost:4747/v1/agl",
        help="Base URL for the Lightning Store API (without the /statistics suffix).",
    )
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds.")
    parser.add_argument("--start", type=str, help="ISO timestamp (e.g. 2024-05-01T12:00:00Z).")
    parser.add_argument("--end", type=str, help="ISO timestamp (default: now).")
    parser.add_argument(
        "--duration",
        type=str,
        default="5m",
        help="Fallback duration (e.g. 5m, 1h) used when --start is omitted.",
    )
    return parser.parse_args(argv)


def parse_timestamp(value: Optional[str], default: Optional[dt.datetime] = None) -> Optional[dt.datetime]:
    if value is None:
        return default
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)
    except ValueError as exc:  # pragma: no cover - invalid CLI input
        raise SystemExit(f"Invalid timestamp '{value}': {exc}") from exc


def parse_duration(text: str) -> dt.timedelta:
    units = {"s": 1, "m": 60, "h": 3600}
    if text.isdigit():
        return dt.timedelta(seconds=int(text))
    suffix = text[-1]
    if suffix not in units:
        raise SystemExit(f"Unsupported duration '{text}'. Use Ns/Nm/Nh.")
    try:
        value = int(text[:-1])
    except ValueError as exc:  # pragma: no cover - invalid CLI input
        raise SystemExit(f"Invalid duration '{text}': {exc}") from exc
    return dt.timedelta(seconds=value * units[suffix])


def format_window(seconds: float) -> str:
    seconds = max(int(seconds), 1)
    return f"{seconds}s"


def clamp_window_seconds(duration_seconds: float) -> int:
    return max(int(duration_seconds), 1)


def compute_peak_window(duration_seconds: float) -> str:
    peak_seconds = max(min(int(duration_seconds), 60), 1)
    return f"{peak_seconds}s"


def compute_subquery_step(duration_seconds: float) -> str:
    step_seconds = max(int(duration_seconds / 60), 1)
    step_seconds = min(step_seconds, 15)
    return f"{step_seconds}s"


def _sample_value(sample: Mapping[str, object]) -> Optional[float]:
    value_obj = sample.get("value")
    if not isinstance(value_obj, Sequence):
        return None
    value_seq = cast(Sequence[object], value_obj)
    if len(value_seq) < 2:
        return None
    candidate = value_seq[1]
    if isinstance(candidate, (int, float)):
        return float(candidate)
    if isinstance(candidate, str):
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def vector_to_map(
    samples: Optional[Sequence[Mapping[str, object]]],
    labels: Sequence[str],
) -> Dict[Any, float]:
    mapping: Dict[Any, float] = {}
    if not samples:
        return mapping
    for sample in samples:
        metric_obj = sample.get("metric", {})
        if isinstance(metric_obj, Mapping):
            metric: Dict[str, object] = dict(cast(Mapping[str, object], metric_obj))
        else:
            metric = {}
        if len(labels) == 1:
            key: Any = str(metric.get(labels[0], ""))
        else:
            key = tuple(str(metric.get(label, "")) for label in labels)
        value = _sample_value(sample)
        if value is not None:
            mapping[key] = value
    return mapping


def _normalize_label_value(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text if text else "-"


def vector_to_labeled_map(
    samples: Optional[Sequence[Mapping[str, object]]],
    labels: Sequence[str],
) -> Dict[Tuple[str, ...], float]:
    mapping: Dict[Tuple[str, ...], float] = {}
    if not samples:
        return mapping
    for sample in samples:
        metric_obj = sample.get("metric", {})
        if isinstance(metric_obj, Mapping):
            metric = dict(cast(Mapping[str, object], metric_obj))
        else:
            metric = {}
        if labels:
            key = tuple(_normalize_label_value(metric.get(label)) for label in labels)
        else:
            key = tuple()
        value = _sample_value(sample)
        if value is not None:
            mapping[key] = value
    return mapping


def sum_by_clause(labels: Sequence[str]) -> str:
    if labels:
        joined = ", ".join(labels)
        return f"sum by ({joined})"
    return "sum"


def histogram_sum_by_clause(labels: Sequence[str]) -> str:
    le_prefixed = ("le", *labels)
    joined = ", ".join(le_prefixed)
    return f"sum by ({joined})"


def histogram_sum_metric_name(bucket_metric: str) -> str:
    if bucket_metric.endswith("_bucket"):
        return f"{bucket_metric[: -len('_bucket')]}_sum"
    return f"{bucket_metric}_sum"


def histogram_count_metric_name(bucket_metric: str) -> str:
    if bucket_metric.endswith("_bucket"):
        return f"{bucket_metric[: -len('_bucket')]}_count"
    return f"{bucket_metric}_count"


def divide_or_none(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def compute_average_time_map(
    time_totals: Mapping[Tuple[str, ...], float],
    count_totals: Mapping[Tuple[str, ...], float],
) -> Dict[Tuple[str, ...], float]:
    averages: Dict[Tuple[str, ...], float] = {}
    keys = set(time_totals.keys()).union(count_totals.keys())
    for key in keys:
        avg = divide_or_none(time_totals.get(key), count_totals.get(key))
        if avg is not None:
            averages[key] = avg
    return averages


def safe_vector(client: PrometheusClient, expr: str) -> Optional[List[Mapping[str, object]]]:
    try:
        return client.query_vector(expr)
    except PrometheusQueryError as exc:
        print(f"[warn] Prometheus query failed: {exc} (expr={expr})")
        return None


def safe_scalar(client: PrometheusClient, expr: str) -> Optional[float]:
    try:
        return client.query_scalar(expr)
    except PrometheusQueryError as exc:
        print(f"[warn] Prometheus query failed: {exc} (expr={expr})")
        return None


def fetch_store_statistics(store_url: str, timeout: float) -> Optional[Dict[str, Any]]:
    store_url = store_url.rstrip("/")
    stats_url = f"{store_url}/statistics"
    req = request.Request(stats_url)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            loaded = json.loads(resp.read().decode())
            if isinstance(loaded, Mapping):
                return dict(cast(Mapping[str, Any], loaded))
            return None
    except error.URLError as exc:
        print(f"[warn] Failed to fetch store statistics: {exc} (url={stats_url})")
        return None
    except json.JSONDecodeError as exc:
        print(f"[warn] Failed to decode store statistics: {exc} (url={stats_url})")
        return None
    except aiohttp.ClientError as exc:
        print(f"[warn] HTTP error fetching store statistics: {exc} (url={stats_url})")
        return None


@dataclass
class CollectionThroughput:
    name: str
    count: Optional[float]
    per_sec: Optional[float]


@dataclass
class MetricRow:
    label_values: Tuple[str, ...]
    avg_rate: Optional[float]
    max_rate: Optional[float]
    min_rate: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]
    max_latency: Optional[float]
    time_per_sec: Optional[float]
    time_per_request: Optional[float]
    avg_rate_delta: Optional[float]
    p50_delta: Optional[float]
    p95_delta: Optional[float]
    time_delta: Optional[float]
    time_per_request_delta: Optional[float]


@dataclass(frozen=True)
class MetricGroupSpec:
    title: str
    histogram_bucket_metric: str
    label_names: Tuple[str, ...]
    label_headers: Tuple[str, ...]
    selector: str = ""
    sum_metric: Optional[str] = None
    count_metric: Optional[str] = None


def metric_row_sort_key(row: MetricRow) -> Tuple[str, ...]:
    return row.label_values


STORE_TOTAL_FIELDS = {
    "rollouts": "total_rollouts",
    "spans": "total_spans",
    "attempts": "total_attempts",
    "resources": "total_resources",
    "workers": "total_workers",
}
STORE_TOTAL_COLLECTIONS = tuple(STORE_TOTAL_FIELDS.keys())


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return None
    return None


def extract_store_totals(stats: Optional[Mapping[str, Any]]) -> Dict[str, Optional[int]]:
    totals: Dict[str, Optional[int]] = {}
    if not stats:
        return totals
    for display_name, field_name in STORE_TOTAL_FIELDS.items():
        if field_name in stats:
            totals[display_name] = _coerce_int(stats.get(field_name))
        else:
            totals[display_name] = None
    return totals


def gather_collection_throughput(
    client: PrometheusClient, collections: Sequence[str], duration_seconds: float
) -> List[CollectionThroughput]:
    rows: List[CollectionThroughput] = []
    window = format_window(duration_seconds)
    for collection in collections:
        # Successful insert operations reflect the number of new records.
        expr = (
            "sum("
            f'increase(mongo_operation_total{{collection="{collection}", operation="insert", status="ok"}}[{window}])'
            ")"
        )
        count = safe_scalar(client, expr)
        if count is not None and count < 0:
            count = 0.0
        per_sec = (count / duration_seconds) if (count is not None and duration_seconds > 0) else None
        rows.append(CollectionThroughput(collection, count, per_sec))
    return rows


def gather_metric_group(
    client: PrometheusClient,
    spec: MetricGroupSpec,
    *,
    window: str,
    window_seconds: int,
    peak_window: str,
    subquery_step: str,
    half_window: Optional[str],
    half_window_seconds: Optional[int],
) -> List[MetricRow]:
    label_names = spec.label_names
    sum_clause = sum_by_clause(label_names)
    hist_clause = histogram_sum_by_clause(label_names)
    bucket_metric = f"{spec.histogram_bucket_metric}{spec.selector}" if spec.selector else spec.histogram_bucket_metric
    base_sum_metric = spec.sum_metric or histogram_sum_metric_name(spec.histogram_bucket_metric)
    sum_metric = f"{base_sum_metric}{spec.selector}" if spec.selector else base_sum_metric
    base_count_metric = spec.count_metric or histogram_count_metric_name(spec.histogram_bucket_metric)
    count_metric = f"{base_count_metric}{spec.selector}" if spec.selector else base_count_metric

    count_total_expr = f"{sum_clause}(increase({count_metric}[{window}]))"
    count_total_map = vector_to_labeled_map(safe_vector(client, count_total_expr), label_names)
    avg_map = {key: value / window_seconds for key, value in count_total_map.items()} if window_seconds > 0 else {}

    peak_expr = f"{sum_clause}(irate({count_metric}[{peak_window}]))"
    max_expr = f"max_over_time(({peak_expr})[{window}:{subquery_step}])"
    min_expr = f"min_over_time(({peak_expr})[{window}:{subquery_step}])"
    max_map = vector_to_labeled_map(safe_vector(client, max_expr), label_names)
    min_map = vector_to_labeled_map(safe_vector(client, min_expr), label_names)

    p50_map = vector_to_labeled_map(
        safe_vector(
            client,
            f"histogram_quantile(0.50, {hist_clause}(increase({bucket_metric}[{window}])))",
        ),
        label_names,
    )
    p95_map = vector_to_labeled_map(
        safe_vector(
            client,
            f"histogram_quantile(0.95, {hist_clause}(increase({bucket_metric}[{window}])))",
        ),
        label_names,
    )
    p99_map = vector_to_labeled_map(
        safe_vector(
            client,
            f"histogram_quantile(0.99, {hist_clause}(increase({bucket_metric}[{window}])))",
        ),
        label_names,
    )
    max_latency_map = vector_to_labeled_map(
        safe_vector(
            client,
            f"histogram_quantile(1.00, {hist_clause}(increase({bucket_metric}[{window}])))",
        ),
        label_names,
    )

    time_total_expr = f"{sum_clause}(increase({sum_metric}[{window}]))"
    time_total_map = vector_to_labeled_map(safe_vector(client, time_total_expr), label_names)
    time_rate_map = {key: value / window_seconds for key, value in time_total_map.items()} if window_seconds > 0 else {}
    avg_time_map = compute_average_time_map(time_total_map, count_total_map)

    if half_window and half_window_seconds and half_window_seconds > 0:
        count_late_expr = f"{sum_clause}(increase({count_metric}[{half_window}]))"
        count_early_expr = f"{sum_clause}(increase({count_metric}[{half_window}] offset {half_window}))"
        count_late_total_map = vector_to_labeled_map(safe_vector(client, count_late_expr), label_names)
        count_early_total_map = vector_to_labeled_map(safe_vector(client, count_early_expr), label_names)
        avg_late_map = {key: value / half_window_seconds for key, value in count_late_total_map.items()}
        avg_early_map = {key: value / half_window_seconds for key, value in count_early_total_map.items()}

        p50_late_expr = f"histogram_quantile(0.50, {hist_clause}(increase({bucket_metric}[{half_window}])))"
        p50_early_expr = (
            f"histogram_quantile(0.50, {hist_clause}(increase({bucket_metric}[{half_window}] offset {half_window})))"
        )
        p50_late_map = vector_to_labeled_map(safe_vector(client, p50_late_expr), label_names)
        p50_early_map = vector_to_labeled_map(safe_vector(client, p50_early_expr), label_names)

        p95_late_expr = f"histogram_quantile(0.95, {hist_clause}(increase({bucket_metric}[{half_window}])))"
        p95_early_expr = (
            f"histogram_quantile(0.95, {hist_clause}(increase({bucket_metric}[{half_window}] offset {half_window})))"
        )
        p95_late_map = vector_to_labeled_map(safe_vector(client, p95_late_expr), label_names)
        p95_early_map = vector_to_labeled_map(safe_vector(client, p95_early_expr), label_names)

        time_late_expr = f"{sum_clause}(increase({sum_metric}[{half_window}]))"
        time_early_expr = f"{sum_clause}(increase({sum_metric}[{half_window}] offset {half_window}))"
        time_late_total_map = vector_to_labeled_map(safe_vector(client, time_late_expr), label_names)
        time_early_total_map = vector_to_labeled_map(safe_vector(client, time_early_expr), label_names)
        time_late_map = {key: value / half_window_seconds for key, value in time_late_total_map.items()}
        time_early_map = {key: value / half_window_seconds for key, value in time_early_total_map.items()}
        avg_time_late_map = compute_average_time_map(time_late_total_map, count_late_total_map)
        avg_time_early_map = compute_average_time_map(time_early_total_map, count_early_total_map)
    else:
        count_late_total_map: Dict[Tuple[str, ...], float] = {}
        count_early_total_map: Dict[Tuple[str, ...], float] = {}
        avg_late_map: Dict[Tuple[str, ...], float] = {}
        avg_early_map: Dict[Tuple[str, ...], float] = {}
        p50_late_map: Dict[Tuple[str, ...], float] = {}
        p50_early_map: Dict[Tuple[str, ...], float] = {}
        p95_late_map: Dict[Tuple[str, ...], float] = {}
        p95_early_map: Dict[Tuple[str, ...], float] = {}
        time_late_map: Dict[Tuple[str, ...], float] = {}
        time_early_map: Dict[Tuple[str, ...], float] = {}
        time_late_total_map = {}
        time_early_total_map = {}
        avg_time_late_map = {}
        avg_time_early_map = {}

    all_keys: Set[Tuple[str, ...]] = set()
    all_keys.update(count_total_map.keys())
    all_keys.update(avg_map.keys())
    all_keys.update(max_map.keys())
    all_keys.update(min_map.keys())
    all_keys.update(p50_map.keys())
    all_keys.update(p95_map.keys())
    all_keys.update(p99_map.keys())
    all_keys.update(max_latency_map.keys())
    all_keys.update(time_rate_map.keys())
    all_keys.update(avg_time_map.keys())
    all_keys.update(count_late_total_map.keys())
    all_keys.update(count_early_total_map.keys())
    all_keys.update(avg_late_map.keys())
    all_keys.update(avg_early_map.keys())
    all_keys.update(p50_late_map.keys())
    all_keys.update(p50_early_map.keys())
    all_keys.update(p95_late_map.keys())
    all_keys.update(p95_early_map.keys())
    all_keys.update(time_late_map.keys())
    all_keys.update(time_early_map.keys())
    all_keys.update(avg_time_late_map.keys())
    all_keys.update(avg_time_early_map.keys())

    if not all_keys:
        return []

    def build_delta(
        late_map: Mapping[Tuple[str, ...], float],
        early_map: Mapping[Tuple[str, ...], float],
        key: Tuple[str, ...],
    ) -> Optional[float]:
        late = late_map.get(key)
        early = early_map.get(key)
        if late is None or early is None:
            return None
        return late - early

    rows: List[MetricRow] = []
    for key in sorted(all_keys):
        rows.append(
            MetricRow(
                label_values=key,
                avg_rate=avg_map.get(key),
                max_rate=max_map.get(key),
                min_rate=min_map.get(key),
                p50=p50_map.get(key),
                p95=p95_map.get(key),
                p99=p99_map.get(key),
                max_latency=max_latency_map.get(key),
                time_per_sec=time_rate_map.get(key),
                time_per_request=avg_time_map.get(key),
                avg_rate_delta=build_delta(avg_late_map, avg_early_map, key),
                p50_delta=build_delta(p50_late_map, p50_early_map, key),
                p95_delta=build_delta(p95_late_map, p95_early_map, key),
                time_delta=build_delta(time_late_map, time_early_map, key),
                time_per_request_delta=build_delta(avg_time_late_map, avg_time_early_map, key),
            )
        )
    return rows


def gather_diagnostics(client: PrometheusClient, window: str) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    diagnostics["mongo_ops"] = vector_to_map(
        safe_vector(
            client,
            f"sum by (operation)(rate(mongo_operation_total{{operation!='ensure_collection'}}[{window}]))",
        ),
        ("operation",),
    )
    diagnostics["mongo_latency_p50"] = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.50, sum by (le, operation)(rate(mongo_operation_duration_seconds_bucket{{operation!='ensure_collection'}}[{window}])))",
        ),
        ("operation",),
    )
    diagnostics["mongo_latency_p95"] = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.95, sum by (le, operation)(rate(mongo_operation_duration_seconds_bucket{{operation!='ensure_collection'}}[{window}])))",
        ),
        ("operation",),
    )
    diagnostics["mongo_latency_p99"] = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.99, sum by (le, operation)(rate(mongo_operation_duration_seconds_bucket{{operation!='ensure_collection'}}[{window}])))",
        ),
        ("operation",),
    )
    opcounters_samples = safe_vector(client, f"sum by (legacy_op_type)(rate(mongodb_ss_opcounters[{window}]))")
    mongo_opcounters: Dict[str, float] = {}
    if opcounters_samples:
        for sample in opcounters_samples:
            metric_obj = sample.get("metric", {})
            if isinstance(metric_obj, Mapping):
                metric: Dict[str, object] = dict(cast(Mapping[str, object], metric_obj))
            else:
                metric = {}
            label_value = metric.get("legacy_op_type") or metric.get("type")
            label = str(label_value) if label_value is not None else ""
            value = _sample_value(sample)
            if value is not None:
                mongo_opcounters[str(label or "-")] = value
    diagnostics["mongo_opcounters"] = mongo_opcounters
    diagnostics["mongo_connections"] = safe_scalar(client, "avg(mongodb_ss_connections{conn_type='current'})")
    diagnostics["memory_lock_rate"] = vector_to_map(
        safe_vector(client, f"sum by (collection)(rate(memory_collection_lock_rate_total[{window}]))"),
        ("collection",),
    )
    diagnostics["memory_lock_p50"] = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.50, sum by (le, collection)(rate(memory_collection_lock_latency_seconds_bucket[{window}])))",
        ),
        ("collection",),
    )
    diagnostics["memory_lock_p95"] = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.95, sum by (le, collection)(rate(memory_collection_lock_latency_seconds_bucket[{window}])))",
        ),
        ("collection",),
    )
    diagnostics["memory_lock_p99"] = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.99, sum by (le, collection)(rate(memory_collection_lock_latency_seconds_bucket[{window}])))",
        ),
        ("collection",),
    )
    diagnostics["cpu_usage"] = safe_scalar(client, f"1 - avg(rate(node_cpu_seconds_total{{mode='idle'}}[{window}]))")
    diagnostics["memory_total"] = safe_scalar(client, "avg(node_memory_MemTotal_bytes)")
    diagnostics["memory_available"] = safe_scalar(client, "avg(node_memory_MemAvailable_bytes)")
    diagnostics["network_rx"] = safe_scalar(
        client,
        f"sum(rate(node_network_receive_bytes_total{{device!~'lo|docker.*'}}[{window}]))",
    )
    diagnostics["network_tx"] = safe_scalar(
        client,
        f"sum(rate(node_network_transmit_bytes_total{{device!~'lo|docker.*'}}[{window}]))",
    )
    diagnostics["disk_read_ops"] = safe_scalar(client, f"sum(rate(node_disk_reads_completed_total[{window}]))")
    diagnostics["disk_write_ops"] = safe_scalar(client, f"sum(rate(node_disk_writes_completed_total[{window}]))")
    diagnostics["disk_read_bytes"] = safe_scalar(client, f"sum(rate(node_disk_read_bytes_total[{window}]))")
    diagnostics["disk_write_bytes"] = safe_scalar(client, f"sum(rate(node_disk_written_bytes_total[{window}]))")
    return diagnostics


def render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    if not rows:
        return [f"(no data for {headers})"]
    widths = [len(h) for h in headers]
    rendered: List[List[str]] = []
    for row in rows:
        rendered_row = [str(cell) for cell in row]
        for idx, cell in enumerate(rendered_row):
            widths[idx] = max(widths[idx], len(cell))
        rendered.append(rendered_row)

    lines = [
        " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))),
        "-+-".join("-" * widths[idx] for idx in range(len(headers))),
    ]
    for row in rendered:
        lines.append(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))
    return lines


def fmt_rate(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value:.2f}/s"


def fmt_latency(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    if abs(value) < 0.5:
        return f"{value * 1e3:.2f} ms"
    return f"{value:.2f} s"


def fmt_bytes(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 0
    current = value
    while current >= 1024 and idx < len(units) - 1:
        current /= 1024
        idx += 1
    return f"{current:.2f} {units[idx]}"


def fmt_percentage(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "-"
    return f"{value * 100:4.1f}%"


def section(title: str, body: Iterable[str]) -> List[str]:
    lines = [f"## {title}"]
    lines.extend(body)
    lines.append("")
    return lines


def render_metric_group_table(
    spec: MetricGroupSpec,
    rows: Sequence[MetricRow],
    extra_columns: Optional[Sequence[Tuple[str, Callable[[MetricRow], str]]]] = None,
) -> List[str]:
    headers = list(spec.label_headers)
    headers.extend(
        [
            "Avg Rate/s",
            "Max Rate/s",
            "Min Rate/s",
            "P50",
            "P95",
            "P99",
            "Max Latency",
            "Time/s",
            "Avg Time/req",
            "Avg Rate Δ",
            "P50 Δ",
            "P95 Δ",
            "Time Δ",
            "Avg Time/req Δ",
        ]
    )
    column_renderers: Sequence[Tuple[str, Callable[[MetricRow], str]]] = extra_columns or ()
    for header, _ in column_renderers:
        headers.append(header)
    if not rows:
        return render_table(headers, [])
    sorted_rows = sorted(rows, key=metric_row_sort_key)
    rendered_rows: List[List[str]] = []
    for row in sorted_rows:
        label_cells = list(row.label_values) if spec.label_headers else []
        metrics = [
            fmt_rate(row.avg_rate),
            fmt_rate(row.max_rate),
            fmt_rate(row.min_rate),
            fmt_latency(row.p50),
            fmt_latency(row.p95),
            fmt_latency(row.p99),
            fmt_latency(row.max_latency),
            fmt_latency(row.time_per_sec),
            fmt_latency(row.time_per_request),
            fmt_rate(row.avg_rate_delta),
            fmt_latency(row.p50_delta),
            fmt_latency(row.p95_delta),
            fmt_latency(row.time_delta),
            fmt_latency(row.time_per_request_delta),
        ]
        extra_cells = [renderer(row) for _, renderer in column_renderers]
        rendered_rows.append(label_cells + metrics + extra_cells)
    return render_table(headers, rendered_rows)


def make_store_time_share_column(
    store_time_per_sec: Mapping[str, Optional[float]],
) -> Tuple[str, Callable[[MetricRow], str]]:
    def render_cell(row: MetricRow) -> str:
        if not row.label_values:
            return "-"
        method = row.label_values[0]
        store_time = store_time_per_sec.get(method)
        collection_time = row.time_per_sec
        if (
            store_time is None
            or collection_time is None
            or math.isnan(store_time)
            or math.isnan(collection_time)
            or store_time <= 0
        ):
            return "-"
        return fmt_percentage(collection_time / store_time)

    return ("Store Time %", render_cell)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    end = parse_timestamp(args.end, default=dt.datetime.now(dt.timezone.utc))
    if end is None:
        raise SystemExit("End timestamp could not be determined.")
    start = parse_timestamp(args.start)
    if start is None:
        duration = parse_duration(args.duration)
        start = end - duration
    assert start is not None
    duration_seconds = max((end - start).total_seconds(), 1.0)
    window_seconds = clamp_window_seconds(duration_seconds)
    window = format_window(duration_seconds)
    peak_window = compute_peak_window(duration_seconds)
    subquery_step = compute_subquery_step(duration_seconds)
    half_window_seconds = window_seconds // 2 if window_seconds // 2 >= 1 else None
    half_window = format_window(half_window_seconds) if half_window_seconds else None

    client = PrometheusClient(args.prom_url, timeout=args.timeout, default_time=end)
    store_stats = fetch_store_statistics(args.store_url, timeout=args.timeout)
    store_totals = extract_store_totals(store_stats)
    lines: List[str] = [
        f"Agent Lightning benchmark report",
        f"Range: {start.isoformat()} — {end.isoformat()} ({duration_seconds:.0f}s window)",
        f"Prometheus: {args.prom_url}",
        f"Store: {args.store_url}",
        "",
    ]

    # Throughput
    throughput_rows = gather_collection_throughput(
        client, collections=STORE_TOTAL_COLLECTIONS, duration_seconds=duration_seconds
    )
    throughput_table: List[List[str]] = []
    for item in throughput_rows:
        store_total = store_totals.get(item.name)
        if store_total is not None:
            count_value: Optional[int] = store_total
        elif item.count is not None:
            count_value = int(item.count)
        else:
            count_value = None
        if count_value is None:
            count_str = "-"
        else:
            count_str = f"{count_value:,}"
        if count_value is not None and duration_seconds > 0:
            per_sec_value = float(count_value) / duration_seconds
        else:
            per_sec_value = item.per_sec
        throughput_table.append([item.name, count_str, fmt_rate(per_sec_value)])
    lines.extend(
        section(
            "Rollout / Attempt / Span / Resource / Worker Throughput",
            render_table(["Collection", "Count", "Per Sec"], throughput_table),
        )
    )

    metric_categories: Sequence[Tuple[str, Sequence[MetricGroupSpec]]] = [
        (
            "HTTP Metrics",
            (
                MetricGroupSpec(
                    title="agl.http ungrouped",
                    histogram_bucket_metric="agl_http_latency_bucket",
                    label_names=tuple(),
                    label_headers=tuple(),
                ),
                MetricGroupSpec(
                    title="agl.http grouped by path, method",
                    histogram_bucket_metric="agl_http_latency_bucket",
                    label_names=("path", "method"),
                    label_headers=("Path", "Method"),
                ),
                MetricGroupSpec(
                    title="agl.http grouped by path, method, status",
                    histogram_bucket_metric="agl_http_latency_bucket",
                    label_names=("path", "method", "status"),
                    label_headers=("Path", "Method", "Status"),
                ),
            ),
        ),
        (
            "Store Metrics",
            (
                MetricGroupSpec(
                    title="agl.store ungrouped",
                    histogram_bucket_metric="agl_store_latency_bucket",
                    label_names=tuple(),
                    label_headers=tuple(),
                ),
                MetricGroupSpec(
                    title="agl.store grouped by method",
                    histogram_bucket_metric="agl_store_latency_bucket",
                    label_names=("method",),
                    label_headers=("Method",),
                ),
                MetricGroupSpec(
                    title="agl.store grouped by method, status",
                    histogram_bucket_metric="agl_store_latency_bucket",
                    label_names=("method", "status"),
                    label_headers=("Method", "Status"),
                ),
            ),
        ),
        (
            "Rollout Outcomes",
            (
                MetricGroupSpec(
                    title="agl.rollouts ungrouped",
                    histogram_bucket_metric="agl_rollouts_duration_bucket",
                    label_names=tuple(),
                    label_headers=tuple(),
                ),
                MetricGroupSpec(
                    title="agl.rollouts grouped by status",
                    histogram_bucket_metric="agl_rollouts_duration_bucket",
                    label_names=("status",),
                    label_headers=("Status",),
                ),
            ),
        ),
        (
            "Collection Metrics",
            (
                MetricGroupSpec(
                    title="agl.collections grouped by store_method, collection, operation",
                    histogram_bucket_metric="agl_collections_latency_bucket",
                    label_names=("store_method", "collection"),
                    label_headers=("Store Method", "Collection"),
                ),
                MetricGroupSpec(
                    title="agl.collections grouped by store_method, collection, operation, status",
                    histogram_bucket_metric="agl_collections_latency_bucket",
                    label_names=("store_method", "collection", "operation", "status"),
                    label_headers=("Store Method", "Collection", "Operation", "Status"),
                ),
                MetricGroupSpec(
                    title="agl.collections grouped by collection, operation, status",
                    histogram_bucket_metric="agl_collections_latency_bucket",
                    label_names=("collection", "operation", "status"),
                    label_headers=("Collection", "Operation", "Status"),
                ),
            ),
        ),
    ]

    store_method_time_per_sec: Dict[str, Optional[float]] = {}

    for category_title, specs in metric_categories:
        category_lines: List[str] = []
        for idx, spec in enumerate(specs):
            rows = gather_metric_group(
                client,
                spec,
                window=window,
                window_seconds=window_seconds,
                peak_window=peak_window,
                subquery_step=subquery_step,
                half_window=half_window,
                half_window_seconds=half_window_seconds,
            )
            if spec.histogram_bucket_metric == "agl_store_latency_bucket" and spec.label_names == ("method",):
                store_method_time_per_sec = {
                    row.label_values[0]: row.time_per_sec
                    for row in rows
                    if row.label_values and len(row.label_values) == 1
                }
            extra_columns: Optional[Sequence[Tuple[str, Callable[[MetricRow], str]]]] = None
            if "store_method" in spec.label_names:
                extra_columns = [make_store_time_share_column(store_method_time_per_sec)]
            category_lines.append("### " + spec.title)
            category_lines.extend(render_metric_group_table(spec, rows, extra_columns=extra_columns))
            if idx != len(specs) - 1:
                category_lines.append("")
        lines.extend(section(category_title, category_lines))

    # Diagnostics
    diag = gather_diagnostics(client, window)
    diagnostics_blocks: List[List[str]] = []

    mongo_ops = cast(Dict[str, float], diag.get("mongo_ops", {}))
    mongo_latency_p50 = cast(Dict[str, float], diag.get("mongo_latency_p50", {}))
    mongo_latency_p95 = cast(Dict[str, float], diag.get("mongo_latency_p95", {}))
    mongo_latency_p99 = cast(Dict[str, float], diag.get("mongo_latency_p99", {}))
    mongo_op_keys = sorted(
        {
            *mongo_ops.keys(),
            *mongo_latency_p50.keys(),
            *mongo_latency_p95.keys(),
            *mongo_latency_p99.keys(),
        },
        key=str,
    )
    mongo_ops_rows = [
        [
            op or "-",
            fmt_rate(mongo_ops.get(op)),
            fmt_latency(mongo_latency_p50.get(op)),
            fmt_latency(mongo_latency_p95.get(op)),
            fmt_latency(mongo_latency_p99.get(op)),
        ]
        for op in mongo_op_keys
    ]
    diagnostics_blocks.append(render_table(["Mongo Operation", "Ops/s", "P50", "P95", "P99"], mongo_ops_rows))

    mongo_opcounters = cast(Dict[str, float], diag.get("mongo_opcounters", {}))
    mongo_opcounters_rows = [
        [op_type or "-", fmt_rate(rate)]
        for op_type, rate in sorted(mongo_opcounters.items(), key=lambda item: str(item[0]))
    ]
    diagnostics_blocks.append(render_table(["MongoDB Opcounter", "Ops/s"], mongo_opcounters_rows))

    mongo_misc_rows: List[List[str]] = []
    if diag.get("mongo_connections") is not None:
        mongo_misc_rows.append(["MongoDB connections (avg)", f"{diag['mongo_connections']:.2f}"])
    if mongo_misc_rows:
        diagnostics_blocks.append(render_table(["Mongo Metric", "Value"], mongo_misc_rows))

    node_rows: List[List[str]] = []
    if diag.get("cpu_usage") is not None:
        node_rows.append(["CPU usage", fmt_percentage(diag["cpu_usage"])])
    mem_total = diag.get("memory_total")
    mem_available = diag.get("memory_available")
    if mem_total and mem_available:
        used = mem_total - mem_available
        node_rows.append(
            ["Memory usage", f"{fmt_bytes(used)} / {fmt_bytes(mem_total)} ({fmt_percentage(used / mem_total)})"]
        )
    node_rows.append(["Network rx", f"{fmt_bytes(diag.get('network_rx'))}/s"])
    node_rows.append(["Network tx", f"{fmt_bytes(diag.get('network_tx'))}/s"])
    node_rows.append(["Disk read ops", fmt_rate(diag.get("disk_read_ops"))])
    node_rows.append(["Disk read bytes", f"{fmt_bytes(diag.get('disk_read_bytes'))}/s"])
    node_rows.append(["Disk write ops", fmt_rate(diag.get("disk_write_ops"))])
    node_rows.append(["Disk write bytes", f"{fmt_bytes(diag.get('disk_write_bytes'))}/s"])
    diagnostics_blocks.append(render_table(["Node Metric", "Value"], node_rows))

    diagnostics_lines: List[str] = []
    for idx, block in enumerate(diagnostics_blocks):
        diagnostics_lines.extend(block)
        if idx != len(diagnostics_blocks) - 1:
            diagnostics_lines.append("")

    lines.extend(section("Diagnostics", diagnostics_lines))

    print("\n".join(lines))


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
