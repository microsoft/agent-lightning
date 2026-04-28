# Copyright (c) Microsoft. All rights reserved.

"""Lightweight benchmark report for the Prometheus + Grafana stack shipped with Agent Lightning."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeGuard, cast
from urllib import error, parse, request


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
    parser.add_argument("--top", type=int, default=8, help="Number of rows to show per table.")
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


def compute_rate_window(duration_seconds: float) -> str:
    return format_window(min(duration_seconds, 60.0))


def compute_subquery_step(duration_seconds: float) -> str:
    step_seconds = max(int(duration_seconds / 60), 1)
    step_seconds = min(step_seconds, 15)
    return f"{step_seconds}s"


def _is_http_pair(value: Any) -> TypeGuard[Tuple[Any, Any]]:
    if not isinstance(value, tuple):
        return False
    try:
        value[0]
        value[1]
    except IndexError:
        return False
    return True


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


# ---------------------------------------------------------------------------
# Part 1 – high level throughput
# ---------------------------------------------------------------------------


@dataclass
class CollectionThroughput:
    name: str
    count: Optional[float]
    per_sec: Optional[float]


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


# ---------------------------------------------------------------------------
# Part 2 – CollectionBasedLightningStore method stats
# ---------------------------------------------------------------------------


@dataclass
class StoreMethodStats:
    method: str
    ops_mean: float
    ops_max: Optional[float]
    ops_min: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]


StatsSummary = Dict[str, Optional[float]]


@dataclass
class RolloutOutcomeStats:
    status: str
    rate: Optional[float]
    p25: Optional[float]
    p50: Optional[float]
    p75: Optional[float]
    max_latency: Optional[float]


def gather_store_methods(
    client: PrometheusClient,
    window: str,
    rate_window: str,
    subquery_step: str,
) -> Tuple[List[StoreMethodStats], StatsSummary]:
    ops_expr = f"sum by (method)(rate(collection_store_total[{rate_window}]))"
    ops_mean = vector_to_map(
        safe_vector(client, f"avg_over_time(({ops_expr})[{window}:{subquery_step}])"),
        ("method",),
    )
    ops_max = vector_to_map(
        safe_vector(client, f"max_over_time(({ops_expr})[{window}:{subquery_step}])"),
        ("method",),
    )
    ops_min = vector_to_map(
        safe_vector(client, f"min_over_time(({ops_expr})[{window}:{subquery_step}])"),
        ("method",),
    )
    p50 = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.50, sum by (le, method)(rate(collection_store_latency_seconds_bucket[{window}])))",
        ),
        ("method",),
    )
    p95 = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.95, sum by (le, method)(rate(collection_store_latency_seconds_bucket[{window}])))",
        ),
        ("method",),
    )
    p99 = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.99, sum by (le, method)(rate(collection_store_latency_seconds_bucket[{window}])))",
        ),
        ("method",),
    )
    rows: List[StoreMethodStats] = []
    for method, rate in sorted(ops_mean.items(), key=lambda item: str(item[0])):
        p50_val = p50.get(method)
        p95_val = p95.get(method)
        p99_val = p99.get(method)
        rows.append(
            StoreMethodStats(
                str(method or "-"),
                rate,
                ops_max.get(method),
                ops_min.get(method),
                p50_val,
                p95_val,
                p99_val,
            )
        )

    overall: StatsSummary = {
        "ops_mean": safe_scalar(
            client,
            f"avg_over_time((sum(rate(collection_store_total[{rate_window}])))[{window}:{subquery_step}])",
        )
        or 0.0,
        "ops_max": safe_scalar(
            client,
            f"max_over_time((sum(rate(collection_store_total[{rate_window}])))[{window}:{subquery_step}])",
        ),
        "ops_min": safe_scalar(
            client,
            f"min_over_time((sum(rate(collection_store_total[{rate_window}])))[{window}:{subquery_step}])",
        ),
        "p50": safe_scalar(
            client,
            f"histogram_quantile(0.50, sum by (le)(rate(collection_store_latency_seconds_bucket[{window}])))",
        )
        or 0.0,
        "p95": safe_scalar(
            client,
            f"histogram_quantile(0.95, sum by (le)(rate(collection_store_latency_seconds_bucket[{window}])))",
        )
        or 0.0,
        "p99": safe_scalar(
            client,
            f"histogram_quantile(0.99, sum by (le)(rate(collection_store_latency_seconds_bucket[{window}])))",
        )
        or 0.0,
    }
    return rows, overall


def gather_rollout_outcomes(
    client: PrometheusClient,
    window: str,
    rate_window: str,
) -> List[RolloutOutcomeStats]:
    rate_map = vector_to_map(
        safe_vector(client, f"sum by (status)(rate(collection_store_rollout_total[{rate_window}]))"),
        ("status",),
    )
    p25_map = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.25, "
            f"sum by (le, status)(rate(collection_store_rollout_duration_seconds_bucket[{window}])))",
        ),
        ("status",),
    )
    p50_map = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.50, "
            f"sum by (le, status)(rate(collection_store_rollout_duration_seconds_bucket[{window}])))",
        ),
        ("status",),
    )
    p75_map = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.75, "
            f"sum by (le, status)(rate(collection_store_rollout_duration_seconds_bucket[{window}])))",
        ),
        ("status",),
    )
    max_map = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(1.00, "
            f"sum by (le, status)(rate(collection_store_rollout_duration_seconds_bucket[{window}])))",
        ),
        ("status",),
    )
    statuses = sorted({*rate_map.keys(), *p25_map.keys(), *p50_map.keys(), *p75_map.keys(), *max_map.keys()}, key=str)
    stats: List[RolloutOutcomeStats] = []
    for status in statuses:
        stats.append(
            RolloutOutcomeStats(
                status=str(status or "-"),
                rate=rate_map.get(status),
                p25=p25_map.get(status),
                p50=p50_map.get(status),
                p75=p75_map.get(status),
                max_latency=max_map.get(status),
            )
        )
    return stats


# ---------------------------------------------------------------------------
# Part 3 – HTTP traffic
# ---------------------------------------------------------------------------


@dataclass
class HttpPathStats:
    method: str
    path: str
    qps_mean: float
    qps_max: Optional[float]
    qps_min: Optional[float]
    p50: Optional[float]
    p95: Optional[float]
    p99: Optional[float]


def gather_http_paths(
    client: PrometheusClient,
    window: str,
    rate_window: str,
    subquery_step: str,
) -> Tuple[List[HttpPathStats], StatsSummary]:
    qps_expr = f"sum by (method, path)(rate(http_requests_total[{rate_window}]))"
    qps_mean = vector_to_map(
        safe_vector(client, f"avg_over_time(({qps_expr})[{window}:{subquery_step}])"),
        ("method", "path"),
    )
    qps_max = vector_to_map(
        safe_vector(client, f"max_over_time(({qps_expr})[{window}:{subquery_step}])"),
        ("method", "path"),
    )
    qps_min = vector_to_map(
        safe_vector(client, f"min_over_time(({qps_expr})[{window}:{subquery_step}])"),
        ("method", "path"),
    )
    p50 = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.50, sum by (le, method, path)(rate(http_request_duration_seconds_bucket[{window}])))",
        ),
        ("method", "path"),
    )
    p95 = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.95, sum by (le, method, path)(rate(http_request_duration_seconds_bucket[{window}])))",
        ),
        ("method", "path"),
    )
    p99 = vector_to_map(
        safe_vector(
            client,
            f"histogram_quantile(0.99, sum by (le, method, path)(rate(http_request_duration_seconds_bucket[{window}])))",
        ),
        ("method", "path"),
    )

    def normalize_http_key(raw_key: Any) -> Tuple[str, str]:
        if _is_http_pair(raw_key):
            method_raw, path_raw = raw_key
            return (str(method_raw or "-"), str(path_raw or "-"))
        return ("-", str(raw_key))

    def normalize_dict(source: Mapping[Any, Optional[float]]) -> Dict[Tuple[str, str], Optional[float]]:
        normalized: Dict[Tuple[str, str], Optional[float]] = {}
        for raw_key, value in source.items():
            normalized[normalize_http_key(raw_key)] = value
        return normalized

    qps_mean_norm = normalize_dict(cast(Mapping[Any, Optional[float]], qps_mean))
    qps_max_norm = normalize_dict(cast(Mapping[Any, Optional[float]], qps_max))
    qps_min_norm = normalize_dict(cast(Mapping[Any, Optional[float]], qps_min))
    p50_norm = normalize_dict(cast(Mapping[Any, Optional[float]], p50))
    p95_norm = normalize_dict(cast(Mapping[Any, Optional[float]], p95))
    p99_norm = normalize_dict(cast(Mapping[Any, Optional[float]], p99))

    path_stats: List[HttpPathStats] = []
    for method_path in sorted(qps_mean_norm.keys()):
        method_label, path_label = method_path
        path_stats.append(
            HttpPathStats(
                method_label,
                path_label,
                qps_mean_norm.get(method_path, 0.0) or 0.0,
                qps_max_norm.get(method_path),
                qps_min_norm.get(method_path),
                p50_norm.get(method_path),
                p95_norm.get(method_path),
                p99_norm.get(method_path),
            )
        )
    overall: StatsSummary = {
        "qps_mean": safe_scalar(
            client,
            f"avg_over_time((sum(rate(http_requests_total[{rate_window}])))" f"[{window}:{subquery_step}])",
        )
        or 0.0,
        "qps_max": safe_scalar(
            client,
            f"max_over_time((sum(rate(http_requests_total[{rate_window}])))" f"[{window}:{subquery_step}])",
        ),
        "qps_min": safe_scalar(
            client,
            f"min_over_time((sum(rate(http_requests_total[{rate_window}])))" f"[{window}:{subquery_step}])",
        ),
        "p50": safe_scalar(
            client, f"histogram_quantile(0.50, sum by (le)(rate(http_request_duration_seconds_bucket[{window}])))"
        )
        or 0.0,
        "p95": safe_scalar(
            client, f"histogram_quantile(0.95, sum by (le)(rate(http_request_duration_seconds_bucket[{window}])))"
        )
        or 0.0,
        "p99": safe_scalar(
            client, f"histogram_quantile(0.99, sum by (le)(rate(http_request_duration_seconds_bucket[{window}])))"
        )
        or 0.0,
    }
    return path_stats, overall


# ---------------------------------------------------------------------------
# Part 4 – diagnostics
# ---------------------------------------------------------------------------


def gather_diagnostics(client: PrometheusClient, window: str) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {}
    diagnostics["mongo_ops"] = vector_to_map(
        safe_vector(
            client,
            f"sum by (operation)(rate(mongo_operation_total{{operation!='ensure_collection'}}[{window}]))",
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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


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
    if value < 0.5:
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    window = format_window(duration_seconds)
    rate_window = compute_rate_window(duration_seconds)
    subquery_step = compute_subquery_step(duration_seconds)

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
    rows: List[List[str]] = []
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
        rows.append([item.name, count_str, fmt_rate(per_sec_value)])
    lines.extend(
        section(
            "Rollout / Attempt / Span / Resource / Worker Throughput",
            render_table(["Collection", "Count", "Per Sec"], rows),
        )
    )

    # Store internals
    store_stats, store_overall = gather_store_methods(client, window, rate_window, subquery_step)
    store_rows: List[List[str]] = [
        [
            stat.method,
            fmt_rate(stat.ops_mean),
            fmt_rate(stat.ops_max),
            fmt_rate(stat.ops_min),
            fmt_latency(stat.p50),
            fmt_latency(stat.p95),
            fmt_latency(stat.p99),
        ]
        for stat in store_stats
    ]
    store_lines = render_table(
        ["Method", "Mean Ops/s", "Max Ops/s", "Min Ops/s", "P50", "P95", "P99"],
        store_rows,
    )
    store_lines.append(
        f"Overall: mean={fmt_rate(store_overall['ops_mean'])}, "
        f"max={fmt_rate(store_overall['ops_max'])}, "
        f"min={fmt_rate(store_overall['ops_min'])}, "
        f"P50={fmt_latency(store_overall['p50'])}, "
        f"P95={fmt_latency(store_overall['p95'])}, "
        f"P99={fmt_latency(store_overall['p99'])}"
    )
    lines.extend(section("CollectionBasedLightningStore", store_lines))

    rollout_outcomes = gather_rollout_outcomes(client, window, rate_window)
    rollout_rows = [
        [
            stat.status,
            fmt_rate(stat.rate),
            fmt_latency(stat.p25),
            fmt_latency(stat.p50),
            fmt_latency(stat.p75),
            fmt_latency(stat.max_latency),
        ]
        for stat in rollout_outcomes
    ]
    lines.extend(
        section("Rollout Outcomes", render_table(["Status", "Rate", "P25", "P50", "P75", "Max"], rollout_rows))
    )

    # HTTP traffic
    http_paths, http_overall = gather_http_paths(client, window, rate_window, subquery_step)
    http_rows: List[List[str]] = [
        [
            stat.method,
            stat.path,
            fmt_rate(stat.qps_mean),
            fmt_rate(stat.qps_max),
            fmt_rate(stat.qps_min),
            fmt_latency(stat.p50),
            fmt_latency(stat.p95),
            fmt_latency(stat.p99),
        ]
        for stat in http_paths
    ]
    http_lines = render_table(
        ["Method", "Path", "Mean Req/s", "Max Req/s", "Min Req/s", "P50", "P95", "P99"], http_rows
    )
    http_lines.append(
        f"Overall HTTP: mean={fmt_rate(http_overall['qps_mean'])}, "
        f"max={fmt_rate(http_overall['qps_max'])}, "
        f"min={fmt_rate(http_overall['qps_min'])}, "
        f"P50={fmt_latency(http_overall['p50'])}, "
        f"P95={fmt_latency(http_overall['p95'])}, "
        f"P99={fmt_latency(http_overall['p99'])}"
    )
    lines.extend(section("HTTP Endpoints", http_lines))

    # Diagnostics
    diag = gather_diagnostics(client, window)
    diagnostics_blocks: List[List[str]] = []

    mongo_ops = cast(Dict[str, float], diag.get("mongo_ops", {}))
    mongo_ops_rows = [
        [operation or "-", fmt_rate(rate)]
        for operation, rate in sorted(mongo_ops.items(), key=lambda item: str(item[0]))
    ]
    diagnostics_blocks.append(render_table(["Mongo Operation", "Ops/s"], mongo_ops_rows))

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
