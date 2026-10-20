# Copyright (c) Microsoft. All rights reserved.

"""Pre-Ray dataset filtering using CPU-side Kubernetes image readiness."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agentlightning.client import AgentLightningSyncClient
from agentlightning.k8s import extract_pod_images, normalize_image_reference, render_job_template
from agentlightning.schemas import K8sImageReadinessSnapshot

__all__ = [
    "DatasetFilterReport",
    "PreparedDatasets",
    "filter_dataset_by_images",
    "prepare_datasets",
    "wait_for_k8s_image_readiness",
]


@dataclass
class DatasetFilterReport:
    split: str
    source_count: int
    kept_count: int
    dropped_count: int
    dropped_data_ids: list[str]
    missing_image_counts: dict[str, int]


@dataclass
class PreparedDatasets:
    train: list[Any]
    val: list[Any]
    train_report: DatasetFilterReport | None
    val_report: DatasetFilterReport | None
    readiness: K8sImageReadinessSnapshot | None


def filter_dataset_by_images(
    dataset: Sequence[Any],
    *,
    split: str,
    job_template: str,
    ready_images: set[str] | frozenset[str],
) -> tuple[list[Any], DatasetFilterReport]:
    """Return rows whose rendered Job images are all available."""
    ready = {normalize_image_reference(image) for image in ready_images}
    kept: list[Any] = []
    dropped_ids: list[str] = []
    missing_counts: Counter[str] = Counter()

    for index, row in enumerate(dataset):
        job = render_job_template(
            job_template,
            job_name=f"agl-image-check-{index}",
            input_data=row,
        )
        missing = sorted(extract_pod_images(job) - ready)
        if not missing:
            kept.append(row)
            continue

        row_id = str(row.get("data_id") or row.get("instance_id") or index) if isinstance(row, Mapping) else str(index)
        if len(dropped_ids) < 20:
            dropped_ids.append(row_id)
        missing_counts.update(missing)

    report = DatasetFilterReport(
        split=split,
        source_count=len(dataset),
        kept_count=len(kept),
        dropped_count=len(dataset) - len(kept),
        dropped_data_ids=dropped_ids,
        missing_image_counts=dict(sorted(missing_counts.items())),
    )
    return kept, report


def wait_for_k8s_image_readiness(
    client: httpx.Client,
    *,
    timeout_seconds: float,
) -> K8sImageReadinessSnapshot:
    """Wait for a fresh server-validated readiness snapshot."""
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout_seconds must be a finite non-negative number")

    endpoint = "/api/runner-readiness/k8s"
    deadline = time.monotonic() + timeout_seconds
    last_detail = "readiness has not been published"

    while True:
        remaining = deadline - time.monotonic()
        try:
            response = client.get(endpoint, timeout=min(5.0, max(0.0, remaining)))
        except httpx.RequestError as exc:
            last_detail = str(exc)
        else:
            if response.status_code == 200:
                return K8sImageReadinessSnapshot.model_validate(response.json())
            if response.status_code != 503:
                response.raise_for_status()
            try:
                payload = response.json()
                last_detail = str(payload.get("detail", response.text))
            except ValueError:
                last_detail = response.text

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"Timed out waiting for {endpoint}: {last_detail}")
        time.sleep(min(1.0, remaining))


def _fetch_snapshot(config: Any) -> K8sImageReadinessSnapshot:
    with AgentLightningSyncClient(
        base_url=str(config.agentlightning.agl_base_url),
        key=str(config.agentlightning.agl_key or "") or None,
        max_retries=0,
        timeout=5.0,
    ) as client:
        return wait_for_k8s_image_readiness(
            client,
            timeout_seconds=float(config.agentlightning.k8s.image_readiness_timeout_seconds),
        )


def _require_non_empty(train_rows: Sequence[Any], val_rows: Sequence[Any]) -> None:
    if not train_rows:
        raise ValueError("train dataset is empty after image filtering")
    if not val_rows:
        raise ValueError("validation dataset is empty after image filtering")


def prepare_datasets(
    config: Any,
    train_dataset: Sequence[Any],
    val_dataset: Sequence[Any],
    *,
    max_val_instances: int | None,
) -> PreparedDatasets:
    """Prepare in-memory datasets while preserving disabled-mode behavior."""
    train_rows = list(train_dataset)
    val_rows = list(val_dataset)
    enabled = bool(config.agentlightning.k8s.filter_unavailable_images)

    if not enabled:
        if max_val_instances:
            val_rows = val_rows[:max_val_instances]
        return PreparedDatasets(train_rows, val_rows, None, None, None)

    template_path = config.agentlightning.k8s.job_template_path
    if not template_path:
        raise ValueError("agentlightning.k8s.job_template_path is required when filter_unavailable_images=true")

    job_template = Path(str(template_path)).read_text(encoding="utf-8")
    snapshot = _fetch_snapshot(config)
    train_rows, train_report = filter_dataset_by_images(
        train_rows,
        split="train",
        job_template=job_template,
        ready_images=set(snapshot.images),
    )
    val_rows, val_report = filter_dataset_by_images(
        val_rows,
        split="val",
        job_template=job_template,
        ready_images=set(snapshot.images),
    )
    if max_val_instances:
        val_rows = val_rows[:max_val_instances]
    _require_non_empty(train_rows, val_rows)
    return PreparedDatasets(
        train=train_rows,
        val=val_rows,
        train_report=train_report,
        val_report=val_report,
        readiness=snapshot,
    )
