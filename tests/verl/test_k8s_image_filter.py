# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import yaml
from omegaconf import DictConfig, OmegaConf

from agentlightning.schemas import K8sImageReadinessSnapshot
from agentlightning.verl.k8s_image_filter import (
    filter_dataset_by_images,
    prepare_datasets,
    wait_for_k8s_image_readiness,
)

TEMPLATE = """
apiVersion: batch/v1
kind: Job
metadata: {}
spec:
  template:
    spec:
      containers:
        - name: agent
          image: {{ (input.image_name ~ ":openai") | yaml_escape }}
"""


@pytest.fixture
def enabled_config(tmp_path: Path) -> DictConfig:
    template_path = tmp_path / "job.yaml"
    template_path.write_text(TEMPLATE)
    return OmegaConf.create(
        {
            "agentlightning": {
                "agl_base_url": "http://server:8080",
                "agl_key": "secret",
                "k8s": {
                    "job_template_path": str(template_path),
                    "filter_unavailable_images": True,
                    "image_readiness_timeout_seconds": 60,
                },
            }
        }
    )


def _snapshot(images: list[str]) -> K8sImageReadinessSnapshot:
    return K8sImageReadinessSnapshot(
        images=images,
        node_count=1,
        observed_at=100.0,
        expires_at=130.0,
    )


def test_filter_dataset_keeps_available_rows_without_deduplicating() -> None:
    rows = [
        {"instance_id": "missing", "image_name": "swebench/missing"},
        {"instance_id": "ready", "image_name": "swebench/ready"},
        {"instance_id": "ready", "image_name": "swebench/ready"},
    ]

    kept, report = filter_dataset_by_images(
        rows,
        split="train",
        job_template=TEMPLATE,
        ready_images={"docker.io/swebench/ready:openai"},
    )

    assert kept == [rows[1], rows[2]]
    assert report.source_count == 3
    assert report.kept_count == 2
    assert report.dropped_count == 1
    assert report.dropped_data_ids == ["missing"]
    assert report.missing_image_counts == {"docker.io/swebench/missing:openai": 1}


def test_prepare_datasets_filters_before_validation_cap(
    enabled_config: DictConfig,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "agentlightning.verl.k8s_image_filter._fetch_snapshot",
        lambda _config: _snapshot(["docker.io/swebench/ready:openai"]),
    )
    val = [
        {"instance_id": "missing", "image_name": "swebench/missing"},
        {"instance_id": "ready-1", "image_name": "swebench/ready"},
        {"instance_id": "ready-2", "image_name": "swebench/ready"},
    ]

    prepared = prepare_datasets(
        enabled_config,
        [{"instance_id": "train", "image_name": "swebench/ready"}],
        val,
        max_val_instances=1,
    )

    assert [row["instance_id"] for row in prepared.val] == ["ready-1"]
    assert prepared.val_report is not None
    assert prepared.val_report.source_count == 3
    assert prepared.val_report.kept_count == 2
    assert prepared.val_report.dropped_count == 1


def test_prepare_datasets_disabled_bypasses_readiness_and_template(monkeypatch) -> None:
    monkeypatch.setattr(
        "agentlightning.verl.k8s_image_filter._fetch_snapshot",
        lambda _config: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    config = OmegaConf.create(
        {
            "agentlightning": {
                "k8s": {
                    "job_template_path": "/does/not/exist.yaml",
                    "filter_unavailable_images": False,
                    "image_readiness_timeout_seconds": 60,
                }
            }
        }
    )
    train = [{"instance_id": "train"}]
    val = [{"instance_id": "val-1"}, {"instance_id": "val-2"}]

    prepared = prepare_datasets(config, train, val, max_val_instances=1)

    assert prepared.train == train
    assert prepared.val == val[:1]
    assert prepared.train_report is None
    assert prepared.val_report is None
    assert prepared.readiness is None


def test_enabled_filter_rejects_missing_job_template(enabled_config: DictConfig) -> None:
    enabled_config.agentlightning.k8s.job_template_path = None

    with pytest.raises(ValueError, match="job_template_path"):
        prepare_datasets(enabled_config, [{"id": 1}], [{"id": 2}], max_val_instances=None)


def test_enabled_filter_rejects_empty_train_after_filter(
    enabled_config: DictConfig,
    monkeypatch,
) -> None:
    monkeypatch.setattr("agentlightning.verl.k8s_image_filter._fetch_snapshot", lambda _config: _snapshot([]))

    with pytest.raises(ValueError, match=r"train dataset.*empty"):
        prepare_datasets(
            enabled_config,
            [{"instance_id": "missing", "image_name": "swebench/missing"}],
            [{"instance_id": "missing-val", "image_name": "swebench/missing"}],
            max_val_instances=None,
        )


def test_enabled_filter_surfaces_malformed_job_yaml(
    enabled_config: DictConfig,
    monkeypatch,
) -> None:
    Path(str(enabled_config.agentlightning.k8s.job_template_path)).write_text("apiVersion: batch/v1\nkind: [")
    monkeypatch.setattr(
        "agentlightning.verl.k8s_image_filter._fetch_snapshot",
        lambda _config: _snapshot(["docker.io/swebench/ready:openai"]),
    )

    with pytest.raises(yaml.YAMLError):
        prepare_datasets(
            enabled_config,
            [{"instance_id": "train", "image_name": "swebench/ready"}],
            [{"instance_id": "val", "image_name": "swebench/ready"}],
            max_val_instances=None,
        )


def test_wait_for_readiness_times_out_with_last_503_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            request=request,
            json={"detail": "K8s image readiness snapshot has expired"},
        )

    with (
        httpx.Client(base_url="http://server:8080", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(RuntimeError, match="snapshot has expired"),
    ):
        wait_for_k8s_image_readiness(client, timeout_seconds=0)


def test_wait_for_readiness_does_not_retry_auth_failure() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401, request=request, json={"detail": "invalid key"})

    with (
        httpx.Client(base_url="http://server:8080", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(httpx.HTTPStatusError),
    ):
        wait_for_k8s_image_readiness(client, timeout_seconds=60)

    assert requests == 1


@pytest.mark.parametrize("timeout_seconds", [-1.0, float("nan"), float("inf")])
def test_wait_for_readiness_rejects_invalid_timeout(timeout_seconds: float) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request, json={"detail": "not ready"})

    with (
        httpx.Client(base_url="http://server:8080", transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="finite non-negative"),
    ):
        wait_for_k8s_image_readiness(client, timeout_seconds=timeout_seconds)
