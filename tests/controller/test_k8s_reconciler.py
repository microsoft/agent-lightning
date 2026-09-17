# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for the Kubernetes controller; no cluster or GPU is required."""

from __future__ import annotations

import asyncio
import time

import pytest
from kr8s.asyncio import objects as k8s_objects
from omegaconf import OmegaConf

from agentlightning.controller.k8s_reconciler import (
    MANAGED_BY_SELECTOR,
    K8sReconciler,
    build_job_spec,
    images_available_on_all_ready_nodes,
)
from agentlightning.schemas import (
    Rollout,
    RolloutConfig,
    RolloutK8sConfig,
    RolloutLifecycleStatus,
    RolloutState,
)


def test_build_job_spec_uses_agentlightning_labels() -> None:
    rollout = Rollout(
        rollout_id="test-id",
        input={"question": "1 + 1"},
        config=RolloutConfig(
            k8s=RolloutK8sConfig(
                job_template="""
apiVersion: batch/v1
kind: Job
metadata: {}
spec:
  template:
    spec:
      containers:
        - name: agent
          image: example-agent:latest
"""
            )
        ),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )
    config = OmegaConf.create(
        {
            "agl_server": {"url": "http://server:8080", "key": "secret"},
            "k8s_runner": {"namespace": "default", "ttl_after_finished": 60},
        }
    )

    manifest = build_job_spec(rollout, config)
    labels = manifest["metadata"]["labels"]

    assert MANAGED_BY_SELECTOR == "app.kubernetes.io/managed-by=agentlightning"
    assert labels == {
        "app.kubernetes.io/managed-by": "agentlightning",
        "agentlightning/rollout-id": "test-id",
        "agentlightning/attempt-id": "0",
    }
    env = {item["name"]: item["value"] for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["AGL_KEY"] == "secret"
    assert "/rollout/test-id/attempt/0/mode/train/" in env["AGL_OPENAI_BASE_URL"]


def _node(
    name: str,
    images: list[str],
    *,
    ready: bool = True,
    unschedulable: bool = False,
) -> dict:
    return {
        "metadata": {"name": name},
        "spec": {"unschedulable": unschedulable},
        "status": {
            "conditions": [{"type": "Ready", "status": "True" if ready else "False"}],
            "images": [{"names": [image]} for image in images],
        },
    }


def _controller_config(*, heartbeat: float = 5, lease: float = 30):
    return OmegaConf.create(
        {
            "agl_server": {"url": "http://server:8080", "key": "secret"},
            "k8s_runner": {
                "namespace": "default",
                "ttl_after_finished": 60,
                "max_jobs_per_minute": 100,
                "poll_interval": 5,
                "image_readiness": {
                    "enabled": True,
                    "heartbeat_seconds": heartbeat,
                    "lease_seconds": lease,
                },
            },
        }
    )


def test_images_available_on_all_ready_nodes_uses_intersection() -> None:
    images, node_count = images_available_on_all_ready_nodes(
        [
            _node("node-a", ["docker.io/shared:v1", "docker.io/only-a:v1"]),
            _node("node-b", ["docker.io/library/shared:v1", "docker.io/only-b:v1"]),
            _node("not-ready", ["docker.io/not-ready:v1"], ready=False),
            _node("cordoned", ["docker.io/cordoned:v1"], unschedulable=True),
        ]
    )

    assert node_count == 2
    assert images == frozenset({"docker.io/library/shared:v1"})


def test_images_available_on_all_ready_nodes_rejects_no_eligible_nodes() -> None:
    with pytest.raises(RuntimeError, match="no Ready schedulable Kubernetes nodes"):
        images_available_on_all_ready_nodes([_node("node-a", [], ready=False)])


@pytest.mark.parametrize(
    ("heartbeat", "lease"),
    [(0, 30), (30, 30), (31, 30), (5, 301)],
)
def test_reconciler_rejects_invalid_readiness_timing(heartbeat: float, lease: float) -> None:
    with pytest.raises(ValueError, match="heartbeat_seconds"):
        K8sReconciler(object(), _controller_config(heartbeat=heartbeat, lease=lease))


def test_publish_image_readiness_sends_normalized_snapshot(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    class Api:
        async def put(self, path: str, *, json: dict) -> Response:
            calls.append((path, json))
            return Response()

    reconciler = K8sReconciler(Api(), _controller_config())

    async def scan() -> tuple[frozenset[str], int]:
        return frozenset({"docker.io/swebench/repo:openai"}), 1

    monkeypatch.setattr(reconciler, "_scan_preloaded_images", scan)
    asyncio.run(reconciler._publish_image_readiness_once())

    assert calls == [
        (
            "/api/runner-readiness/k8s",
            {
                "images": ["docker.io/swebench/repo:openai"],
                "node_count": 1,
                "lease_seconds": 30.0,
            },
        )
    ]
    assert reconciler._ready_images == frozenset({"docker.io/swebench/repo:openai"})


def test_publish_failure_does_not_renew_local_cache(monkeypatch) -> None:
    class Response:
        def raise_for_status(self) -> None:
            raise RuntimeError("server unavailable")

    class Api:
        async def put(self, _path: str, *, json: dict) -> Response:
            return Response()

    reconciler = K8sReconciler(Api(), _controller_config())
    reconciler._ready_images = frozenset({"docker.io/old:v1"})
    reconciler._ready_images_expires_at = 123.0

    async def scan() -> tuple[frozenset[str], int]:
        return frozenset({"docker.io/new:v1"}), 1

    monkeypatch.setattr(reconciler, "_scan_preloaded_images", scan)
    with pytest.raises(RuntimeError, match="server unavailable"):
        asyncio.run(reconciler._publish_image_readiness_once())

    assert reconciler._ready_images == frozenset({"docker.io/old:v1"})
    assert reconciler._ready_images_expires_at == 123.0


def _image_rollout(*, require_preloaded_images: bool) -> Rollout:
    return Rollout(
        rollout_id="rollout-image-check",
        input={"image_name": "swebench/missing"},
        config=RolloutConfig(
            k8s=RolloutK8sConfig(
                job_template="""
apiVersion: batch/v1
kind: Job
metadata: {}
spec:
  template:
    spec:
      containers:
        - name: agent
          image: {{ (input.image_name ~ ":openai") | yaml_escape }}
""",
                require_preloaded_images=require_preloaded_images,
            )
        ),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )


def test_guarded_rollout_fails_before_job_creation_when_image_is_missing(monkeypatch) -> None:
    reconciler = K8sReconciler(object(), _controller_config())
    reconciler._ready_images = frozenset({"docker.io/swebench/ready:openai"})
    reconciler._ready_images_expires_at = time.monotonic() + 30
    patched_status: dict = {}

    async def capture_patch(_rollout_id: str, **status) -> bool:
        patched_status.update(status)
        return True

    async def unexpected_k8s_api():
        raise AssertionError("guarded missing image must not create a Job")

    monkeypatch.setattr(reconciler, "_patch_status", capture_patch)
    monkeypatch.setattr(reconciler, "_get_k8s_api", unexpected_k8s_api)

    asyncio.run(reconciler._create_job(_image_rollout(require_preloaded_images=True)))

    assert patched_status == {
        "state": RolloutState.FAILED,
        "error_message": "Required Kubernetes image(s) are not preloaded: docker.io/swebench/missing:openai",
    }


def test_guarded_rollout_checks_images_even_when_creation_is_rate_limited(monkeypatch) -> None:
    reconciler = K8sReconciler(object(), _controller_config())
    reconciler._ready_images = frozenset({"docker.io/swebench/ready:openai"})
    reconciler._ready_images_expires_at = time.monotonic() + 30
    reconciler._job_creation_timestamps.extend([time.monotonic()] * 100)
    patched_status: dict = {}

    async def capture_patch(_rollout_id: str, **status) -> bool:
        patched_status.update(status)
        return True

    async def unexpected_k8s_api():
        raise AssertionError("guarded missing image must not create a Job")

    monkeypatch.setattr(reconciler, "_patch_status", capture_patch)
    monkeypatch.setattr(reconciler, "_get_k8s_api", unexpected_k8s_api)

    asyncio.run(reconciler._create_job(_image_rollout(require_preloaded_images=True)))

    assert patched_status == {
        "state": RolloutState.FAILED,
        "error_message": "Required Kubernetes image(s) are not preloaded: docker.io/swebench/missing:openai",
    }


def test_guarded_rollout_fails_when_local_readiness_is_not_fresh(monkeypatch) -> None:
    reconciler = K8sReconciler(object(), _controller_config())
    reconciler._ready_images = None
    reconciler._ready_images_expires_at = 0.0
    patched_status: dict = {}

    async def capture_patch(_rollout_id: str, **status) -> bool:
        patched_status.update(status)
        return True

    async def unexpected_k8s_api():
        raise AssertionError("guarded stale readiness must not create a Job")

    monkeypatch.setattr(reconciler, "_patch_status", capture_patch)
    monkeypatch.setattr(reconciler, "_get_k8s_api", unexpected_k8s_api)

    asyncio.run(reconciler._create_job(_image_rollout(require_preloaded_images=True)))

    assert patched_status == {
        "state": RolloutState.FAILED,
        "error_message": "Fresh Kubernetes image readiness is unavailable",
    }


def test_unguarded_rollout_keeps_original_job_creation_path(monkeypatch) -> None:
    reconciler = K8sReconciler(object(), _controller_config())
    reconciler._ready_images = None
    created: list[dict] = []

    class FakeJob:
        def __init__(self, manifest: dict, *, api) -> None:
            created.append(manifest)

        async def async_create(self) -> None:
            return None

    async def fake_k8s_api():
        return object()

    monkeypatch.setattr(reconciler, "_get_k8s_api", fake_k8s_api)
    monkeypatch.setattr(k8s_objects, "Job", FakeJob)

    asyncio.run(reconciler._create_job(_image_rollout(require_preloaded_images=False)))

    assert len(created) == 1


def test_invalid_job_template_fails_rollout_instead_of_retrying(monkeypatch) -> None:
    rollout = _image_rollout(require_preloaded_images=False)
    assert rollout.config.k8s is not None
    rollout.config.k8s.job_template = "apiVersion: v1\nkind: Pod\nmetadata: {}\n"
    reconciler = K8sReconciler(object(), _controller_config())
    patched_status: dict = {}

    async def capture_patch(_rollout_id: str, **status) -> bool:
        patched_status.update(status)
        return True

    async def unexpected_k8s_api():
        raise AssertionError("invalid templates must not reach the Kubernetes API")

    monkeypatch.setattr(reconciler, "_patch_status", capture_patch)
    monkeypatch.setattr(reconciler, "_get_k8s_api", unexpected_k8s_api)

    asyncio.run(reconciler._create_job(rollout))

    assert patched_status == {
        "state": RolloutState.FAILED,
        "error_message": "Invalid Job spec: job template must render a Kubernetes Job",
    }
