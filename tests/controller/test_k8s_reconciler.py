# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for controller manifests; no cluster or GPU is required."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from agentlightning.controller.k8s_reconciler import MANAGED_BY_SELECTOR, K8sReconciler, build_job_spec
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus


def _reconciler() -> K8sReconciler:
    config = OmegaConf.create(
        {
            "k8s_runner": {
                "namespace": "default",
                "poll_interval": 0.01,
            }
        }
    )
    reconciler = K8sReconciler(AsyncMock(), config)
    reconciler._get_k8s_api = AsyncMock(return_value=object())  # type: ignore[method-assign]
    reconciler._reconcile_once = AsyncMock()  # type: ignore[method-assign]
    return reconciler


def _idle_watch(entered: asyncio.Event, closed: asyncio.Event):
    async def watch(*args: object, **kwargs: object):
        del args, kwargs
        try:
            entered.set()
            await asyncio.Event().wait()
            yield None
        finally:
            closed.set()

    return watch


@pytest.mark.asyncio
async def test_stop_closes_idle_k8s_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    reconciler = _reconciler()
    monkeypatch.setattr("agentlightning.controller.k8s_reconciler.kr8s.asyncio.watch", _idle_watch(entered, closed))
    running = asyncio.create_task(reconciler.run())
    await asyncio.wait_for(entered.wait(), timeout=1)

    try:
        reconciler.stop()
        await asyncio.wait_for(asyncio.shield(running), timeout=0.5)
    finally:
        if not running.done():
            running.cancel()
            await asyncio.gather(running, return_exceptions=True)

    assert closed.is_set()


@pytest.mark.asyncio
async def test_external_cancellation_closes_idle_k8s_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    reconciler = _reconciler()
    monkeypatch.setattr("agentlightning.controller.k8s_reconciler.kr8s.asyncio.watch", _idle_watch(entered, closed))
    running = asyncio.create_task(reconciler.run())
    await asyncio.wait_for(entered.wait(), timeout=1)

    running.cancel()
    await asyncio.wait_for(running, timeout=1)

    assert closed.is_set()


@pytest.mark.asyncio
async def test_worker_error_propagates_and_closes_idle_k8s_watch(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    closed = asyncio.Event()
    reconciler = _reconciler()
    monkeypatch.setattr("agentlightning.controller.k8s_reconciler.kr8s.asyncio.watch", _idle_watch(entered, closed))

    async def fail_after_watch_starts() -> None:
        await entered.wait()
        raise RuntimeError("periodic worker failed")

    monkeypatch.setattr(reconciler, "_periodic_reconcile_loop", fail_after_watch_starts)

    with pytest.raises(RuntimeError, match="periodic worker failed"):
        await asyncio.wait_for(reconciler.run(), timeout=1)

    assert closed.is_set()


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
