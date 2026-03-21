"""Integration tests for Kr8sClient against real minikube.

Requires:
  - Running minikube cluster
  - Namespace 'agl-test' exists: kubectl create namespace agl-test

Run with:
    uv run pytest tests/controller/test_kr8s_adapter.py -v

These tests create/delete real K8s resources in the 'agl-test' namespace.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from agl_lite.controller.kr8s_adapter import Kr8sClient

NAMESPACE = "agl-test"
LABEL_SELECTOR = "test-suite=kr8s-adapter"


def _unique_name() -> str:
    return f"test-kr8s-{uuid.uuid4().hex[:8]}"


def _make_job_manifest(name: str, succeed: bool = True) -> dict:
    """Build a minimal Job manifest for testing."""
    cmd = ["echo", "hello"] if succeed else ["false"]
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": name,
            "namespace": NAMESPACE,
            "labels": {
                "app.kubernetes.io/managed-by": "agl-lite",
                "test-suite": "kr8s-adapter",
            },
        },
        "spec": {
            "backoffLimit": 0,
            "ttlSecondsAfterFinished": 120,
            "template": {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/managed-by": "agl-lite",
                        "test-suite": "kr8s-adapter",
                    },
                },
                "spec": {
                    "restartPolicy": "Never",
                    "containers": [
                        {
                            "name": "test",
                            "image": "busybox:latest",
                            "command": cmd,
                        }
                    ],
                },
            },
        },
    }


class TestKr8sClient:
    """Integration tests — require running minikube + agl-test namespace."""

    async def test_create_and_get_job(self) -> None:
        k8s = Kr8sClient(namespace=NAMESPACE)
        name = _unique_name()
        await k8s.create_job(_make_job_manifest(name))

        job = await k8s.get_job(name, NAMESPACE)
        assert job is not None
        assert job["metadata"]["name"] == name

    async def test_get_job_not_found(self) -> None:
        k8s = Kr8sClient(namespace=NAMESPACE)
        job = await k8s.get_job("nonexistent-job-xyz", NAMESPACE)
        assert job is None

    async def test_delete_job(self) -> None:
        k8s = Kr8sClient(namespace=NAMESPACE)
        name = _unique_name()
        await k8s.create_job(_make_job_manifest(name))

        job = await k8s.get_job(name, NAMESPACE)
        assert job is not None

        await k8s.delete_job(name, NAMESPACE)
        await asyncio.sleep(2)

        job = await k8s.get_job(name, NAMESPACE)
        if job is not None:
            assert "deletionTimestamp" in job.get("metadata", {})

    async def test_delete_job_idempotent(self) -> None:
        """Deleting a nonexistent Job should not raise."""
        k8s = Kr8sClient(namespace=NAMESPACE)
        await k8s.delete_job("nonexistent-job-xyz", NAMESPACE)

    async def test_list_jobs(self) -> None:
        k8s = Kr8sClient(namespace=NAMESPACE)
        name = _unique_name()
        await k8s.create_job(_make_job_manifest(name))

        jobs = await k8s.list_jobs(NAMESPACE, LABEL_SELECTOR)
        names = [j["metadata"]["name"] for j in jobs]
        assert name in names

    async def test_list_pods(self) -> None:
        k8s = Kr8sClient(namespace=NAMESPACE)
        name = _unique_name()
        await k8s.create_job(_make_job_manifest(name))

        # Wait for pod to be created.
        pods = []
        for _ in range(15):
            pods = await k8s.list_pods(NAMESPACE, f"job-name={name}")
            if pods:
                break
            await asyncio.sleep(1)

        assert len(pods) >= 1
        assert pods[0]["metadata"]["labels"]["job-name"] == name

    async def test_watch_jobs(self) -> None:
        """Watch should yield ADDED event for a new Job."""
        k8s = Kr8sClient(namespace=NAMESPACE)
        watcher = await k8s.watch_jobs(NAMESPACE, LABEL_SELECTOR)

        name = _unique_name()

        async def create_after_delay():
            await asyncio.sleep(1)
            # Use a fresh client to avoid sharing connections.
            k8s2 = Kr8sClient(namespace=NAMESPACE)
            await k8s2.create_job(_make_job_manifest(name))

        create_task = asyncio.create_task(create_after_delay())

        events_seen = []
        try:
            async with asyncio.timeout(15):
                async for event_type, job_dict in watcher:
                    if job_dict["metadata"]["name"] == name:
                        events_seen.append(event_type)
                        if len(events_seen) >= 2:
                            break
        except TimeoutError:
            pass

        await create_task
        assert "ADDED" in events_seen

    async def test_job_completes_successfully(self) -> None:
        """A Job with 'echo hello' should reach Complete condition."""
        k8s = Kr8sClient(namespace=NAMESPACE)
        name = _unique_name()
        await k8s.create_job(_make_job_manifest(name, succeed=True))

        for _ in range(30):
            job = await k8s.get_job(name, NAMESPACE)
            if job:
                conditions = job.get("status", {}).get("conditions", [])
                for c in conditions:
                    if c["type"] == "Complete" and c["status"] == "True":
                        return
            await asyncio.sleep(1)

        pytest.fail(f"Job {name} did not complete within 30s")

    async def test_job_fails(self) -> None:
        """A Job with 'false' command should reach Failed condition."""
        k8s = Kr8sClient(namespace=NAMESPACE)
        name = _unique_name()
        await k8s.create_job(_make_job_manifest(name, succeed=False))

        for _ in range(30):
            job = await k8s.get_job(name, NAMESPACE)
            if job:
                conditions = job.get("status", {}).get("conditions", [])
                for c in conditions:
                    if c["type"] == "Failed" and c["status"] == "True":
                        return
            await asyncio.sleep(1)

        pytest.fail(f"Job {name} did not fail within 30s")
