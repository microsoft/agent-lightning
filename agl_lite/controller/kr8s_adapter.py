"""Kr8s adapter — real K8s client implementing the K8sClient protocol.

Wraps kr8s async API to satisfy the K8sClient protocol defined in reconciler.py.
Used in production; tests use mocks instead.
"""

from __future__ import annotations

from typing import Any

import kr8s.asyncio
from kr8s.asyncio import objects as k8s_objects


class Kr8sJobWatcher:
    """Async iterator of (event_type, job_dict) tuples from kr8s watch."""

    def __init__(self, namespace: str, label_selector: str, api: kr8s.asyncio.Api) -> None:
        self._watcher = kr8s.asyncio.watch(
            "jobs",
            namespace=namespace,
            label_selector=label_selector,
            api=api,
        )

    def __aiter__(self) -> Kr8sJobWatcher:
        return self

    async def __anext__(self) -> tuple[str, dict[str, Any]]:
        event_type, obj = await self._watcher.__anext__()
        return event_type, obj.raw


class Kr8sClient:
    """Real K8s client — implements K8sClient protocol using kr8s.

    Args:
        namespace: Default namespace (used for API init).
    """

    def __init__(self, namespace: str = "default") -> None:
        self._namespace = namespace
        self._api: kr8s.asyncio.Api | None = None

    async def _get_api(self) -> kr8s.asyncio.Api:
        if self._api is None:
            self._api = await kr8s.asyncio.api()
        return self._api

    async def create_job(self, manifest: dict[str, Any]) -> None:
        api = await self._get_api()
        job = k8s_objects.Job(manifest, api=api)
        await job.async_create()

    async def delete_job(self, name: str, namespace: str) -> None:
        api = await self._get_api()
        try:
            job = await k8s_objects.Job.async_get(name, namespace=namespace, api=api)
            await job.async_delete(propagation_policy="Background")
        except kr8s.NotFoundError:
            pass  # Already deleted — idempotent.

    async def get_job(self, name: str, namespace: str) -> dict[str, Any] | None:
        api = await self._get_api()
        try:
            job = await k8s_objects.Job.async_get(name, namespace=namespace, api=api)
            return job.raw
        except kr8s.NotFoundError:
            return None

    async def list_jobs(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        api = await self._get_api()
        jobs = [j async for j in k8s_objects.Job.async_list(namespace=namespace, label_selector=label_selector, api=api)]
        return [j.raw for j in jobs]

    async def list_pods(self, namespace: str, label_selector: str) -> list[dict[str, Any]]:
        api = await self._get_api()
        pods = [p async for p in k8s_objects.Pod.async_list(namespace=namespace, label_selector=label_selector, api=api)]
        return [p.raw for p in pods]

    async def watch_jobs(self, namespace: str, label_selector: str) -> Kr8sJobWatcher:
        api = await self._get_api()
        return Kr8sJobWatcher(namespace=namespace, label_selector=label_selector, api=api)
