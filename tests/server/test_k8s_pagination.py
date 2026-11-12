# Copyright (c) Microsoft. All rights reserved.

"""Exercise controller pagination against the real rollout HTTP API."""

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller import k8s_reconciler
from agentlightning.controller.k8s_reconciler import K8sReconciler
from agentlightning.schemas import RolloutState

JOB_TEMPLATE = """
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


def _controller(api: AgentLightningAsyncClient, max_jobs: int = 1000) -> K8sReconciler:
    return K8sReconciler(
        api,
        OmegaConf.create(
            {
                "agl_server": {"url": "http://server", "key": ""},
                "k8s_runner": {"namespace": "default", "ttl_after_finished": 60, "max_jobs_per_minute": max_jobs},
            }
        ),
    )


async def _enqueue(api: AgentLightningAsyncClient, ids: list[str], *, running: bool = False) -> None:
    response = await api.post(
        "/api/rollouts",
        json=[{"rollout_id": rid, "input": {}, "config": {"k8s": {"job_template": JOB_TEMPLATE}}} for rid in ids],
    )
    response.raise_for_status()
    if running:
        for rid in ids:
            response = await api.patch(f"/api/rollouts/{rid}", json={"status": {"state": "running"}})
            response.raise_for_status()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_jobs", [2, 1000])
async def test_k8s_reconciles_beyond_first_page(
    app: FastAPI, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch, max_jobs: int
) -> None:
    jobs: dict[str, dict[str, Any]] = {
        f"agl-rollout-running-{i}": {"metadata": {"name": f"agl-rollout-running-{i}"}, "status": {"active": 1}}
        for i in range(500)
    }

    class Job:
        def __init__(self, manifest: dict[str, Any], **kwargs: Any) -> None:
            self.raw = manifest

        @classmethod
        async def async_list(cls, **kwargs: Any) -> AsyncIterator["Job"]:
            for manifest in list(jobs.values()):
                yield cls(manifest)

        async def async_create(self) -> None:
            name = self.raw["metadata"]["name"]
            assert name not in jobs, "an existing Job must not be created twice"
            jobs[name] = self.raw

    monkeypatch.setattr(k8s_reconciler.k8s_objects, "Job", Job)
    async with AgentLightningAsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://server", headers=auth_headers
    ) as api:
        await _enqueue(api, [f"running-{i}" for i in range(500)] + ["orphan"], running=True)
        await _enqueue(api, ["queued-0", "queued-1", "queued-2"])
        controller = _controller(api, max_jobs)
        monkeypatch.setattr(controller, "_get_k8s_api", AsyncMock(return_value=object()))

        await controller._reconcile_once()
        await controller._reconcile_once()

        orphan = (await api.get("/api/rollouts/orphan")).json()["rollout"]
        assert orphan["status"]["state"] == "failed"
        assert orphan["status"]["error_message"] == "Job disappeared"
        expected = {"agl-rollout-queued-0", "agl-rollout-queued-1"}
        if max_jobs > 2:
            expected.add("agl-rollout-queued-2")
        assert {name for name in jobs if "queued" in name} == expected
        for name in expected:
            rollout = (await api.get(f"/api/rollouts/{name.removeprefix('agl-rollout-')}")).json()["rollout"]
            assert rollout["status"]["state"] == "running"


@pytest.mark.asyncio
async def test_reconcile_reads_all_pages_before_removing_active_rows(
    app: FastAPI, auth_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    async def no_jobs(**kwargs: Any) -> AsyncIterator[Any]:
        for job in []:
            yield job

    monkeypatch.setattr(k8s_reconciler.k8s_objects.Job, "async_list", no_jobs)
    async with AgentLightningAsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://server", headers=auth_headers
    ) as api:
        await _enqueue(api, [f"orphan-{i}" for i in range(501)], running=True)
        controller = _controller(api)
        monkeypatch.setattr(controller, "_get_k8s_api", AsyncMock(return_value=object()))
        await controller._reconcile_once()

        remaining = await api.get("/api/rollouts", params={"state_in": "running"})
        assert remaining.json() == []
        terminal = await api.get("/api/rollouts/terminal")
        assert terminal.json()["total_terminal"] == 501


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 3, 6, 7])
async def test_controller_query_handles_empty_exact_and_partial_pages(
    app: FastAPI, auth_headers: dict[str, str], count: int
) -> None:
    async with AgentLightningAsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://server", headers=auth_headers
    ) as api:
        ids = [f"queued-{i}" for i in range(count)]
        await _enqueue(api, ids)
        await _enqueue(api, ["excluded"], running=True)
        result = await _controller(api)._query_rollouts(state_in=[RolloutState.QUEUING], limit=3)
        assert [row.rollout_id for row in result] == ids


@pytest.mark.asyncio
async def test_rollout_offset_applies_after_state_filter(app: FastAPI, auth_headers: dict[str, str]) -> None:
    async with AgentLightningAsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://server", headers=auth_headers
    ) as api:
        await _enqueue(api, ["excluded"], running=True)
        await _enqueue(api, ["q0", "q1", "q2"])
        page = await api.get("/api/rollouts", params={"state_in": "queuing", "limit": 1, "offset": 1})
        assert [row["rollout_id"] for row in page.json()] == ["q1"]
        end = await api.get("/api/rollouts", params={"state_in": "queuing", "offset": 3})
        assert end.json() == []
        invalid = await api.get("/api/rollouts", params={"state_in": "queuing", "offset": -1})
        assert invalid.status_code == 422


@pytest.mark.asyncio
async def test_controller_does_not_loop_when_older_server_ignores_offset(
    app: FastAPI, auth_headers: dict[str, str]
) -> None:
    transport = httpx.ASGITransport(app=app)
    list_requests = 0

    async def legacy_server(request: httpx.Request) -> httpx.Response:
        nonlocal list_requests
        if request.method == "GET" and request.url.path == "/api/rollouts":
            list_requests += 1
            assert list_requests <= 2, "offset-ignorant servers must not cause an endless pagination loop"
            request.url = request.url.copy_remove_param("offset")
        return await transport.handle_async_request(request)

    async with AgentLightningAsyncClient(
        transport=httpx.MockTransport(legacy_server), base_url="http://server", headers=auth_headers
    ) as api:
        await _enqueue(api, ["q0", "q1", "q2", "q3"])
        rows = await _controller(api)._query_rollouts(state_in=[RolloutState.QUEUING], limit=3)
        assert [row.rollout_id for row in rows] == ["q0", "q1", "q2"]
