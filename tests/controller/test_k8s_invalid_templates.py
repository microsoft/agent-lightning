# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for local K8s Job template validation."""

import time
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller import k8s_reconciler
from agentlightning.controller.k8s_reconciler import K8sReconciler
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus, RolloutState

INVALID_TEMPLATES = [
    "kind: Job\nspec: [\n",
    "kind: Job\nmetadata:\n  name: {{ job_name\n",
    "apiVersion: v1\nkind: Pod\nmetadata: {}\n",
]
VALID_TEMPLATE = "apiVersion: batch/v1\nkind: Job\nspec: {}\n"


def _reconciler() -> tuple[K8sReconciler, AsyncMock]:
    api = AsyncMock(spec=AgentLightningAsyncClient)
    api.patch.return_value = httpx.Response(200, request=httpx.Request("PATCH", "http://store"))
    config = OmegaConf.create(
        {
            "agl_server": {"url": "http://store", "key": ""},
            "k8s_runner": {
                "namespace": "default",
                "ttl_after_finished": 600,
                "max_jobs_per_minute": 100,
            },
        }
    )
    return K8sReconciler(api, config), api


def _rollout(
    template: str, *, rollout_id: str = "invalid-template", state: RolloutState = RolloutState.QUEUING
) -> Rollout:
    return Rollout(
        rollout_id=rollout_id,
        input={},
        config=RolloutConfig(k8s=RolloutK8sConfig(job_template=template)),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0, state=state),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template",
    INVALID_TEMPLATES,
    ids=["malformed-yaml", "malformed-jinja", "invalid-kind"],
)
@pytest.mark.parametrize("quota_full", [False, True])
async def test_invalid_local_template_fails_without_cluster_call(template: str, quota_full: bool) -> None:
    reconciler, api = _reconciler()
    get_k8s_api = AsyncMock()
    reconciler._get_k8s_api = get_k8s_api
    if quota_full:
        reconciler._job_creation_timestamps.extend([time.monotonic()] * 100)

    await reconciler._create_job(_rollout(template))

    get_k8s_api.assert_not_awaited()
    api.patch.assert_awaited_once()
    request = api.patch.await_args
    assert request.args == ("/api/rollouts/invalid-template",)
    assert request.kwargs["json"]["status"]["state"] == "failed"
    assert request.kwargs["json"]["status"]["error_message"].startswith("Invalid Job spec: ")
    assert len(reconciler._job_creation_timestamps) == (100 if quota_full else 0)


@pytest.mark.asyncio
async def test_transient_cluster_error_for_valid_template_is_retried() -> None:
    reconciler, api = _reconciler()
    get_k8s_api = AsyncMock(side_effect=RuntimeError("cluster temporarily unavailable"))
    reconciler._get_k8s_api = get_k8s_api

    await reconciler._create_job(_rollout(VALID_TEMPLATE))

    get_k8s_api.assert_awaited_once()
    api.patch.assert_not_awaited()
    assert not reconciler._job_creation_timestamps


def _query_response(api: AsyncMock, *rollouts: Rollout) -> None:
    api.get.return_value = httpx.Response(
        200,
        request=httpx.Request("GET", "http://store/api/rollouts"),
        json=[rollout.model_dump(mode="json") for rollout in rollouts],
    )


async def _job_listing(jobs: list[dict[str, Any]], error: Exception | None = None) -> AsyncIterator[Any]:
    if error is not None:
        raise error
    for job in jobs:
        yield SimpleNamespace(raw=job)


def _mock_job_type(monkeypatch: pytest.MonkeyPatch, *listings: AsyncIterator[Any]) -> Mock:
    job = Mock()
    job.async_create = AsyncMock()
    job_type = Mock(return_value=job)
    job_type.async_list = Mock(side_effect=listings)
    monkeypatch.setattr(k8s_reconciler.k8s_objects, "Job", job_type)
    return job_type


@pytest.mark.asyncio
@pytest.mark.parametrize("template", INVALID_TEMPLATES, ids=["malformed-yaml", "malformed-jinja", "invalid-kind"])
@pytest.mark.parametrize("gate", ["api-unavailable", "listing-unavailable", "quota-full"])
async def test_reconcile_rejects_invalid_templates_before_cluster_and_quota(
    monkeypatch: pytest.MonkeyPatch, template: str, gate: str
) -> None:
    reconciler, api = _reconciler()
    _query_response(api, _rollout(template))
    get_k8s_api = AsyncMock(return_value=object())
    if gate == "api-unavailable":
        get_k8s_api.side_effect = RuntimeError("cluster temporarily unavailable")
    reconciler._get_k8s_api = get_k8s_api
    listing_error = RuntimeError("list temporarily unavailable") if gate == "listing-unavailable" else None
    job_type = _mock_job_type(monkeypatch, _job_listing([], listing_error))
    if gate == "quota-full":
        reconciler._job_creation_timestamps.extend([time.monotonic()] * 100)

    await reconciler._reconcile_once()

    get_k8s_api.assert_not_awaited()
    job_type.async_list.assert_not_called()
    job_type.assert_not_called()
    api.patch.assert_awaited_once()
    assert api.patch.await_args.kwargs["json"]["status"]["state"] == "failed"
    assert api.patch.await_args.kwargs["json"]["status"]["error_message"].startswith("Invalid Job spec: ")
    assert len(reconciler._job_creation_timestamps) == (100 if gate == "quota-full" else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", ["api", "listing"])
async def test_reconcile_valid_job_retries_cluster_failure(monkeypatch: pytest.MonkeyPatch, gate: str) -> None:
    reconciler, api = _reconciler()
    _query_response(api, _rollout(VALID_TEMPLATE))
    get_k8s_api = AsyncMock(return_value=object())
    failure = RuntimeError("cluster temporarily unavailable")
    if gate == "api":
        get_k8s_api.side_effect = [failure, object(), object()]
        listings = [_job_listing([])]
    else:
        listings = [_job_listing([], failure), _job_listing([])]
    reconciler._get_k8s_api = get_k8s_api
    job_type = _mock_job_type(monkeypatch, *listings)

    with pytest.raises(RuntimeError, match="cluster temporarily unavailable"):
        await reconciler._reconcile_once()
    api.patch.assert_not_awaited()
    job_type.return_value.async_create.assert_not_awaited()
    assert not reconciler._job_creation_timestamps

    await reconciler._reconcile_once()
    job_type.return_value.async_create.assert_awaited_once()
    api.patch.assert_not_awaited()
    assert len(reconciler._job_creation_timestamps) == 1


@pytest.mark.asyncio
async def test_reconcile_valid_job_waits_for_quota_and_reuses_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciler, api = _reconciler()
    _query_response(api, _rollout(VALID_TEMPLATE))
    reconciler._get_k8s_api = AsyncMock(return_value=object())
    job_type = _mock_job_type(monkeypatch, _job_listing([]), _job_listing([]))
    build = Mock(wraps=k8s_reconciler.build_job_spec)
    monkeypatch.setattr(k8s_reconciler, "build_job_spec", build)
    reconciler._job_creation_timestamps.extend([time.monotonic()] * 100)

    await reconciler._reconcile_once()
    job_type.return_value.async_create.assert_not_awaited()
    api.patch.assert_not_awaited()
    assert len(reconciler._job_creation_timestamps) == 100

    # Let the same occupied slots age out of the real rate-limit window.
    reconciler._job_creation_timestamps.clear()
    reconciler._job_creation_timestamps.extend(
        [time.monotonic() - k8s_reconciler.JOB_CREATION_WINDOW_SECONDS - 1] * 100
    )
    await reconciler._reconcile_once()
    job_type.return_value.async_create.assert_awaited_once()
    api.patch.assert_not_awaited()
    assert len(reconciler._job_creation_timestamps) == 1
    assert build.call_count == 2  # One render per cycle, reused for submission.


@pytest.mark.asyncio
async def test_reconcile_keeps_observing_running_jobs_without_rendering_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler, api = _reconciler()
    _query_response(api, _rollout(INVALID_TEMPLATES[0], state=RolloutState.RUNNING))
    reconciler._get_k8s_api = AsyncMock(return_value=object())
    job_type = _mock_job_type(
        monkeypatch,
        _job_listing([{"metadata": {"name": "agl-rollout-invalid-template"}, "status": {"succeeded": 1}}]),
    )

    await reconciler._reconcile_once()

    job_type.assert_not_called()
    api.patch.assert_awaited_once()
    assert api.patch.await_args.kwargs["json"]["status"]["state"] == "succeeded"


@pytest.mark.asyncio
async def test_reconcile_retries_failed_invalid_template_status_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    reconciler, api = _reconciler()
    _query_response(api, _rollout(INVALID_TEMPLATES[0]))
    api.patch.return_value = httpx.Response(503, request=httpx.Request("PATCH", "http://store"))
    get_k8s_api = AsyncMock(side_effect=RuntimeError("cluster temporarily unavailable"))
    reconciler._get_k8s_api = get_k8s_api
    job_type = _mock_job_type(monkeypatch)

    for _ in range(2):
        await reconciler._reconcile_once()

    assert api.patch.await_count == 2
    assert all(call.kwargs["json"]["status"]["state"] == "failed" for call in api.patch.await_args_list)
    get_k8s_api.assert_not_awaited()
    job_type.async_list.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_valid_first_does_not_hide_invalid_template_during_cluster_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler, api = _reconciler()
    _query_response(api, _rollout(VALID_TEMPLATE, rollout_id="valid"), _rollout(INVALID_TEMPLATES[0]))

    async def unavailable() -> None:
        # Every queued rollout must be validated before the first cluster call.
        api.patch.assert_awaited_once()
        assert api.patch.await_args.args == ("/api/rollouts/invalid-template",)
        raise RuntimeError("cluster temporarily unavailable")

    reconciler._get_k8s_api = AsyncMock(side_effect=unavailable)
    job_type = _mock_job_type(monkeypatch)
    with pytest.raises(RuntimeError, match="cluster temporarily unavailable"):
        await reconciler._reconcile_once()

    api.patch.assert_awaited_once()
    assert api.patch.await_args.kwargs["json"]["status"]["state"] == "failed"
    job_type.assert_not_called()
    assert not reconciler._job_creation_timestamps


@pytest.mark.asyncio
async def test_reconcile_invalid_first_keeps_creating_valid_jobs_and_observing_running_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reconciler, api = _reconciler()
    _query_response(
        api,
        _rollout(INVALID_TEMPLATES[0]),
        _rollout(VALID_TEMPLATE, rollout_id="valid"),
        _rollout(INVALID_TEMPLATES[0], rollout_id="running", state=RolloutState.RUNNING),
    )
    reconciler._get_k8s_api = AsyncMock(return_value=object())
    job_type = _mock_job_type(
        monkeypatch,
        _job_listing([{"metadata": {"name": "agl-rollout-running"}, "status": {"succeeded": 1}}]),
    )
    build = Mock(wraps=k8s_reconciler.build_job_spec)
    monkeypatch.setattr(k8s_reconciler, "build_job_spec", build)

    await reconciler._reconcile_once()

    job_type.return_value.async_create.assert_awaited_once()
    assert job_type.call_args.args[0]["metadata"]["name"] == "agl-rollout-valid"
    assert [(call.args[0], call.kwargs["json"]["status"]["state"]) for call in api.patch.await_args_list] == [
        ("/api/rollouts/invalid-template", "failed"),
        ("/api/rollouts/running", "succeeded"),
    ]
    assert [call.args[0].rollout_id for call in build.call_args_list] == ["invalid-template", "valid"]
    assert len(reconciler._job_creation_timestamps) == 1
