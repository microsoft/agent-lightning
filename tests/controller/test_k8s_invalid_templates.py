# Copyright (c) Microsoft. All rights reserved.

"""Regression tests for local K8s Job template validation."""

from unittest.mock import AsyncMock

import httpx
import pytest
from omegaconf import OmegaConf

from agentlightning.client import AgentLightningAsyncClient
from agentlightning.controller.k8s_reconciler import K8sReconciler
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus


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


def _rollout(template: str) -> Rollout:
    return Rollout(
        rollout_id="invalid-template",
        input={},
        config=RolloutConfig(k8s=RolloutK8sConfig(job_template=template)),
        status=RolloutLifecycleStatus(created_at=1.0, updated_at=1.0),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template",
    [
        "kind: Job\nspec: [\n",
        "kind: Job\nmetadata:\n  name: {{ job_name\n",
        "apiVersion: v1\nkind: Pod\nmetadata: {}\n",
    ],
    ids=["malformed-yaml", "malformed-jinja", "invalid-kind"],
)
async def test_invalid_local_template_fails_without_cluster_call(template: str) -> None:
    reconciler, api = _reconciler()
    get_k8s_api = AsyncMock()
    reconciler._get_k8s_api = get_k8s_api

    await reconciler._create_job(_rollout(template))

    get_k8s_api.assert_not_awaited()
    api.patch.assert_awaited_once()
    request = api.patch.await_args
    assert request.args == ("/api/rollouts/invalid-template",)
    assert request.kwargs["json"]["status"]["state"] == "failed"
    assert request.kwargs["json"]["status"]["error_message"].startswith("Invalid Job spec: ")
    assert not reconciler._job_creation_timestamps


@pytest.mark.asyncio
async def test_transient_cluster_error_for_valid_template_is_retried() -> None:
    reconciler, api = _reconciler()
    get_k8s_api = AsyncMock(side_effect=RuntimeError("cluster temporarily unavailable"))
    reconciler._get_k8s_api = get_k8s_api

    await reconciler._create_job(_rollout("apiVersion: batch/v1\nkind: Job\nspec: {}\n"))

    get_k8s_api.assert_awaited_once()
    api.patch.assert_not_awaited()
    assert not reconciler._job_creation_timestamps
