# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for controller manifests; no cluster or GPU is required."""

from omegaconf import OmegaConf

from agentlightning.controller.k8s_reconciler import MANAGED_BY_SELECTOR, build_job_spec
from agentlightning.schemas import Rollout, RolloutConfig, RolloutK8sConfig, RolloutLifecycleStatus


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
