"""Tests for job_builder — pure function, no K8s needed."""

from __future__ import annotations

import json

import pytest

from agl_lite.controller.config import ControllerSettings
from agl_lite.controller.job_builder import _deep_merge, build_job_name, build_job_spec
from agl_lite.schemas.resources import JobDefaults, K8sResources
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutStatus


def _make_rollout(
    rollout_id: str = "r1",
    image: str = "agent:v1",
    input: dict | None = None,
    config_overrides: dict | None = None,
) -> Rollout:
    """Create a minimal rollout for testing."""
    config_kwargs = {"image": image}
    if config_overrides:
        config_kwargs.update(config_overrides)
    return Rollout(
        rollout_id=rollout_id,
        status=RolloutStatus.QUEUING,
        input=input or {"prompt": "hello"},
        config=RolloutConfig(**config_kwargs),
        created_at=1000.0,
        updated_at=1000.0,
    )


@pytest.fixture
def settings() -> ControllerSettings:
    return ControllerSettings(
        lite_url="http://agl-lite:8000",
        key="test-key",
        namespace="test-ns",
        secret_name="agl-secrets",
    )


class TestBuildJobName:
    def test_deterministic(self):
        assert build_job_name("abc-123") == "agl-rollout-abc-123"

    def test_idempotent(self):
        assert build_job_name("r1") == build_job_name("r1")


class TestBuildJobSpec:
    def test_minimal(self, settings: ControllerSettings):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings)

        assert job["apiVersion"] == "batch/v1"
        assert job["kind"] == "Job"
        assert job["metadata"]["name"] == "agl-rollout-r1"
        assert job["metadata"]["namespace"] == "test-ns"

        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["name"] == "agent"
        assert container["image"] == "agent:v1"

    def test_env_vars_injected(self, settings: ControllerSettings):
        rollout = _make_rollout(input={"task": "code"})
        job = build_job_spec(rollout, None, settings)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}

        # Pod UID via Downward API.
        assert env_map["AGL_POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"

        # OpenAI SDK.
        assert "$(AGL_POD_UID)" in env_map["OPENAI_BASE_URL"]["value"]
        assert env_map["OPENAI_BASE_URL"]["value"].endswith("/v1")
        assert env_map["OPENAI_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "agl-secrets"

        # Anthropic SDK.
        assert "$(AGL_POD_UID)" in env_map["ANTHROPIC_BASE_URL"]["value"]
        assert env_map["ANTHROPIC_API_KEY"]["valueFrom"]["secretKeyRef"]["name"] == "agl-secrets"

        # Task input.
        assert json.loads(env_map["AGL_TASK_INPUT"]["value"]) == {"task": "code"}

        # Event URL.
        assert "/events" in env_map["AGL_EVENT_URL"]["value"]

    def test_user_env_vars_appended(self, settings: ControllerSettings):
        rollout = _make_rollout(config_overrides={"environment_variables": {"MY_VAR": "hello"}})
        job = build_job_spec(rollout, None, settings)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}
        assert env_map["MY_VAR"]["value"] == "hello"

    def test_command(self, settings: ControllerSettings):
        rollout = _make_rollout(config_overrides={"command": ["python", "run.py"]})
        job = build_job_spec(rollout, None, settings)

        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["command"] == ["python", "run.py"]

    def test_no_command_omitted(self, settings: ControllerSettings):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings)

        container = job["spec"]["template"]["spec"]["containers"][0]
        assert "command" not in container

    def test_timeout_from_config(self, settings: ControllerSettings):
        rollout = _make_rollout(config_overrides={"timeout": 300})
        job = build_job_spec(rollout, None, settings)

        assert job["spec"]["activeDeadlineSeconds"] == 300

    def test_timeout_from_defaults(self, settings: ControllerSettings):
        rollout = _make_rollout()
        defaults = JobDefaults(timeout=600)
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["activeDeadlineSeconds"] == 600

    def test_timeout_config_overrides_defaults(self, settings: ControllerSettings):
        rollout = _make_rollout(config_overrides={"timeout": 300})
        defaults = JobDefaults(timeout=600)
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["activeDeadlineSeconds"] == 300

    def test_no_timeout(self, settings: ControllerSettings):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings)

        assert "activeDeadlineSeconds" not in job["spec"]

    def test_max_retries_from_config(self, settings: ControllerSettings):
        rollout = _make_rollout(config_overrides={"max_retries": 3})
        job = build_job_spec(rollout, None, settings)

        assert job["spec"]["backoffLimit"] == 3

    def test_max_retries_from_defaults(self, settings: ControllerSettings):
        rollout = _make_rollout()
        defaults = JobDefaults(max_retries=5)
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["backoffLimit"] == 5

    def test_max_retries_config_overrides_defaults(self, settings: ControllerSettings):
        rollout = _make_rollout(config_overrides={"max_retries": 2})
        defaults = JobDefaults(max_retries=5)
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["backoffLimit"] == 2

    def test_default_backoff_limit_zero(self, settings: ControllerSettings):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings)

        assert job["spec"]["backoffLimit"] == 0

    def test_ttl_after_finished(self, settings: ControllerSettings):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings)

        assert job["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_restart_policy_never(self, settings: ControllerSettings):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings)

        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_labels(self, settings: ControllerSettings):
        rollout = _make_rollout(rollout_id="r42")
        job = build_job_spec(rollout, None, settings)

        # Job-level labels.
        assert job["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "agl-lite"
        assert job["metadata"]["labels"]["agl-lite/rollout-id"] == "r42"

        # Pod template labels.
        pod_labels = job["spec"]["template"]["metadata"]["labels"]
        assert pod_labels["agl-lite/rollout-id"] == "r42"

    def test_resources_from_defaults(self, settings: ControllerSettings):
        defaults = JobDefaults(
            resources=K8sResources(
                requests={"cpu": "500m", "memory": "1Gi"},
                limits={"cpu": "2", "memory": "4Gi"},
            )
        )
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["resources"]["requests"]["cpu"] == "500m"
        assert container["resources"]["limits"]["memory"] == "4Gi"

    def test_node_selector(self, settings: ControllerSettings):
        defaults = JobDefaults(node_selector={"gpu": "a100"})
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["template"]["spec"]["nodeSelector"] == {"gpu": "a100"}

    def test_tolerations(self, settings: ControllerSettings):
        toleration = {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"}
        defaults = JobDefaults(tolerations=[toleration])
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["template"]["spec"]["tolerations"] == [toleration]

    def test_service_account(self, settings: ControllerSettings):
        defaults = JobDefaults(service_account="agent-sa")
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["template"]["spec"]["serviceAccountName"] == "agent-sa"

    def test_image_pull_secrets(self, settings: ControllerSettings):
        defaults = JobDefaults(image_pull_secrets=["my-registry"])
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name": "my-registry"}]

    def test_mounts_host_path(self, settings: ControllerSettings):
        rollout = _make_rollout(
            config_overrides={
                "mount": [{"name": "data", "mount_path": "/data", "source": "/host/data"}],
            }
        )
        job = build_job_spec(rollout, None, settings)

        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["volumeMounts"][0]["name"] == "data"
        assert container["volumeMounts"][0]["mountPath"] == "/data"

        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert volumes[0]["hostPath"]["path"] == "/host/data"

    def test_mounts_pvc(self, settings: ControllerSettings):
        rollout = _make_rollout(
            config_overrides={
                "mount": [{"name": "workspace", "mount_path": "/work", "source": "pvc:my-pvc"}],
            }
        )
        job = build_job_spec(rollout, None, settings)

        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert volumes[0]["persistentVolumeClaim"]["claimName"] == "my-pvc"

    def test_mounts_configmap(self, settings: ControllerSettings):
        rollout = _make_rollout(
            config_overrides={
                "mount": [{"name": "cfg", "mount_path": "/etc/config", "source": "my-configmap"}],
            }
        )
        job = build_job_spec(rollout, None, settings)

        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert volumes[0]["configMap"]["name"] == "my-configmap"

    def test_overrides_escape_hatch(self, settings: ControllerSettings):
        defaults = JobDefaults(
            overrides={
                "template": {"metadata": {"annotations": {"iam.amazonaws.com/role": "my-role"}}},
            }
        )
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        annotations = job["spec"]["template"]["metadata"]["annotations"]
        assert annotations["iam.amazonaws.com/role"] == "my-role"
        # Labels should still be preserved (deep merge, not replace).
        assert "agl-lite/rollout-id" in job["spec"]["template"]["metadata"]["labels"]

    def test_overrides_add_new_fields(self, settings: ControllerSettings):
        defaults = JobDefaults(overrides={"completionMode": "Indexed"})
        rollout = _make_rollout()
        job = build_job_spec(rollout, defaults, settings)

        assert job["spec"]["completionMode"] == "Indexed"


class TestDeepMerge:
    def test_simple(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_nested(self):
        base = {"a": {"x": 1, "y": 2}}
        _deep_merge(base, {"a": {"y": 3, "z": 4}})
        assert base == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_override_dict_with_non_dict(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": "replaced"})
        assert base == {"a": "replaced"}
