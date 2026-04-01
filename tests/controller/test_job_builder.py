"""Tests for job_builder — pure function, no K8s needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from agl_lite.controller.config import ControllerSettings
from agl_lite.controller.job_builder import (
    PodPatcher,
    _deep_merge,
    _merge_by_name,
    _merge_env,
    build_job_name,
    build_job_spec,
)
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutStatus


def _make_rollout(
    rollout_id: str = "r1",
    pod_spec: dict | None = None,
    config_overrides: dict | None = None,
) -> Rollout:
    config_kwargs: dict = {}
    if pod_spec is not None:
        config_kwargs["pod_spec"] = pod_spec
    if config_overrides:
        config_kwargs.update(config_overrides)
    return Rollout(
        rollout_id=rollout_id,
        status=RolloutStatus.QUEUING,
        input={"prompt": "hello"},
        config=RolloutConfig(**config_kwargs),
        created_at=1000.0,
        updated_at=1000.0,
    )


def _agent_pod_spec() -> dict:
    """Minimal pod spec with a single agent container."""
    return {
        "containers": [
            {
                "name": "agent",
                "image": "agent:v1",
                "imagePullPolicy": "Never",
                "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}},
            }
        ],
        "serviceAccountName": "default",
    }


def _multi_container_pod_spec() -> dict:
    """Pod spec with agent + scorer sidecar."""
    return {
        "containers": [
            {"name": "agent", "image": "agent:v1", "imagePullPolicy": "Never"},
            {"name": "scorer", "image": "scorer:latest", "command": ["python", "score.py"]},
        ],
        "volumes": [{"name": "workspace", "emptyDir": {}}],
    }


@pytest.fixture(scope="module")
def manifest_template() -> str:
    return Path("deploy/controller/job-template.yaml.j2").read_text()


@pytest.fixture
def settings() -> ControllerSettings:
    return ControllerSettings(
        base_url="http://agl-lite:8000",
        key="test-key",
        namespace="test-ns",
        job_manifest_template="deploy/controller/job-template.yaml.j2",
    )


class TestBuildJobName:
    def test_deterministic(self):
        assert build_job_name("abc-123") == "agl-rollout-abc-123"

    def test_idempotent(self):
        assert build_job_name("r1") == build_job_name("r1")


class TestBuildJobSpecBasic:
    def test_no_pod_spec(self, settings, manifest_template):
        """No pod_spec — containers list is empty, Job scaffold is still valid."""
        rollout = _make_rollout()
        job = build_job_spec(rollout, settings, manifest_template)

        assert job["apiVersion"] == "batch/v1"
        assert job["kind"] == "Job"
        assert job["metadata"]["name"] == "agl-rollout-r1"
        assert job["metadata"]["namespace"] == "test-ns"
        assert job["spec"]["template"]["spec"]["containers"] == []

    def test_with_pod_spec(self, settings, manifest_template):
        """pod_spec provides containers and pod-level fields."""
        rollout = _make_rollout(pod_spec=_agent_pod_spec())
        job = build_job_spec(rollout, settings, manifest_template)

        pod_spec = job["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "default"

        container = pod_spec["containers"][0]
        assert container["name"] == "agent"
        assert container["image"] == "agent:v1"
        assert container["imagePullPolicy"] == "Never"

    def test_restart_policy_never(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_pod_spec_not_mutated(self, settings, manifest_template):
        """Original pod_spec dict is not modified."""
        original = _agent_pod_spec()
        import copy
        snapshot = copy.deepcopy(original)
        rollout = _make_rollout(pod_spec=original)
        build_job_spec(rollout, settings, manifest_template)
        assert original == snapshot


class TestPodSpecPassthrough:
    def test_node_selector(self, settings, manifest_template):
        ps = {"containers": [{"name": "agent", "image": "x"}], "nodeSelector": {"gpu": "a100"}}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["nodeSelector"] == {"gpu": "a100"}

    def test_tolerations(self, settings, manifest_template):
        tol = {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"}
        ps = {"containers": [{"name": "agent", "image": "x"}], "tolerations": [tol]}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["tolerations"] == [tol]

    def test_service_account(self, settings, manifest_template):
        ps = {"containers": [{"name": "agent"}], "serviceAccountName": "agent-sa"}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["serviceAccountName"] == "agent-sa"

    def test_volumes_preserved(self, settings, manifest_template):
        rollout = _make_rollout(pod_spec=_multi_container_pod_spec())
        job = build_job_spec(rollout, settings, manifest_template)
        vols = job["spec"]["template"]["spec"]["volumes"]
        assert any(v["name"] == "workspace" for v in vols)

    def test_multiple_containers(self, settings, manifest_template):
        rollout = _make_rollout(pod_spec=_multi_container_pod_spec())
        job = build_job_spec(rollout, settings, manifest_template)
        names = [c["name"] for c in job["spec"]["template"]["spec"]["containers"]]
        assert "agent" in names
        assert "scorer" in names

    def test_active_deadline_seconds_hoisted(self, settings, manifest_template):
        """activeDeadlineSeconds at pod_spec root is hoisted to job spec."""
        ps = {"containers": [{"name": "agent"}], "activeDeadlineSeconds": 5400}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["activeDeadlineSeconds"] == 5400
        # Should not appear in pod spec itself.
        assert "activeDeadlineSeconds" not in job["spec"]["template"]["spec"]


class TestEnvVarInjection:
    def test_controller_env_in_agent(self, settings, manifest_template):
        rollout = _make_rollout(pod_spec=_agent_pod_spec())
        job = build_job_spec(rollout, settings, manifest_template)
        env_map = {e["name"]: e for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}

        assert env_map["AGL_POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
        for name in ("AGL_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            ref = env_map[name]["valueFrom"]["secretKeyRef"]
            assert ref["name"] == "agl-lite-keys"
        assert env_map["OPENAI_BASE_URL"]["value"].endswith("/v1")
        assert "/events" in env_map["AGL_EVENT_URL"]["value"]

    def test_controller_env_in_all_containers(self, settings, manifest_template):
        """PodPatcher env vars injected into every container including sidecars."""
        rollout = _make_rollout(pod_spec=_multi_container_pod_spec())
        job = build_job_spec(rollout, settings, manifest_template)
        for container in job["spec"]["template"]["spec"]["containers"]:
            env_names = {e["name"] for e in container["env"]}
            assert "AGL_ROLLOUT_ID" in env_names
            assert "OPENAI_BASE_URL" in env_names

    def test_container_env_wins_on_conflict(self, settings, manifest_template):
        """Container's own env var beats patcher env on name conflict."""
        custom_url = "http://my-proxy/v1"
        ps = {
            "containers": [
                {"name": "agent", "image": "x", "env": [{"name": "OPENAI_BASE_URL", "value": custom_url}]}
            ]
        }
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, manifest_template)
        env_map = {e["name"]: e for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert env_map["OPENAI_BASE_URL"]["value"] == custom_url

    def test_container_env_preserved_alongside_patcher(self, settings, manifest_template):
        """Container's non-conflicting env vars are kept alongside patcher env."""
        ps = {"containers": [{"name": "agent", "image": "x", "env": [{"name": "MY_VAR", "value": "hi"}]}]}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, manifest_template)
        env_map = {e["name"]: e for e in job["spec"]["template"]["spec"]["containers"][0]["env"]}
        assert "MY_VAR" in env_map
        assert "AGL_EVENT_URL" in env_map


class TestJobSpecFields:
    def test_timeout(self, settings, manifest_template):
        rollout = _make_rollout(config_overrides={"timeout": 300})
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["activeDeadlineSeconds"] == 300

    def test_no_timeout(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, settings, manifest_template)
        assert "activeDeadlineSeconds" not in job["spec"]

    def test_config_timeout_beats_pod_spec(self, settings, manifest_template):
        """config.timeout takes precedence over pod_spec activeDeadlineSeconds."""
        ps = {"containers": [{"name": "agent"}], "activeDeadlineSeconds": 5400}
        rollout = _make_rollout(pod_spec=ps, config_overrides={"timeout": 300})
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["activeDeadlineSeconds"] == 300

    def test_max_retries(self, settings, manifest_template):
        rollout = _make_rollout(config_overrides={"max_retries": 3})
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["backoffLimit"] == 3

    def test_default_backoff_zero(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["backoffLimit"] == 0

    def test_ttl(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_labels(self, settings, manifest_template):
        rollout = _make_rollout(rollout_id="r42")
        job = build_job_spec(rollout, settings, manifest_template)
        assert job["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "agl-lite"
        assert job["metadata"]["labels"]["agl-lite/rollout-id"] == "r42"
        assert job["spec"]["template"]["metadata"]["labels"]["agl-lite/rollout-id"] == "r42"


class TestPodPatcher:
    def test_parsed_from_template(self, manifest_template):
        import yaml
        from jinja2 import Template

        rendered = Template(manifest_template).render(
            job_name="test-job", rollout_id="r1", namespace="default",
        )
        docs = list(yaml.safe_load_all(rendered))
        patcher = PodPatcher.model_validate(docs[1])
        names = [e["name"] for e in patcher.env]
        assert "AGL_POD_UID" in names
        assert "OPENAI_BASE_URL" in names
        assert patcher.volumes == []

    def test_custom_patcher_volumes(self, settings):
        """Custom template with patcher volumes injects them into pod spec."""
        custom_template = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
  labels: {}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: {{ ttl_after_finished }}
  template:
    metadata:
      labels: {}
    spec:
      restartPolicy: Never
      containers: []
      volumes: []
---
env: []
volumes:
  - name: shared-cache
    emptyDir: {}
"""
        ps = {"containers": [{"name": "agent"}], "volumes": [{"name": "data", "emptyDir": {}}]}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, custom_template)
        vol_names = [v["name"] for v in job["spec"]["template"]["spec"]["volumes"]]
        assert "shared-cache" in vol_names
        assert "data" in vol_names

    def test_user_volume_wins_over_patcher(self, settings):
        custom_template = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
  labels: {}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: {{ ttl_after_finished }}
  template:
    metadata:
      labels: {}
    spec:
      restartPolicy: Never
      containers: []
      volumes: []
---
env: []
volumes:
  - name: cache
    emptyDir: {}
"""
        ps = {"containers": [{"name": "agent"}], "volumes": [{"name": "cache", "hostPath": {"path": "/data"}}]}
        rollout = _make_rollout(pod_spec=ps)
        job = build_job_spec(rollout, settings, custom_template)
        vols = {v["name"]: v for v in job["spec"]["template"]["spec"]["volumes"]}
        assert "hostPath" in vols["cache"]


class TestMergeHelpers:
    def test_merge_env_override_wins(self):
        base = [{"name": "FOO", "value": "base"}, {"name": "BAR", "value": "bar"}]
        override = [{"name": "FOO", "value": "override"}, {"name": "BAZ", "value": "baz"}]
        result = _merge_env(base, override)
        by_name = {e["name"]: e["value"] for e in result}
        assert by_name["FOO"] == "override"
        assert by_name["BAR"] == "bar"
        assert by_name["BAZ"] == "baz"

    def test_merge_by_name_override_wins(self):
        base = [{"name": "a", "x": 1}, {"name": "b", "x": 2}]
        override = [{"name": "a", "x": 99}, {"name": "c", "x": 3}]
        result = _merge_by_name(base, override)
        by_name = {item["name"]: item["x"] for item in result}
        assert by_name["a"] == 99
        assert by_name["b"] == 2
        assert by_name["c"] == 3

    def test_deep_merge_simple(self):
        base = {"a": 1, "b": 2}
        _deep_merge(base, {"b": 3, "c": 4})
        assert base == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge_nested(self):
        base = {"a": {"x": 1, "y": 2}}
        _deep_merge(base, {"a": {"y": 3, "z": 4}})
        assert base == {"a": {"x": 1, "y": 3, "z": 4}}
