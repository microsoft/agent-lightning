"""Tests for job_builder — pure function, no K8s needed."""

from __future__ import annotations

import pytest

from agl_lite.controller.config import ControllerSettings
from agl_lite.controller.job_builder import (
    PodPatcher,
    _deep_merge,
    _merge_by_name,
    _merge_env,
    build_job_name,
    build_job_spec,
    load_manifest_template,
)
from agl_lite.schemas.rollout import Rollout, RolloutConfig, RolloutStatus


def _make_rollout(
    rollout_id: str = "r1",
    image: str = "agent:v1",
    input: dict | None = None,
    config_overrides: dict | None = None,
) -> Rollout:
    config_kwargs: dict = {"image": image}
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


def _simple_template() -> dict:
    """Minimal job template — single agent container."""
    return {
        "spec": {
            "serviceAccountName": "default",
            "containers": [
                {
                    "name": "agent",
                    "imagePullPolicy": "Never",
                    "resources": {"requests": {"cpu": "100m", "memory": "128Mi"}},
                }
            ],
        }
    }


def _multi_container_template() -> dict:
    """Job template with agent + scorer sidecar."""
    return {
        "spec": {
            "containers": [
                {"name": "agent", "imagePullPolicy": "Never"},
                {"name": "scorer", "image": "scorer:latest", "command": ["python", "score.py"]},
            ],
            "volumes": [{"name": "workspace", "emptyDir": {}}],
        }
    }


@pytest.fixture(scope="module")
def manifest_template() -> str:
    """Load the default packaged Jinja2 job manifest template."""
    return load_manifest_template()


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


class TestBuildJobSpecBasic:
    def test_minimal_no_template(self, settings, manifest_template):
        """No job_template — agent container created from scratch."""
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings, manifest_template)

        assert job["apiVersion"] == "batch/v1"
        assert job["kind"] == "Job"
        assert job["metadata"]["name"] == "agl-rollout-r1"
        assert job["metadata"]["namespace"] == "test-ns"

        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["name"] == "agent"
        assert container["image"] == "agent:v1"

    def test_with_template(self, settings, manifest_template):
        """job_template provides pod spec, rollout config fills agent container."""
        rollout = _make_rollout()
        job = build_job_spec(rollout, _simple_template(), settings, manifest_template)

        pod_spec = job["spec"]["template"]["spec"]
        assert pod_spec["serviceAccountName"] == "default"

        container = pod_spec["containers"][0]
        assert container["name"] == "agent"
        assert container["image"] == "agent:v1"
        assert container["imagePullPolicy"] == "Never"
        assert container["resources"]["requests"]["cpu"] == "100m"

    def test_restart_policy_always_never(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"

    def test_template_restart_policy_preserved(self, settings, manifest_template):
        """restartPolicy: Never is set in the manifest template and preserved."""
        rollout = _make_rollout()
        template = {"spec": {"restartPolicy": "Never", "containers": [{"name": "agent"}]}}
        job = build_job_spec(rollout, template, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["restartPolicy"] == "Never"


class TestEnvVarInjection:
    def test_controller_env_vars(self, settings, manifest_template):
        rollout = _make_rollout(input={"task": "code"})
        job = build_job_spec(rollout, None, settings, manifest_template)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}

        assert env_map["AGL_POD_UID"]["valueFrom"]["fieldRef"]["fieldPath"] == "metadata.uid"
        for name in ("AGL_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            ref = env_map[name]["valueFrom"]["secretKeyRef"]
            assert ref["name"] == "agl-secrets"
            assert ref["key"] == "AGL_KEY"

        assert "$(AGL_POD_UID)" in env_map["OPENAI_BASE_URL"]["value"]
        assert env_map["OPENAI_BASE_URL"]["value"].endswith("/v1")
        assert "/events" in env_map["AGL_EVENT_URL"]["value"]

    def test_controller_env_in_all_containers(self, settings, manifest_template):
        """PodPatcher env vars are injected into every container, including sidecars."""
        rollout = _make_rollout()
        job = build_job_spec(rollout, _multi_container_template(), settings, manifest_template)

        for container in job["spec"]["template"]["spec"]["containers"]:
            env_map = {e["name"]: e for e in container["env"]}
            assert "AGL_ROLLOUT_ID" in env_map
            assert "OPENAI_BASE_URL" in env_map

    def test_user_env_wins_on_conflict(self, settings, manifest_template):
        """Container's own env var beats the patcher env on name conflict."""
        custom_url = "http://my-proxy/v1"
        template = {
            "spec": {
                "containers": [
                    {
                        "name": "agent",
                        "env": [{"name": "OPENAI_BASE_URL", "value": custom_url}],
                    }
                ]
            }
        }
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}
        assert env_map["OPENAI_BASE_URL"]["value"] == custom_url

    def test_user_env_vars(self, settings, manifest_template):
        rollout = _make_rollout(config_overrides={"environment_variables": {"MY_VAR": "hello"}})
        job = build_job_spec(rollout, None, settings, manifest_template)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}
        assert env_map["MY_VAR"]["value"] == "hello"

    def test_rollout_env_overrides_patcher(self, settings, manifest_template):
        """rollout.config.environment_variables override patcher env on the agent container."""
        rollout = _make_rollout(
            config_overrides={"environment_variables": {"OPENAI_BASE_URL": "http://override/v1"}}
        )
        job = build_job_spec(rollout, None, settings, manifest_template)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}
        assert env_map["OPENAI_BASE_URL"]["value"] == "http://override/v1"

    def test_template_env_vars_preserved(self, settings, manifest_template):
        """Container's own env vars (not conflicting) are kept alongside patcher env."""
        template = {
            "spec": {
                "containers": [{"name": "agent", "env": [{"name": "FROM_TEMPLATE", "value": "yes"}]}]
            }
        }
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)

        container = job["spec"]["template"]["spec"]["containers"][0]
        env_map = {e["name"]: e for e in container["env"]}
        assert "FROM_TEMPLATE" in env_map
        assert "AGL_EVENT_URL" in env_map


class TestCommandAndImage:
    def test_command(self, settings, manifest_template):
        rollout = _make_rollout(config_overrides={"command": ["python", "run.py"]})
        job = build_job_spec(rollout, None, settings, manifest_template)
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["command"] == ["python", "run.py"]

    def test_no_command_omitted(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings, manifest_template)
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert "command" not in container

    def test_image_overrides_template(self, settings, manifest_template):
        """RolloutConfig.image always overrides what's in the job_template."""
        template = {"spec": {"containers": [{"name": "agent", "image": "old:v1"}]}}
        rollout = _make_rollout(image="new:v2")
        job = build_job_spec(rollout, template, settings, manifest_template)
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "new:v2"


class TestJobSpecFields:
    def test_timeout(self, settings, manifest_template):
        rollout = _make_rollout(config_overrides={"timeout": 300})
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert job["spec"]["activeDeadlineSeconds"] == 300

    def test_no_timeout(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert "activeDeadlineSeconds" not in job["spec"]

    def test_timeout_from_job_template(self, settings, manifest_template):
        """activeDeadlineSeconds at root of job_template is hoisted to job spec."""
        # Need it at the pod fragment root, not inside spec.
        raw_template = {"containers": [{"name": "agent"}], "activeDeadlineSeconds": 5400}
        rollout = _make_rollout()
        job = build_job_spec(rollout, raw_template, settings, manifest_template)
        assert job["spec"]["activeDeadlineSeconds"] == 5400

    def test_rollout_timeout_beats_job_template(self, settings, manifest_template):
        """rollout.config.timeout takes precedence over job_template activeDeadlineSeconds."""
        raw_template = {"containers": [{"name": "agent"}], "activeDeadlineSeconds": 5400}
        rollout = _make_rollout(config_overrides={"timeout": 300})
        job = build_job_spec(rollout, raw_template, settings, manifest_template)
        assert job["spec"]["activeDeadlineSeconds"] == 300

    def test_max_retries(self, settings, manifest_template):
        rollout = _make_rollout(config_overrides={"max_retries": 3})
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert job["spec"]["backoffLimit"] == 3

    def test_default_backoff_zero(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert job["spec"]["backoffLimit"] == 0

    def test_ttl(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert job["spec"]["ttlSecondsAfterFinished"] == 3600

    def test_labels(self, settings, manifest_template):
        rollout = _make_rollout(rollout_id="r42")
        job = build_job_spec(rollout, None, settings, manifest_template)
        assert job["metadata"]["labels"]["app.kubernetes.io/managed-by"] == "agl-lite"
        assert job["metadata"]["labels"]["agl-lite/rollout-id"] == "r42"
        assert job["spec"]["template"]["metadata"]["labels"]["agl-lite/rollout-id"] == "r42"


class TestTemplatePassthrough:
    def test_node_selector(self, settings, manifest_template):
        template = {"spec": {"nodeSelector": {"gpu": "a100"}, "containers": [{"name": "agent"}]}}
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["nodeSelector"] == {"gpu": "a100"}

    def test_tolerations(self, settings, manifest_template):
        tol = {"key": "gpu", "operator": "Exists", "effect": "NoSchedule"}
        template = {"spec": {"tolerations": [tol], "containers": [{"name": "agent"}]}}
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["tolerations"] == [tol]

    def test_service_account(self, settings, manifest_template):
        template = {"spec": {"serviceAccountName": "agent-sa", "containers": [{"name": "agent"}]}}
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["serviceAccountName"] == "agent-sa"

    def test_image_pull_secrets(self, settings, manifest_template):
        template = {
            "spec": {"imagePullSecrets": [{"name": "my-reg"}], "containers": [{"name": "agent"}]}
        }
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["imagePullSecrets"] == [{"name": "my-reg"}]

    def test_arbitrary_k8s_fields(self, settings, manifest_template):
        template = {
            "spec": {"dnsPolicy": "ClusterFirst", "hostNetwork": True, "containers": [{"name": "agent"}]}
        }
        rollout = _make_rollout()
        job = build_job_spec(rollout, template, settings, manifest_template)
        assert job["spec"]["template"]["spec"]["dnsPolicy"] == "ClusterFirst"
        assert job["spec"]["template"]["spec"]["hostNetwork"] is True


class TestMultiContainer:
    def test_other_containers_preserved(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, _multi_container_template(), settings, manifest_template)

        containers = job["spec"]["template"]["spec"]["containers"]
        names = [c["name"] for c in containers]
        assert "agent" in names
        assert "scorer" in names

        scorer = next(c for c in containers if c["name"] == "scorer")
        assert scorer["image"] == "scorer:latest"
        assert scorer["command"] == ["python", "score.py"]

    def test_volumes_preserved(self, settings, manifest_template):
        rollout = _make_rollout()
        job = build_job_spec(rollout, _multi_container_template(), settings, manifest_template)
        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert any(v["name"] == "workspace" for v in volumes)


class TestOverrides:
    def test_override_other_container_image(self, settings, manifest_template):
        """Per-rollout override of scorer container image (SWE-bench case)."""
        rollout = _make_rollout(
            config_overrides={
                "overrides": {"containers": [{"name": "scorer", "image": "repo-123-test"}]}
            }
        )
        job = build_job_spec(rollout, _multi_container_template(), settings, manifest_template)

        scorer = next(
            c for c in job["spec"]["template"]["spec"]["containers"] if c["name"] == "scorer"
        )
        assert scorer["image"] == "repo-123-test"
        # Original command preserved.
        assert scorer["command"] == ["python", "score.py"]

    def test_override_pod_level_field(self, settings, manifest_template):
        """Override adds a pod-level field."""
        rollout = _make_rollout(config_overrides={"overrides": {"dnsPolicy": "Default"}})
        job = build_job_spec(rollout, _simple_template(), settings, manifest_template)
        assert job["spec"]["template"]["spec"]["dnsPolicy"] == "Default"

    def test_override_unknown_container_ignored(self, settings, manifest_template):
        """Override for a container not in the template is silently ignored."""
        rollout = _make_rollout(
            config_overrides={
                "overrides": {"containers": [{"name": "nonexistent", "image": "foo"}]}
            }
        )
        job = build_job_spec(rollout, _simple_template(), settings, manifest_template)
        names = [c["name"] for c in job["spec"]["template"]["spec"]["containers"]]
        assert "nonexistent" not in names

    def test_template_not_mutated(self, settings, manifest_template):
        """Ensure the original job_template dict is not modified."""
        template = _simple_template()
        original_image_pull = template["spec"]["containers"][0].get("imagePullPolicy")

        rollout = _make_rollout(
            config_overrides={
                "overrides": {"containers": [{"name": "agent", "resources": {"limits": {"gpu": "1"}}}]}
            }
        )
        build_job_spec(rollout, template, settings, manifest_template)

        assert template["spec"]["containers"][0].get("imagePullPolicy") == original_image_pull
        assert "limits" not in template["spec"]["containers"][0].get("resources", {})


class TestMounts:
    def test_host_path(self, settings, manifest_template):
        rollout = _make_rollout(
            config_overrides={"mount": [{"name": "data", "mount_path": "/data", "source": "/host/data"}]}
        )
        job = build_job_spec(rollout, None, settings, manifest_template)
        container = job["spec"]["template"]["spec"]["containers"][0]
        assert container["volumeMounts"][0]["mountPath"] == "/data"
        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert volumes[0]["hostPath"]["path"] == "/host/data"

    def test_pvc(self, settings, manifest_template):
        rollout = _make_rollout(
            config_overrides={"mount": [{"name": "ws", "mount_path": "/work", "source": "pvc:my-pvc"}]}
        )
        job = build_job_spec(rollout, None, settings, manifest_template)
        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert volumes[0]["persistentVolumeClaim"]["claimName"] == "my-pvc"

    def test_configmap(self, settings, manifest_template):
        rollout = _make_rollout(
            config_overrides={
                "mount": [{"name": "cfg", "mount_path": "/etc/config", "source": "my-cm"}]
            }
        )
        job = build_job_spec(rollout, None, settings, manifest_template)
        volumes = job["spec"]["template"]["spec"]["volumes"]
        assert volumes[0]["configMap"]["name"] == "my-cm"


class TestPodPatcher:
    def test_parsed_from_template(self, manifest_template):
        """PodPatcher is correctly parsed from the second document in the default template."""
        import yaml
        from jinja2 import Template

        rendered = Template(manifest_template).render(
            job_name="test-job",
            rollout_id="r1",
            namespace="default",
            secret_name="agl-secrets",
            lite_url="http://agl-lite:8000",
            ttl_after_finished=3600,
        )
        docs = list(yaml.safe_load_all(rendered))
        patcher = PodPatcher.model_validate(docs[1])
        names = [e["name"] for e in patcher.env]
        assert "AGL_POD_UID" in names
        assert "OPENAI_BASE_URL" in names
        assert "AGL_ROLLOUT_ID" in names
        assert patcher.volumes == []

    def test_custom_patcher_volumes(self, settings):
        """Custom manifest template with patcher volumes injects them into pod spec."""
        custom_template = """\
apiVersion: batch/v1
kind: Job
metadata:
  name: {{ job_name }}
  namespace: {{ namespace }}
  labels:
    app.kubernetes.io/managed-by: agl-lite
    agl-lite/rollout-id: {{ rollout_id }}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: {{ ttl_after_finished }}
  template:
    metadata:
      labels:
        app.kubernetes.io/managed-by: agl-lite
        agl-lite/rollout-id: {{ rollout_id }}
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
        user_template = {
            "containers": [{"name": "agent"}],
            "volumes": [{"name": "data", "emptyDir": {}}],
        }
        rollout = _make_rollout()
        job = build_job_spec(rollout, user_template, settings, custom_template)
        vol_names = [v["name"] for v in job["spec"]["template"]["spec"]["volumes"]]
        assert "shared-cache" in vol_names
        assert "data" in vol_names

    def test_user_volume_wins_over_patcher(self, settings):
        """User's volume beats patcher volume on name conflict."""
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
        user_template = {
            "containers": [{"name": "agent"}],
            "volumes": [{"name": "cache", "hostPath": {"path": "/data/cache"}}],
        }
        rollout = _make_rollout()
        job = build_job_spec(rollout, user_template, settings, custom_template)
        vols = {v["name"]: v for v in job["spec"]["template"]["spec"]["volumes"]}
        assert "hostPath" in vols["cache"]  # user's hostPath version wins


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

    def test_deep_merge_override_dict_with_non_dict(self):
        base = {"a": {"x": 1}}
        _deep_merge(base, {"a": "replaced"})
        assert base == {"a": "replaced"}
