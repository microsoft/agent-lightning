"""Tests for resource schemas — JobDefaults validation, snapshot semantics."""

from agl_lite.schemas.resources import JobDefaults, K8sResources, ResourcesUpdate


class TestJobDefaults:
    def test_minimal(self):
        jd = JobDefaults()
        assert jd.resources is None
        assert jd.node_selector == {}
        assert jd.tolerations == []
        assert jd.service_account is None
        assert jd.image_pull_secrets == []
        assert jd.timeout is None
        assert jd.max_retries is None
        assert jd.overrides == {}

    def test_full(self):
        jd = JobDefaults(
            resources=K8sResources(
                requests={"cpu": "500m", "memory": "1Gi"},
                limits={"cpu": "2", "memory": "4Gi"},
            ),
            node_selector={"gpu": "a100"},
            tolerations=[{"key": "gpu", "operator": "Exists", "effect": "NoSchedule"}],
            service_account="agl-agent",
            image_pull_secrets=["registry-creds"],
            timeout=300,
            max_retries=2,
        )
        assert jd.resources.requests["cpu"] == "500m"
        assert jd.node_selector["gpu"] == "a100"
        assert jd.timeout == 300

    def test_overrides_escape_hatch(self):
        """Unknown K8s fields go into overrides, not rejected."""
        jd = JobDefaults(
            timeout=600,
            overrides={
                "dnsPolicy": "ClusterFirst",
                "labels": {"team": "ml-infra"},
                "annotations": {"iam.amazonaws.com/role": "arn:aws:iam::role/agent"},
            },
        )
        assert jd.overrides["dnsPolicy"] == "ClusterFirst"
        assert jd.overrides["labels"]["team"] == "ml-infra"
        assert jd.timeout == 600


class TestResourcesUpdate:
    def test_opaque_resources(self):
        r = ResourcesUpdate(
            resources_id="res-1",
            resources={"system_prompt": "You are helpful", "eval_config": {"metric": "pass@1"}},
            created_at=1000.0,
        )
        assert r.resources["system_prompt"] == "You are helpful"

    def test_validates_job_defaults_when_present(self):
        r = ResourcesUpdate(
            resources_id="res-2",
            resources={
                "job_defaults": {"resources": {"requests": {"cpu": "1"}}, "timeout": 600},
                "prompt": "hello",
            },
            created_at=1000.0,
        )
        assert r.resources["job_defaults"]["timeout"] == 600

    def test_validates_job_defaults_with_overrides(self):
        r = ResourcesUpdate(
            resources_id="res-3",
            resources={
                "job_defaults": {
                    "timeout": 300,
                    "overrides": {"dnsPolicy": "ClusterFirst"},
                },
            },
            created_at=1000.0,
        )
        assert r.resources["job_defaults"]["overrides"]["dnsPolicy"] == "ClusterFirst"

    def test_no_validation_without_job_defaults_key(self):
        r = ResourcesUpdate(
            resources_id="res-4",
            resources={"anything": {"nested": {"deeply": True}}},
            created_at=1000.0,
        )
        assert r.resources["anything"]["nested"]["deeply"] is True
