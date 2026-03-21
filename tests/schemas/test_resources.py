"""Tests for resource schemas — snapshot semantics, job_template as opaque dict."""

from agl_lite.schemas.resources import ResourcesUpdate


class TestResourcesUpdate:
    def test_opaque_resources(self):
        r = ResourcesUpdate(
            resources_id="res-1",
            resources={"system_prompt": "You are helpful", "eval_config": {"metric": "pass@1"}},
            created_at=1000.0,
        )
        assert r.resources["system_prompt"] == "You are helpful"

    def test_job_template_is_opaque(self):
        """job_template is stored as-is — no validation at store level."""
        r = ResourcesUpdate(
            resources_id="res-2",
            resources={
                "job_template": {
                    "spec": {
                        "serviceAccountName": "default",
                        "containers": [
                            {"name": "agent", "imagePullPolicy": "Never", "resources": {"requests": {"cpu": "1"}}}
                        ],
                    }
                },
                "prompt": "hello",
            },
            created_at=1000.0,
        )
        assert r.resources["job_template"]["spec"]["serviceAccountName"] == "default"

    def test_job_template_any_k8s_fields(self):
        """Any valid K8s field can go in job_template — no schema restrictions."""
        r = ResourcesUpdate(
            resources_id="res-3",
            resources={
                "job_template": {
                    "spec": {
                        "nodeSelector": {"gpu": "a100"},
                        "tolerations": [{"key": "gpu", "operator": "Exists"}],
                        "dnsPolicy": "ClusterFirst",
                        "containers": [{"name": "agent"}, {"name": "scorer", "image": "scorer:latest"}],
                    }
                },
            },
            created_at=1000.0,
        )
        template = r.resources["job_template"]
        assert template["spec"]["nodeSelector"]["gpu"] == "a100"
        assert len(template["spec"]["containers"]) == 2

    def test_no_job_template_is_fine(self):
        r = ResourcesUpdate(
            resources_id="res-4",
            resources={"anything": {"nested": {"deeply": True}}},
            created_at=1000.0,
        )
        assert r.resources["anything"]["nested"]["deeply"] is True
