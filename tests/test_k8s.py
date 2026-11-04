# Copyright (c) Microsoft. All rights reserved.

"""Tests for Kubernetes Job rendering and image matching helpers."""

import pytest

from agentlightning.k8s import extract_pod_images, normalize_image_reference, render_job_template


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("ubuntu", "docker.io/library/ubuntu:latest"),
        ("ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
        ("docker.io/ubuntu:24.04", "docker.io/library/ubuntu:24.04"),
        ("swebench/repo:openai", "docker.io/swebench/repo:openai"),
        ("index.docker.io/swebench/repo:openai", "docker.io/swebench/repo:openai"),
        ("docker.io/swebench/repo@sha256:abc", "docker.io/swebench/repo@sha256:abc"),
        ("localhost:5000/repo:v1", "localhost:5000/repo:v1"),
    ],
)
def test_normalize_image_reference_uses_canonical_docker_names(reference: str, expected: str) -> None:
    assert normalize_image_reference(reference) == expected


def test_normalize_image_reference_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_image_reference("  ")


def test_render_and_extract_uses_every_pod_container_type() -> None:
    template = """
apiVersion: batch/v1
kind: Job
metadata: {}
spec:
  template:
    spec:
      initContainers:
        - name: setup
          image: busybox:1.36
      containers:
        - name: agent
          image: {{ (input.image_name ~ ":openai") | yaml_escape }}
      ephemeralContainers:
        - name: debugger
          image: alpine
"""

    job = render_job_template(
        template,
        job_name="agl-image-check",
        input_data={"image_name": "swebench/repo"},
    )

    assert extract_pod_images(job) == frozenset(
        {
            "docker.io/library/busybox:1.36",
            "docker.io/library/alpine:latest",
            "docker.io/swebench/repo:openai",
        }
    )


@pytest.mark.parametrize(
    "job",
    [
        {"kind": "Job", "spec": {"template": {"spec": {"containers": []}}}},
        {"kind": "Job", "spec": {"template": {"spec": {"containers": [{"name": "agent"}]}}}},
    ],
)
def test_extract_pod_images_rejects_missing_images(job: dict) -> None:
    with pytest.raises(ValueError, match="image"):
        extract_pod_images(job)


@pytest.mark.parametrize(
    "template",
    [
        "apiVersion: v1\nkind: Pod\nmetadata: {}\n",
        "---\nkind: Job\n---\nkind: Job\n",
    ],
)
def test_render_job_template_requires_exactly_one_job(template: str) -> None:
    with pytest.raises(ValueError, match=r"Job|document"):
        render_job_template(template, job_name="test", input_data={})
