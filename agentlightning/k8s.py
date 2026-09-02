# Copyright (c) Microsoft. All rights reserved.

"""Pure helpers shared by Kubernetes rollout producers and consumers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import yaml
from jinja2 import Environment, Template

__all__ = [
    "extract_pod_images",
    "normalize_image_reference",
    "render_job_template",
]


def normalize_image_reference(image: str) -> str:
    """Return a canonical image reference suitable for exact comparison."""
    reference = image.strip()
    if not reference:
        raise ValueError("container image must be a non-empty string")
    if "://" in reference:
        reference = reference.split("://", 1)[1]

    if "/" not in reference:
        reference = f"docker.io/library/{reference}"
    else:
        first, remainder = reference.split("/", 1)
        if first in {"docker.io", "index.docker.io"}:
            if "/" not in remainder:
                remainder = f"library/{remainder}"
            reference = f"docker.io/{remainder}"
        elif "." not in first and ":" not in first and first != "localhost":
            reference = f"docker.io/{reference}"

    last_component = reference.rsplit("/", 1)[-1]
    if "@" not in reference and ":" not in last_component:
        reference = f"{reference}:latest"
    return reference


@lru_cache(maxsize=32)
def _compile_job_template(job_template: str) -> Template:
    environment = Environment()
    environment.filters["yaml_escape"] = lambda value: json.dumps(str(value), ensure_ascii=True)
    return environment.from_string(job_template)


def render_job_template(
    job_template: str,
    *,
    job_name: str,
    input_data: Any,
) -> dict[str, Any]:
    """Render one Kubernetes Job from the controller-compatible template."""
    rendered = _compile_job_template(job_template).render(
        job_name=job_name,
        input=input_data,
    )
    documents = [document for document in yaml.safe_load_all(rendered) if document is not None]
    if len(documents) != 1:
        raise ValueError("job template must render exactly one YAML document")
    job = documents[0]
    if not isinstance(job, dict) or job.get("kind") != "Job":
        raise ValueError("job template must render a Kubernetes Job")
    return job


def extract_pod_images(job: Mapping[str, Any]) -> frozenset[str]:
    """Extract normalized images from every container list in a Job Pod spec."""
    pod_spec = job.get("spec", {}).get("template", {}).get("spec", {})
    images: set[str] = set()
    for container_key in ("initContainers", "containers", "ephemeralContainers"):
        for container in pod_spec.get(container_key, []) or []:
            image = container.get("image")
            if not isinstance(image, str) or not image.strip():
                raise ValueError(f"{container_key} entry is missing a non-empty image")
            images.add(normalize_image_reference(image))
    if not images:
        raise ValueError("rendered Kubernetes Job contains no container images")
    return frozenset(images)
