"""Resource schemas — versioned immutable snapshots."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class K8sResources(BaseModel):
    """K8s container resource requests/limits."""

    requests: dict[str, str] = Field(default_factory=dict)  # e.g., {"cpu": "500m", "memory": "1Gi"}
    limits: dict[str, str] = Field(default_factory=dict)  # e.g., {"cpu": "2", "memory": "4Gi"}


class JobDefaults(BaseModel):
    """Infra-level Job spec defaults.

    Validated when 'job_defaults' key is present in a resource snapshot.
    Known fields are typed and validated. Unknown/future K8s fields go
    into `overrides` and are merged raw into the Job spec by the controller.
    """

    resources: K8sResources | None = None
    node_selector: dict[str, str] = Field(default_factory=dict)
    tolerations: list[dict[str, Any]] = Field(default_factory=list)
    service_account: str | None = None
    image_pull_secrets: list[str] = Field(default_factory=list)
    timeout: int | None = None  # default timeout (seconds)
    max_retries: int | None = None  # default retry count

    # Escape hatch — raw dict merged into K8s Job spec by controller.
    # For fields not explicitly modeled above (labels, annotations, DNS policy, etc.).
    overrides: dict[str, Any] = Field(default_factory=dict)


class ResourcesUpdate(BaseModel):
    """Immutable resource snapshot. One bundle containing all named resources.

    The 'job_defaults' key is reserved — if present, its value is validated
    as a JobDefaults model. All other keys are user-defined and opaque.
    """

    resources_id: str
    resources: dict[str, Any]  # {"job_defaults": {...}, "system_prompt": "...", ...}
    created_at: float
    version: int = 1

    @model_validator(mode="after")
    def validate_job_defaults(self) -> ResourcesUpdate:
        """If 'job_defaults' key exists, validate it as a typed schema."""
        if "job_defaults" in self.resources:
            JobDefaults.model_validate(self.resources["job_defaults"])
        return self
