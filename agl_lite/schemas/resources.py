"""Resource schemas — versioned immutable snapshots."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ResourcesUpdate(BaseModel):
    """Immutable resource snapshot. One bundle containing all named resources.

    The 'job_template' key is reserved for the K8s pod spec template (raw dict,
    any valid K8s fields). The store does not validate it — validation happens
    when the controller renders and submits the Job to K8s.

    All other keys are user-defined and opaque (prompts, eval configs, etc.).
    """

    resources_id: str
    resources: dict[str, Any]  # {"job_template": {...}, "system_prompt": "...", ...}
    created_at: float
    version: int = 1
