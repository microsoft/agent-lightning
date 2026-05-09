"""Controller configuration — pure data carrier, no env reads."""

from __future__ import annotations

from pydantic import BaseModel, field_validator


class ControllerSettings(BaseModel):
    """Settings for the agl-lite K8s controller. Constructed explicitly by cli.py."""

    # Connection to agl-lite server.
    base_url: str              # AGL_BASE_URL
    key: str = ""              # AGL_KEY — auth key; empty = auth disabled

    # K8s configuration.
    namespace: str             # AGL_NAMESPACE

    # Reconcile timing.
    poll_interval: int = 10    # AGL_POLL_INTERVAL — seconds between reconcile cycles
    max_queue_time: int = 3600 # AGL_MAX_QUEUE_TIME — max seconds a rollout stays in queuing

    # Job defaults.
    ttl_after_finished: int = 3600  # AGL_TTL_AFTER_FINISHED — ttlSecondsAfterFinished on Jobs

    # Pod creation rate limiting.
    max_pods_per_window: int = 100       # AGL_MAX_PODS_PER_WINDOW
    rate_limit_window_seconds: int = 10  # AGL_RATE_LIMIT_WINDOW_SECONDS

    # Job manifest template.
    job_manifest_template: str  # AGL_JOB_MANIFEST_TEMPLATE — always required

    @field_validator("max_pods_per_window", "rate_limit_window_seconds")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rate limit settings must be positive")
        return value
