"""Controller configuration — pure data carrier, no env reads."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator


class RunnerType(StrEnum):
    """Controller execution backend, selected at startup."""

    K8S = "k8s"
    LOCAL = "local"


class ControllerSettings(BaseModel):
    """Settings for the agl-lite controller. Constructed explicitly by cli.py."""

    # Connection to agl-lite server.
    base_url: str  # AGL_BASE_URL
    key: str = ""  # AGL_KEY — auth key; empty = auth disabled

    # K8s configuration.
    namespace: str  # AGL_NAMESPACE

    # Reconcile timing.
    poll_interval: int = 10  # AGL_POLL_INTERVAL — seconds between reconcile cycles
    max_queue_time: int = 3600  # AGL_MAX_QUEUE_TIME — max seconds a rollout stays in queuing

    # Job defaults.
    ttl_after_finished: int = 3600  # AGL_TTL_AFTER_FINISHED — ttlSecondsAfterFinished on Jobs

    # Pod creation rate limiting.
    max_pods_per_window: int = 100  # AGL_MAX_PODS_PER_WINDOW
    rate_limit_window_seconds: int = 10  # AGL_RATE_LIMIT_WINDOW_SECONDS

    # Runner selection.
    runner_type: RunnerType = RunnerType.K8S  # AGL_RUNNER_TYPE

    # Job manifest template — required when runner_type=k8s, ignored otherwise.
    # Validated at the CLI boundary (cli.py) rather than here so the error
    # message can point at the missing flag/env var.
    job_manifest_template: str | None = None  # AGL_JOB_MANIFEST_TEMPLATE

    # Local runner — required when runner_type=local, validated in this class.
    local_pool_size: int | None = None  # AGL_LOCAL_POOL_SIZE
    local_agent_class: str | None = None  # AGL_LOCAL_AGENT_CLASS
    local_tick_interval: float = 5.0  # AGL_LOCAL_TICK_INTERVAL

    @field_validator("max_pods_per_window", "rate_limit_window_seconds")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("rate limit settings must be positive")
        return value

    @field_validator("local_pool_size")
    @classmethod
    def _positive_pool_size(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("local_pool_size must be positive")
        return value

    @field_validator("local_tick_interval")
    @classmethod
    def _positive_tick_interval(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("local_tick_interval must be positive")
        return value

    @model_validator(mode="after")
    def _require_local_fields_for_local(self) -> ControllerSettings:
        if self.runner_type != RunnerType.LOCAL:
            return self
        missing = [
            name
            for name, value in (
                ("local_pool_size", self.local_pool_size),
                ("local_agent_class", self.local_agent_class),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"runner_type=local requires {missing} (set via AGL_LOCAL_* env vars or --local-* options)"
            )
        return self
