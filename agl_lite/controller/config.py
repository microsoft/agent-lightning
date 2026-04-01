"""Controller configuration — settings for the K8s controller process."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class ControllerSettings(BaseSettings):
    """Settings for the agl-lite K8s controller.

    All values can be set via environment variables (prefixed with AGL_).
    """

    model_config = {"env_prefix": "AGL_"}

    # Connection to agl-lite server.
    base_url: str = "http://localhost:8000"  # agl-lite server URL; AGL_BASE_URL
    key: str = ""  # AGL_KEY — auth key for agl-lite API

    # K8s configuration.
    namespace: str = "default"
    secret_name: str = "agl-api-keys"  # K8s Secret with OPENAI_API_KEY, ANTHROPIC_API_KEY

    # Reconcile timing.
    poll_interval: int = 10  # seconds between periodic reconcile cycles
    max_queue_time: int = 3600  # max seconds a rollout can stay in queuing (default 1h)

    # Job defaults (can be overridden by resources snapshot).
    ttl_after_finished: int = 3600  # ttlSecondsAfterFinished on Jobs (pod GC safety)

    job_manifest_template: str  # Jinja2 job scaffold; AGL_JOB_MANIFEST_TEMPLATE — always required in pod
