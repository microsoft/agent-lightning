"""Job spec builder — pure function that converts a rollout into a K8s Job manifest.

No I/O, no K8s API calls. Easy to unit test.

Template format (Jinja2, two YAML documents separated by ---):
  Document 0: K8s Job manifest scaffold — Job metadata, spec shell, empty containers/volumes.
  Document 1: PodPatcher — env vars and volumes injected into ALL containers in the pod.

Merge order (later wins):
  1. manifest_template — Jinja2 template string → Job scaffold + PodPatcher defaults
  2. rollout.config.pod_spec — pod spec fragment assembled by the on_enqueue hook
  3. rollout.config.timeout / max_retries — per-sample execution policy

The hook owns all container-level customisation (image, command, env vars, volumes).
The controller only injects gateway env vars (via PodPatcher) and sets Job-level fields.
"""

from __future__ import annotations

import copy
from typing import Any

import yaml
from jinja2 import Template
from pydantic import BaseModel, Field

from agl_lite.controller.config import ControllerSettings
from agl_lite.schemas.rollout import Rollout


class PodPatcher(BaseModel):
    """Controller-managed contributions injected into all pod containers.

    Parsed from the second YAML document (after ---) in the Jinja2 job manifest template.
      env     — prepended to each container's env; container's own values win on name conflict.
      volumes — merged into pod spec; user's volumes win on name conflict.
    """

    env: list[dict[str, Any]] = Field(default_factory=list)
    volumes: list[dict[str, Any]] = Field(default_factory=list)


def build_job_name(rollout_id: str) -> str:
    """Deterministic Job name from rollout ID."""
    return f"agl-rollout-{rollout_id}"


def build_job_spec(
    rollout: Rollout,
    settings: ControllerSettings,
    manifest_template: str,
) -> dict[str, Any]:
    """Build a K8s Job manifest from rollout config + manifest template.

    Args:
        rollout:           The rollout to build a Job for. rollout.config.pod_spec
                           holds the pod spec fragment assembled by the on_enqueue hook.
        settings:          Controller settings (namespace, secret name, lite URL, ttl).
        manifest_template: Jinja2 template string (two YAML docs separated by ---).
                           Caller is responsible for loading this from disk before calling.
    """
    config = rollout.config

    # --- 1. Render Jinja2 template → two YAML documents ---
    rendered = Template(manifest_template).render(**_template_context(rollout, settings))
    docs = list(yaml.safe_load_all(rendered))

    job_dict: dict[str, Any] = copy.deepcopy(docs[0])
    patcher = PodPatcher.model_validate(docs[1] if len(docs) > 1 else {})

    pod_spec: dict[str, Any] = job_dict["spec"]["template"]["spec"]

    # --- 2. Merge rollout.config.pod_spec (hook-assembled) into scaffold ---
    user_deadline: int | None = None
    if config.pod_spec:
        user_pod = copy.deepcopy(config.pod_spec)
        user_containers: list[dict[str, Any]] = user_pod.pop("containers", [])
        user_volumes: list[dict[str, Any]] = user_pod.pop("volumes", [])
        user_deadline = user_pod.pop("activeDeadlineSeconds", None)

        pod_spec["containers"] = user_containers
        pod_spec["volumes"] = _merge_by_name(patcher.volumes, user_volumes)
        _deep_merge(pod_spec, user_pod)  # nodeSelector, tolerations, etc.
    else:
        pod_spec["volumes"] = list(patcher.volumes)

    # --- 3. Inject patcher env into ALL containers (container's own env wins on conflict) ---
    for container in pod_spec["containers"]:
        container["env"] = _merge_env(patcher.env, container.get("env", []))

    # --- 4. Set per-rollout job spec fields ---
    job_spec: dict[str, Any] = job_dict["spec"]
    job_spec["backoffLimit"] = config.max_retries if config.max_retries is not None else 0

    deadline = config.timeout or user_deadline
    if deadline is not None:
        job_spec["activeDeadlineSeconds"] = deadline
    else:
        job_spec.pop("activeDeadlineSeconds", None)

    return job_dict


# --- Private helpers ---


def _template_context(rollout: Rollout, settings: ControllerSettings) -> dict[str, Any]:
    """Build Jinja2 template variables."""
    return {
        "job_name": build_job_name(rollout.rollout_id),
        "rollout_id": rollout.rollout_id,
        "namespace": settings.namespace,
        "secret_name": settings.secret_name,
        "lite_url": settings.lite_url,
        "ttl_after_finished": settings.ttl_after_finished,
    }


def _merge_env(
    base_env: list[dict[str, Any]],
    override_env: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two env var lists. override_env wins on name conflict."""
    merged: dict[str, dict[str, Any]] = {e["name"]: e for e in base_env}
    for e in override_env:
        merged[e["name"]] = e
    return list(merged.values())


def _merge_by_name(
    base: list[dict[str, Any]],
    override: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two lists of name-keyed dicts. override wins on name conflict."""
    merged: dict[str, dict[str, Any]] = {item["name"]: item for item in base}
    for item in override:
        merged[item["name"]] = item
    return list(merged.values())


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place). Override wins on conflicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
