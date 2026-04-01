"""Job spec builder — pure function that converts rollout + pod spec fragment into a K8s Job manifest.

No I/O, no K8s API calls. Easy to unit test.

Template format (Jinja2, two YAML documents separated by ---):
  Document 0: K8s Job manifest scaffold — Job metadata, spec shell, empty containers/volumes.
  Document 1: PodPatcher — env vars and volumes injected into ALL containers in the pod.

Merge order (later wins):
  1. manifest_template — Jinja2 template string → Job scaffold + PodPatcher defaults
  2. pod_spec          — pod spec fragment from resources (containers, volumes, pod fields)
  3. rollout.config    — image, command, env vars, mounts → agent container
  4. rollout.config.overrides — per-rollout K8s overrides, name-matched containers

PodPatcher env vars are injected into every container; container's own env wins on conflict.
rollout.config.environment_variables further override patcher env on the agent container.
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
    pod_spec: dict[str, Any] | None,
    settings: ControllerSettings,
    manifest_template: str,
) -> dict[str, Any]:
    """Build a K8s Job manifest from rollout config + pod spec fragment + manifest template.

    Args:
        rollout:           The rollout to build a Job for.
        pod_spec:          Pod spec fragment from the resource snapshot — containers, volumes,
                           nodeSelector, tolerations, etc. Sourced from resources["job_template"].
                           None means no user-provided spec; agent container is created bare.
        settings:          Controller settings (namespace, secret name, lite URL, ttl, ...).
        manifest_template: Jinja2 template string (two YAML docs separated by ---). Caller
                           is responsible for loading this from disk before calling.
    """
    config = rollout.config

    # --- 1. Render Jinja2 template → two YAML documents ---
    rendered = Template(manifest_template).render(**_template_context(rollout, settings))
    docs = list(yaml.safe_load_all(rendered))

    job_dict: dict[str, Any] = copy.deepcopy(docs[0])
    patcher = PodPatcher.model_validate(docs[1] if len(docs) > 1 else {})

    scaffold_pod_spec: dict[str, Any] = job_dict["spec"]["template"]["spec"]

    # --- 2. Normalise pod spec fragment from resources ---
    user_pod = copy.deepcopy(pod_spec) if pod_spec else {}
    # Unwrap {"spec": {...}} form (backward compat).
    if "spec" in user_pod and "containers" not in user_pod:
        user_pod = user_pod["spec"]

    user_containers: list[dict[str, Any]] = user_pod.pop("containers", [])
    user_volumes: list[dict[str, Any]] = user_pod.pop("volumes", [])
    # activeDeadlineSeconds at root of user fragment → hoist to job spec later.
    user_deadline: int | None = user_pod.pop("activeDeadlineSeconds", None)

    # --- 3. Merge user pod spec into scaffold ---
    scaffold_pod_spec["containers"] = user_containers  # replaces empty []
    scaffold_pod_spec["volumes"] = _merge_by_name(patcher.volumes, user_volumes)  # user wins on name
    _deep_merge(scaffold_pod_spec, user_pod)  # nodeSelector, tolerations, serviceAccountName, etc.

    # --- 4. Ensure agent container exists (before env injection) ---
    _ensure_agent_container(scaffold_pod_spec)

    # --- 5. Inject patcher env into ALL containers (container's own env wins on conflict) ---
    for container in scaffold_pod_spec["containers"]:
        container["env"] = _merge_env(patcher.env, container.get("env", []))

    # --- 6. Apply rollout.config named fields to agent container ---
    agent = _get_agent_container(scaffold_pod_spec)

    if config.image:
        agent["image"] = config.image

    if config.command:
        agent["command"] = config.command

    # rollout.config env vars override patcher env on the agent container.
    if config.environment_variables:
        rollout_env = [{"name": k, "value": v} for k, v in config.environment_variables.items()]
        agent["env"] = _merge_env(agent.get("env", []), rollout_env)

    # Volume mounts from rollout config.
    if config.mount:
        agent.setdefault("volumeMounts", [])
        for m in config.mount:
            agent["volumeMounts"].append(
                {"name": m.name, "mountPath": m.mount_path, "readOnly": m.read_only}
            )
        scaffold_pod_spec.setdefault("volumes", [])
        for m in config.mount:
            vol: dict[str, Any] = {"name": m.name}
            if m.source.startswith("/"):
                vol["hostPath"] = {"path": m.source}
            elif m.source.startswith("pvc:"):
                vol["persistentVolumeClaim"] = {"claimName": m.source[4:]}
            else:
                vol["configMap"] = {"name": m.source}
            scaffold_pod_spec["volumes"].append(vol)

    # --- 7. Apply per-rollout overrides (name-matched container merge) ---
    if config.overrides:
        overrides = copy.deepcopy(config.overrides)
        override_containers = overrides.pop("containers", [])
        for oc in override_containers:
            oc_name = oc.get("name")
            if not oc_name:
                continue
            for container in scaffold_pod_spec["containers"]:
                if container.get("name") == oc_name:
                    _deep_merge(container, oc)
                    break
        _deep_merge(scaffold_pod_spec, overrides)

    # --- 8. Set per-rollout job spec fields ---
    job_spec: dict[str, Any] = job_dict["spec"]
    job_spec["backoffLimit"] = config.max_retries if config.max_retries is not None else 0

    # activeDeadlineSeconds: rollout.config.timeout > user fragment field > absent.
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


def _ensure_agent_container(pod_spec: dict[str, Any]) -> None:
    """Ensure pod spec has an 'agent' container, inserting one at position 0 if absent."""
    for c in pod_spec["containers"]:
        if c.get("name") == "agent":
            return
    pod_spec["containers"].insert(0, {"name": "agent"})


def _get_agent_container(pod_spec: dict[str, Any]) -> dict[str, Any]:
    """Return the 'agent' container (must exist — call _ensure_agent_container first)."""
    for c in pod_spec["containers"]:
        if c.get("name") == "agent":
            return c
    raise RuntimeError("agent container not found — call _ensure_agent_container first")


def _merge_env(
    base_env: list[dict[str, Any]],
    override_env: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two env var lists. override_env wins on name conflict.

    Order: base entries first, then override-only extras appended at the end.
    """
    merged: dict[str, dict[str, Any]] = {e["name"]: e for e in base_env}
    for e in override_env:
        merged[e["name"]] = e  # override wins
    return list(merged.values())


def _merge_by_name(
    base: list[dict[str, Any]],
    override: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge two lists of name-keyed dicts. override wins on name conflict."""
    merged: dict[str, dict[str, Any]] = {item["name"]: item for item in base}
    for item in override:
        merged[item["name"]] = item  # override wins
    return list(merged.values())


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place). Override wins on conflicts."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
