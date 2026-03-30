"""Deployment helper for agl-lite (Python replacement for scripts/deploy.sh)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import typer
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

SECRET_NAME = "agl-lite-keys"


class DeployMode(str, Enum):
    IN_K8S = "agl-in-k8s"
    IN_HOST = "agl-in-host"
    EXTERNAL = "agl-external"


class HostServeConfig(BaseModel):
    bind: str = Field(default="0.0.0.0", description="Bind address for host-side `agl-lite serve`.")
    port: int = Field(default=8080, ge=1, le=65535, description="Listen port for host-side `agl-lite serve`.")


class ControllerConfig(BaseModel):
    poll_interval_seconds: int = Field(default=10, ge=1)
    max_queue_time_seconds: int = Field(default=3600, ge=1)


class ServerRuntimeConfig(BaseModel):
    gateway_config: str | None = Field(default=None, description="Path to gateway config YAML (loaded by agl-lite serve).")
    hooks: str | None = Field(default=None, description="Path to hooks Python file (loaded by agl-lite serve).")
    artifact_dir: str | None = Field(default=None, description="Artifact directory path for agl-lite serve.")


class DeployConfig(BaseModel):
    namespace: str = Field(description="Kubernetes namespace where agl-lite/controller resources are deployed.")
    mode: DeployMode = Field(description="Deployment mode: agl-in-k8s | agl-in-host | agl-external.")

    agl_base_url_pod: str | None = Field(
        default=None,
        description="agl-lite base URL as seen by controller/agent pods (pod-facing URL).",
    )
    agl_base_url_external: str | None = Field(
        default=None,
        description="agl-lite base URL for external mode (used by both pods and host clients).",
    )

    host_serve: HostServeConfig = Field(default_factory=HostServeConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    server_runtime: ServerRuntimeConfig = Field(default_factory=ServerRuntimeConfig)

    wait_ready_timeout_seconds: int = Field(default=120, ge=1)
    local_state_dir: str = Field(default=".local", description="Directory for generated local state files (.env, pid, logs).")

    @field_validator("namespace")
    @classmethod
    def _namespace_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("namespace cannot be empty")
        return v

    @field_validator("agl_base_url_pod", "agl_base_url_external")
    @classmethod
    def _validate_http_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        p = urlparse(v)
        if p.scheme not in {"http", "https"} or not p.netloc:
            raise ValueError(f"invalid URL: {v}")
        return v

    @field_validator("local_state_dir")
    @classmethod
    def _validate_local_state_dir(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("local_state_dir cannot be empty")
        return v

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> DeployConfig:
        if self.mode == DeployMode.IN_K8S:
            if self.agl_base_url_pod is not None:
                raise ValueError("agl_base_url_pod must be unset when mode=agl-in-k8s (auto-derived)")
            if self.agl_base_url_external is not None:
                raise ValueError("agl_base_url_external must be unset when mode=agl-in-k8s")
        elif self.mode == DeployMode.IN_HOST:
            if self.agl_base_url_external is not None:
                raise ValueError("agl_base_url_external must be unset when mode=agl-in-host")
        elif self.mode == DeployMode.EXTERNAL:
            if not self.agl_base_url_external:
                raise ValueError("agl_base_url_external is required when mode=agl-external")
            if self.agl_base_url_pod is not None:
                raise ValueError("agl_base_url_pod must be unset when mode=agl-external")
        return self


def _run(cmd: list[str], *, check: bool = True, capture: bool = True, input_text: str | None = None) -> str:
    p = subprocess.run(cmd, check=check, text=True, input=input_text, capture_output=capture)
    return (p.stdout or "").strip()


def _run_shell(cmd: str, *, check: bool = True) -> str:
    p = subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True)
    return (p.stdout or "").strip()


def _host_is_localhost(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


def _port_from_url(url: str) -> int | None:
    p = urlparse(url)
    return p.port


def _stop_host_server(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text().strip())
    except Exception:
        pid_file.unlink(missing_ok=True)
        return

    try:
        os.kill(pid, 0)
    except OSError:
        pid_file.unlink(missing_ok=True)
        return

    typer.echo(f"--- Stopping host agl-lite server (pid={pid}) ---")
    try:
        os.kill(pid, 15)
    except OSError:
        pass
    pid_file.unlink(missing_ok=True)


def _ensure_minikube_host_dns(url_pod: str) -> None:
    if "host.minikube.internal" not in url_pod:
        return

    ctx = _run(["kubectl", "config", "current-context"], check=False)
    if ctx != "minikube":
        return

    corefile = _run(
        ["kubectl", "-n", "kube-system", "get", "configmap", "coredns", "-o", "jsonpath={.data.Corefile}"],
        check=False,
    )
    if "host.minikube.internal" in corefile:
        typer.echo("✓ CoreDNS already resolves host.minikube.internal")
        return

    typer.echo("⚠ Patching CoreDNS so pods can resolve host.minikube.internal...")
    host_ip = _run(["minikube", "ssh", "ip route | grep default | awk '{print $3}'"], check=False).replace("\r", "")
    if not host_ip:
        typer.echo("WARNING: Could not detect minikube host IP. Pods may not reach host service.")
        return

    inserted: list[str] = []
    for line in corefile.splitlines():
        inserted.append(line)
        if line.strip() == "ready":
            inserted.extend(
                [
                    "    hosts {",
                    f"       {host_ip} host.minikube.internal",
                    "       fallthrough",
                    "    }",
                ]
            )
    new_corefile = "\n".join(inserted)

    cm_raw = _run(["kubectl", "-n", "kube-system", "get", "configmap", "coredns", "-o", "json"])
    cm = json.loads(cm_raw)
    cm["data"]["Corefile"] = new_corefile
    _run(["kubectl", "apply", "-f", "-"], input_text=json.dumps(cm))

    _run(["kubectl", "-n", "kube-system", "rollout", "restart", "deployment", "coredns"], check=False)
    _run(
        ["kubectl", "-n", "kube-system", "wait", "--for=condition=available", "deployment/coredns", "--timeout=30s"],
        check=False,
    )
    typer.echo(f"✓ CoreDNS patched: host.minikube.internal → {host_ip}")


def _load_config(config_path: Path) -> DeployConfig:
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise typer.BadParameter("Config YAML must be a mapping/object")
    try:
        return DeployConfig.model_validate(raw)
    except Exception as e:  # pydantic raises ValidationError
        raise typer.BadParameter(f"Invalid deploy config: {e}") from e


def deploy(config: str, cleanup: bool) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = Path(config)
    if not config_path.is_absolute():
        config_path = (Path.cwd() / config_path).resolve()
    if not config_path.exists():
        raise typer.BadParameter(f"Config file not found: {config_path}")

    cfg = _load_config(config_path)
    ns = cfg.namespace

    local_state_dir = (repo_root / cfg.local_state_dir).resolve()
    pid_file = local_state_dir / "agl-lite-serve.pid"
    log_file = local_state_dir / "agl-lite-serve.log"
    env_out = local_state_dir / "agl-lite.env"

    if cleanup:
        typer.echo(f"=== Cleaning up namespace: {ns} ===")
        _run(["kubectl", "delete", "namespace", ns, "--ignore-not-found", "--wait"], check=False)
        _stop_host_server(pid_file)
        typer.echo("Done.")
        return

    agl_key = os.environ.get("AGL_KEY", "")
    if not agl_key:
        raise typer.BadParameter("AGL_KEY must be provided via environment, e.g. `export AGL_KEY=...`")

    ctx = _run(["kubectl", "config", "current-context"], check=False)

    if cfg.mode == DeployMode.IN_K8S:
        pod_url = f"http://agl-lite.{ns}.svc:8080"
        host_url = "http://127.0.0.1:8080"

    elif cfg.mode == DeployMode.IN_HOST:
        if cfg.agl_base_url_pod:
            pod_url = cfg.agl_base_url_pod
        elif ctx == "minikube":
            pod_url = f"http://host.minikube.internal:{cfg.host_serve.port}"
        else:
            raise typer.BadParameter(
                "mode=agl-in-host on non-minikube requires agl_base_url_pod (pod-reachable host URL)."
            )

        if ctx != "minikube" and _host_is_localhost(pod_url):
            raise typer.BadParameter(f"agl_base_url_pod is not pod-reachable on remote cluster: {pod_url}")

        pod_port = _port_from_url(pod_url)
        if pod_port is not None and pod_port != cfg.host_serve.port:
            raise typer.BadParameter(
                f"agl_base_url_pod port ({pod_port}) must match host_serve.port ({cfg.host_serve.port}) in agl-in-host mode"
            )

        host_url = f"http://127.0.0.1:{cfg.host_serve.port}"

    else:
        pod_url = cfg.agl_base_url_external or ""
        if _host_is_localhost(pod_url):
            raise typer.BadParameter(f"agl_base_url_external is not pod-reachable: {pod_url}")
        host_url = pod_url

    typer.echo(f"=== Mode: {cfg.mode.value} ===")
    typer.echo(f"Pod URL:  {pod_url}")
    typer.echo(f"Host URL: {host_url}")

    # namespace
    _run_shell(f"kubectl create namespace {shlex.quote(ns)} --dry-run=client -o yaml | kubectl apply -f -")

    # secret
    _run_shell(
        " ".join(
            [
                f"kubectl -n {shlex.quote(ns)} create secret generic {SECRET_NAME}",
                f"--from-literal=AGL_KEY={shlex.quote(agl_key)}",
                "--dry-run=client -o yaml | kubectl apply -f -",
            ]
        )
    )

    cm_env: dict[str, str] = {
        "AGL_K8S_NAMESPACE": ns,
        "AGL_SECRET_NAME": SECRET_NAME,
        "AGL_BASE_URL": pod_url,
        "AGL_POLL_INTERVAL": str(cfg.controller.poll_interval_seconds),
        "AGL_MAX_QUEUE_TIME": str(cfg.controller.max_queue_time_seconds),
    }

    if cfg.server_runtime.gateway_config:
        cm_env["AGL_GATEWAY_CONFIG"] = cfg.server_runtime.gateway_config
    if cfg.server_runtime.hooks:
        cm_env["AGL_HOOKS"] = cfg.server_runtime.hooks
    if cfg.server_runtime.artifact_dir:
        cm_env["AGL_ARTIFACT_DIR"] = cfg.server_runtime.artifact_dir

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for k, v in cm_env.items():
            f.write(f"{k}={v}\n")
        tmp_env = f.name

    try:
        _run_shell(
            f"kubectl -n {shlex.quote(ns)} create configmap agl-lite-config --from-env-file={shlex.quote(tmp_env)} --dry-run=client -o yaml | kubectl apply -f -"
        )
    finally:
        Path(tmp_env).unlink(missing_ok=True)

    _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/controller/rbac.yaml")])
    if cfg.mode == DeployMode.IN_K8S:
        _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/agl-lite/k8s.yaml")])
    _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/controller/k8s.yaml")])

    _ensure_minikube_host_dns(pod_url)

    timeout = f"{cfg.wait_ready_timeout_seconds}s"
    if cfg.mode == DeployMode.IN_K8S:
        _run(["kubectl", "-n", ns, "wait", "--for=condition=available", "deployment/agl-lite", f"--timeout={timeout}"])
    _run(["kubectl", "-n", ns, "wait", "--for=condition=available", "deployment/agl-controller", f"--timeout={timeout}"])

    if cfg.mode == DeployMode.IN_HOST:
        local_state_dir.mkdir(parents=True, exist_ok=True)
        _stop_host_server(pid_file)

        cmd = [
            "uv",
            "run",
            "agl-lite",
            "serve",
            "--host",
            cfg.host_serve.bind,
            "--port",
            str(cfg.host_serve.port),
        ]
        if cfg.server_runtime.gateway_config:
            cmd += ["--gateway-config", cfg.server_runtime.gateway_config]
        if cfg.server_runtime.hooks:
            cmd += ["--hooks", cfg.server_runtime.hooks]
        if cfg.server_runtime.artifact_dir:
            cmd += ["--artifact-dir", cfg.server_runtime.artifact_dir]

        with open(log_file, "w") as lf:
            p = subprocess.Popen(
                cmd,
                env={**os.environ, "AGL_KEY": agl_key},
                stdout=lf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_file.write_text(str(p.pid))

        ready = False
        for _ in range(cfg.wait_ready_timeout_seconds):
            r = subprocess.run(["curl", "-sf", f"{host_url}/healthz"], capture_output=True)
            if r.returncode == 0:
                ready = True
                break
            if p.poll() is not None:
                break
            time.sleep(1)
        if not ready:
            raise RuntimeError(f"Host agl-lite server failed to become ready. See {log_file}")

    local_state_dir.mkdir(parents=True, exist_ok=True)
    env_out.write_text(
        "\n".join(
            [
                "# Generated by agl-lite deploy",
                f'export AGL_BASE_URL="{host_url}"',
                f'export AGL_BASE_URL_POD="{pod_url}"',
                f'export AGL_K8S_NAMESPACE="{ns}"',
                "",
            ]
        )
    )

    typer.echo("\n=== Deploy complete ===")
    _run(["kubectl", "-n", ns, "get", "pods"], capture=False)
    typer.echo(f"Pod-facing URL:  {pod_url}")
    typer.echo(f"Host-facing URL: {host_url}")
    typer.echo(f"Env file: {env_out}")
    typer.echo(f"  source {env_out}")


def deploy_command(
    config: str = typer.Option(..., "--config", help="Path to deploy YAML config file"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Delete namespace and stop managed host service"),
) -> None:
    deploy(config=config, cleanup=cleanup)
