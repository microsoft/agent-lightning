"""Deployment helper for agl-lite (Python replacement for scripts/deploy.sh)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import time
from contextlib import suppress
from enum import StrEnum
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import typer
from dotenv import dotenv_values
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SECRET_NAME = "agl-lite-keys"


class DeployMode(StrEnum):
    IN_K8S = "agl-in-k8s"
    IN_HOST = "agl-in-host"
    EXTERNAL = "agl-external"


class DeploySettings(BaseSettings):
    """Deploy configuration — loaded from a .env file via --env-file.

    All fields are read from environment variables with the AGL_ prefix.
    Extra variables in the .env file (hook config, model endpoints, etc.)
    are silently ignored — the file serves as the single project config.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGL_",
        extra="ignore",
        env_ignore_empty=True,  # empty string values treated as unset → None/default
    )

    namespace: str
    mode: DeployMode

    base_url_k8s_accessible: str | None = None
    host_ip_bind: str = "0.0.0.0"
    host_port: int = 8080

    job_manifest_template: str = "deploy/controller/job-template.yaml.j2"
    max_pods_per_window: int = 100
    rate_limit_window_seconds: int = 10

    # User pod spec template — plain YAML file loaded by the base RolloutHooks.on_startup
    # into self._pod_spec (AGL_POD_SPEC_TEMPLATE).  Often the job-template.yaml in the
    # example folder.  When set, hooks get self._pod_spec for free without overriding
    # on_startup.  Defaults to None (no pod spec loaded by base).
    pod_spec_template: str | None = None
    gateway_config: str | None = None
    hooks: str | None = None
    log_dir: str | None = None
    wait_ready_timeout_seconds: int = 120
    local_state_dir: str = ".local"

    @field_validator("namespace")
    @classmethod
    def _namespace_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("namespace cannot be empty")
        return v

    @field_validator("base_url_k8s_accessible")
    @classmethod
    def _validate_http_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        p = urlparse(v)
        if p.scheme not in {"http", "https"} or not p.netloc:
            raise ValueError(f"invalid URL: {v}")
        return v

    @field_validator("host_ip_bind", "local_state_dir")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("field cannot be empty")
        return v

    @model_validator(mode="after")
    def _validate_mode_requirements(self) -> DeploySettings:
        if self.mode == DeployMode.IN_K8S:
            if self.base_url_k8s_accessible is not None:
                raise ValueError("AGL_BASE_URL_K8S_ACCESSIBLE must be unset when mode=agl-in-k8s")
        elif self.mode == DeployMode.EXTERNAL and not self.base_url_k8s_accessible:
            raise ValueError("AGL_BASE_URL_K8S_ACCESSIBLE is required when mode=agl-external")
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
    return urlparse(url).port


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
    with suppress(OSError):
        os.kill(pid, 15)
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


def _load_settings(env_file: Path) -> DeploySettings:
    if not env_file.exists():
        raise typer.BadParameter(f"Env file not found: {env_file}")
    try:
        settings_cls = cast(Any, DeploySettings)
        return settings_cls(_env_file=str(env_file))
    except Exception as e:
        raise typer.BadParameter(f"Invalid deploy config: {e}") from e


def deploy(env_file: str, cleanup: bool) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_file_path = Path(env_file)
    if not env_file_path.is_absolute():
        env_file_path = (Path.cwd() / env_file_path).resolve()

    cfg = _load_settings(env_file_path)
    ns = cfg.namespace

    local_state_dir = (repo_root / cfg.local_state_dir).resolve()
    pid_file = local_state_dir / "agl-lite-serve.pid"
    env_out = local_state_dir / "agl-lite.env"
    # Resolve log_dir: explicit config wins, else fall back to local_state_dir.
    resolved_log_dir = str((repo_root / cfg.log_dir).resolve()) if cfg.log_dir else str(local_state_dir)

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

    # Phase A/B/C: resolve endpoints + optional host serve
    host_url = f"http://127.0.0.1:{cfg.host_port}"

    if cfg.mode == DeployMode.IN_K8S:
        k8s_accessible_url = f"http://agl-lite.{ns}.svc:8080"

    elif cfg.mode == DeployMode.IN_HOST:
        local_state_dir.mkdir(parents=True, exist_ok=True)
        _stop_host_server(pid_file)

        cmd = ["uv", "run", "agl-lite", "serve", "--host", cfg.host_ip_bind, "--port", str(cfg.host_port)]
        if cfg.gateway_config:
            cmd += ["--gateway-config", cfg.gateway_config]
        if cfg.hooks:
            cmd += ["--hooks", cfg.hooks]
        # Build server env: env-file vars < os.environ < explicit overrides.
        # This ensures hook-specific vars (e.g. AGL_POD_SPEC_TEMPLATE) from the
        # .env file reach the server even if they weren't exported in the shell.
        server_env = {
            **dotenv_values(str(env_file_path)),
            **os.environ,
            "AGL_KEY": agl_key,
            "AGL_LOG_DIR": resolved_log_dir,
        }
        # Detach stdout/stderr so the server doesn't inherit the caller's pipe.
        # Logs go to resolved_log_dir/server.log via configure_logging().
        # Without this, any `cmd | tee file` in the caller's shell keeps
        # its pipe open until the server exits (never), blocking the script.
        p = subprocess.Popen(
                cmd,
                env=server_env,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
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
            raise RuntimeError(f"Host agl-lite server failed to become ready. See {resolved_log_dir}/server.log")

        if ctx == "minikube":
            k8s_accessible_url = f"http://host.minikube.internal:{cfg.host_port}"
        else:
            if cfg.base_url_k8s_accessible:
                k8s_accessible_url = cfg.base_url_k8s_accessible
            else:
                raise typer.BadParameter(
                    "mode=agl-in-host requires AGL_BASE_URL_K8S_ACCESSIBLE on non-minikube clusters"
                )

        if _host_is_localhost(k8s_accessible_url):
            raise typer.BadParameter(f"AGL_BASE_URL_K8S_ACCESSIBLE is not pod-reachable: {k8s_accessible_url}")

    else:  # agl-external
        k8s_accessible_url = cfg.base_url_k8s_accessible or ""
        if _host_is_localhost(k8s_accessible_url):
            raise typer.BadParameter(f"AGL_BASE_URL_K8S_ACCESSIBLE is not pod-reachable: {k8s_accessible_url}")
        host_url = k8s_accessible_url

    typer.echo(f"=== Mode: {cfg.mode.value} ===")
    typer.echo(f"K8s-accessible URL: {k8s_accessible_url}")
    typer.echo(f"Host URL:           {host_url}")

    # Phase D: Kubernetes resources + controller
    _run_shell(f"kubectl create namespace {shlex.quote(ns)} --dry-run=client -o yaml | kubectl apply -f -")

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
        "AGL_NAMESPACE": ns,
        "AGL_BASE_URL": k8s_accessible_url,
        "AGL_MAX_PODS_PER_WINDOW": str(cfg.max_pods_per_window),
        "AGL_RATE_LIMIT_WINDOW_SECONDS": str(cfg.rate_limit_window_seconds),
    }
    if cfg.gateway_config:
        cm_env["AGL_GATEWAY_CONFIG"] = cfg.gateway_config
    if cfg.hooks:
        cm_env["AGL_HOOKS"] = cfg.hooks
    if cfg.pod_spec_template:
        cm_env["AGL_POD_SPEC_TEMPLATE"] = cfg.pod_spec_template
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        for k, v in cm_env.items():
            f.write(f"{k}={v}\n")
        tmp_env = f.name

    try:
        _run_shell(
            f"kubectl -n {shlex.quote(ns)} create configmap agl-lite-config"
            f" --from-env-file={shlex.quote(tmp_env)}"
            " --dry-run=client -o yaml | kubectl apply -f -"
        )
    finally:
        Path(tmp_env).unlink(missing_ok=True)

    _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/controller/rbac.yaml")])
    _ensure_minikube_host_dns(k8s_accessible_url)

    # ConfigMap for the Jinja2 job manifest template.
    job_template_path = repo_root / cfg.job_manifest_template
    _run_shell(
        f"kubectl -n {shlex.quote(ns)} create configmap agl-controller-job-template"
        f" --from-file=job-template.yaml.j2={shlex.quote(str(job_template_path))}"
        " --dry-run=client -o yaml | kubectl apply -f -"
    )

    if cfg.mode == DeployMode.IN_K8S:
        _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/agl-lite/k8s.yaml")])

    _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/controller/k8s.yaml")])

    timeout = f"{cfg.wait_ready_timeout_seconds}s"
    if cfg.mode == DeployMode.IN_K8S:
        _run(["kubectl", "-n", ns, "wait", "--for=condition=available", "deployment/agl-lite", f"--timeout={timeout}"])
    _run([
        "kubectl",
        "-n",
        ns,
        "wait",
        "--for=condition=available",
        "deployment/agl-controller",
        f"--timeout={timeout}",
    ])

    local_state_dir.mkdir(parents=True, exist_ok=True)
    env_out.write_text(
        "\n".join(
            [
                "# Generated by agl-lite deploy",
                f'export AGL_KEY="{agl_key}"',
                f'export AGL_BASE_URL="{host_url}"',
                f'export AGL_BASE_URL_POD="{k8s_accessible_url}"',
                f'export AGL_NAMESPACE="{ns}"',
                "",
            ]
        )
    )

    typer.echo("\n=== Deploy complete ===")
    _run(["kubectl", "-n", ns, "get", "pods"], capture=False)
    typer.echo(f"Pod-facing URL:  {k8s_accessible_url}")
    typer.echo(f"Host-facing URL: {host_url}")
    typer.echo(f"Env file: {env_out}")
    typer.echo(f"  source {env_out}")


def deploy_command(
    env_file: str = typer.Option(..., "--env-file", help="Path to .env deploy config file"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Delete namespace and stop managed host service"),
) -> None:
    deploy(env_file=env_file, cleanup=cleanup)
