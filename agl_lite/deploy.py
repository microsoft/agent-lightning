"""Deployment helper for agl-lite (Python replacement for scripts/deploy.sh)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import typer

DeployMode = Literal["k8s", "host", "external"]


def _run(cmd: list[str], *, check: bool = True, capture: bool = True, input_text: str | None = None) -> str:
    p = subprocess.run(
        cmd,
        check=check,
        text=True,
        input=input_text,
        capture_output=capture,
    )
    return (p.stdout or "").strip()


def _run_shell(cmd: str, *, check: bool = True, capture: bool = True, input_text: str | None = None) -> str:
    p = subprocess.run(cmd, shell=True, check=check, text=True, input=input_text, capture_output=capture)
    return (p.stdout or "").strip()


def _parse_env_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        data[k.strip()] = v.strip()
    return data


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

    inserted = []
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


def deploy(
    config: str,
    mode: DeployMode,
    agl_host_bind: str,
    agl_host_port: int,
    cleanup: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_file = Path(config)
    if not env_file.is_absolute():
        env_file = (Path.cwd() / env_file).resolve()

    if not env_file.exists():
        raise typer.BadParameter(f"Config file not found: {env_file}")

    env_data = _parse_env_file(env_file)
    ns = env_data.get("AGL_K8S_NAMESPACE")
    if not ns:
        raise typer.BadParameter("AGL_K8S_NAMESPACE not set in config")

    host_state = repo_root / ".local"
    pid_file = host_state / "agl-lite-serve.pid"
    log_file = host_state / "agl-lite-serve.log"
    env_out = host_state / "agl-lite.env"

    if cleanup:
        typer.echo(f"=== Cleaning up namespace: {ns} ===")
        _run(["kubectl", "delete", "namespace", ns, "--ignore-not-found", "--wait"], check=False)
        _stop_host_server(pid_file)
        typer.echo("Done.")
        return

    agl_key = os.environ.get("AGL_KEY") or env_data.get("AGL_KEY", "")
    if not agl_key:
        raise typer.BadParameter("AGL_KEY not set (env or config)")

    ctx = _run(["kubectl", "config", "current-context"], check=False)

    pod_url_input = env_data.get("AGL_LITE_URL_POD") or env_data.get("AGL_LITE_URL", "")
    external_url_input = env_data.get("AGL_LITE_URL_EXTERNAL") or env_data.get("AGL_LITE_URL", "")

    if mode == "k8s":
        pod_url = f"http://agl-lite.{ns}.svc:8080"
        host_url = "http://127.0.0.1:8080"
    elif mode == "host":
        if not pod_url_input:
            if ctx == "minikube":
                pod_url = f"http://host.minikube.internal:{agl_host_port}"
            else:
                raise typer.BadParameter(
                    "--agl-in-host on non-minikube requires AGL_LITE_URL_POD (or legacy AGL_LITE_URL)"
                )
        else:
            pod_url = pod_url_input
        if ctx != "minikube" and any(x in pod_url for x in ["localhost", "127.0.0.1", "0.0.0.0"]):
            raise typer.BadParameter(f"Pod URL not reachable from remote cluster: {pod_url}")
        if ":" in pod_url.rsplit("/", 1)[-1]:
            try:
                agl_host_port = int(pod_url.rsplit(":", 1)[-1])
            except Exception:
                pass
        host_url = f"http://127.0.0.1:{agl_host_port}"
    else:
        if not external_url_input:
            raise typer.BadParameter("--agl-external requires AGL_LITE_URL_EXTERNAL (or legacy AGL_LITE_URL)")
        if any(x in external_url_input for x in ["localhost", "127.0.0.1", "0.0.0.0"]):
            raise typer.BadParameter(f"External URL not pod-reachable: {external_url_input}")
        pod_url = external_url_input
        host_url = external_url_input

    typer.echo(f"=== Mode: agl-in-{mode} ===")
    typer.echo(f"Pod URL:  {pod_url}")
    typer.echo(f"Host URL: {host_url}")

    # namespace
    _run_shell(f"kubectl create namespace {shlex.quote(ns)} --dry-run=client -o yaml | kubectl apply -f -")

    # secret
    secret_name = env_data.get("AGL_SECRET_NAME", "agl-lite-keys")
    _run_shell(
        " ".join(
            [
                f"kubectl -n {shlex.quote(ns)} create secret generic {shlex.quote(secret_name)}",
                f"--from-literal=AGL_KEY={shlex.quote(agl_key)}",
                "--dry-run=client -o yaml | kubectl apply -f -",
            ]
        )
    )

    # configmap env
    cm_env: dict[str, str] = {}
    for k, v in env_data.items():
        if k in {"AGL_KEY", "AGL_LITE_URL", "AGL_LITE_URL_POD", "AGL_LITE_URL_EXTERNAL"}:
            continue
        cm_env[k] = v
    cm_env["AGL_LITE_URL"] = pod_url

    # runtime overrides from caller env
    for k in ["AGL_GATEWAY_CONFIG", "AGL_HOOKS", "AGL_ARTIFACT_DIR"]:
        v = os.environ.get(k)
        if v:
            cm_env[k] = v

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

    # rbac + deployments
    _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/controller/rbac.yaml")])
    if mode == "k8s":
        _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/agl-lite/k8s.yaml")])
    _run(["kubectl", "apply", "-n", ns, "-f", str(repo_root / "deploy/controller/k8s.yaml")])

    _ensure_minikube_host_dns(pod_url)

    if mode == "k8s":
        _run(["kubectl", "-n", ns, "wait", "--for=condition=available", "deployment/agl-lite", "--timeout=120s"])
    _run(["kubectl", "-n", ns, "wait", "--for=condition=available", "deployment/agl-controller", "--timeout=120s"])

    # start host service when needed
    if mode == "host":
        host_state.mkdir(parents=True, exist_ok=True)
        _stop_host_server(pid_file)

        cmd = ["uv", "run", "agl-lite", "serve", "--host", agl_host_bind, "--port", str(agl_host_port)]
        gw = os.environ.get("AGL_GATEWAY_CONFIG")
        hk = os.environ.get("AGL_HOOKS")
        ad = os.environ.get("AGL_ARTIFACT_DIR")
        if gw:
            cmd += ["--gateway-config", gw]
        if hk:
            cmd += ["--hooks", hk]
        if ad:
            cmd += ["--artifact-dir", ad]

        with open(log_file, "w") as lf:
            p = subprocess.Popen(cmd, env={**os.environ, "AGL_KEY": agl_key}, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        pid_file.write_text(str(p.pid))

        # health check
        ok = False
        for _ in range(40):
            r = subprocess.run(["curl", "-sf", f"{host_url}/healthz"], capture_output=True)
            if r.returncode == 0:
                ok = True
                break
            if p.poll() is not None:
                break
            import time

            time.sleep(1)
        if not ok:
            raise RuntimeError(f"Host agl-lite server failed to become ready. See {log_file}")

    host_state.mkdir(parents=True, exist_ok=True)
    env_out.write_text(
        "\n".join(
            [
                "# Generated by agl-lite deploy",
                f'export AGL_LITE_URL="{host_url}"',
                f'export AGL_LITE_URL_POD="{pod_url}"',
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
    config: str = typer.Option("deploy/.env", "--config", help="Path to env config file"),
    agl_in_k8s: bool = typer.Option(False, "--agl-in-k8s", help="Run agl-lite in Kubernetes"),
    agl_in_host: bool = typer.Option(False, "--agl-in-host", help="Run agl-lite on this host"),
    agl_external: bool = typer.Option(False, "--agl-external", help="Use external agl-lite service"),
    controller_only: bool = typer.Option(False, "--controller-only", hidden=True),
    no_serve: bool = typer.Option(False, "--no-serve", hidden=True),
    agl_host_bind: str = typer.Option("0.0.0.0", "--agl-host-bind", help="Host bind address for agl-lite serve"),
    agl_host_port: int = typer.Option(8080, "--agl-host-port", help="Host port for agl-lite serve"),
    cleanup: bool = typer.Option(False, "--cleanup", help="Delete namespace and stop managed host service"),
) -> None:
    mode: DeployMode = "k8s"
    if agl_in_host or controller_only or no_serve:
        mode = "host"
    if agl_external:
        mode = "external"
    if agl_in_k8s:
        mode = "k8s"

    deploy(config=config, mode=mode, agl_host_bind=agl_host_bind, agl_host_port=agl_host_port, cleanup=cleanup)
