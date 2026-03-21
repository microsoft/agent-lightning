#!/usr/bin/env python3
"""Deploy agl-lite infrastructure to K8s.

Usage:
    export AGL_KEY=$(openssl rand -hex 32)
    python scripts/deploy.py                   # deploy
    python scripts/deploy.py --teardown        # teardown

Reads: deploy/config.yaml (all config), AGL_KEY env var (secret).
Creates: namespace, Secret, ConfigMap, RBAC, agl-lite serve, controller.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy"
CONFIG_PATH = DEPLOY_DIR / "config.yaml"


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, print it, and return the result."""
    print(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def kubectl(*args: str) -> subprocess.CompletedProcess:
    return run(["kubectl", *args])


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        print(f"ERROR: {CONFIG_PATH} not found. Copy from config.example.yaml and edit.")
        sys.exit(1)
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_agl_key() -> str:
    key = os.environ.get("AGL_KEY", "")
    if not key:
        print("ERROR: AGL_KEY environment variable not set.")
        print("  export AGL_KEY=$(openssl rand -hex 32)")
        sys.exit(1)
    return key


def deploy() -> None:
    config = load_config()
    agl_key = get_agl_key()

    k8s_config = config.get("k8s", {})
    controller_config = config.get("controller", {})

    ns = k8s_config.get("AGL_K8S_NAMESPACE", "agl")
    secret_name = k8s_config.get("AGL_SECRET_NAME", "agl-lite-keys")

    print(f"=== Deploying agl-lite to namespace: {ns} ===\n")

    # 1. Namespace
    print("--- Creating namespace ---")
    run(["bash", "-c",
         f"kubectl create namespace {ns} --dry-run=client -o yaml | kubectl apply -f -"])

    # 2. Secret
    print("\n--- Creating secret ---")
    run(["bash", "-c",
         f'kubectl -n {ns} create secret generic {secret_name}'
         f' --from-literal=AGL_KEY="{agl_key}"'
         f' --dry-run=client -o yaml | kubectl apply -f -'])

    # 3. ConfigMap — flatten all sections into individual keys + include full file
    print("\n--- Creating configmap ---")
    literals: dict[str, str] = {}

    # k8s section → flat keys
    for key, value in k8s_config.items():
        literals[key] = str(value)

    # controller section → prefixed keys
    prefix_map = {"poll_interval": "AGL_POLL_INTERVAL", "max_queue_time": "AGL_MAX_QUEUE_TIME"}
    for key, value in controller_config.items():
        env_key = prefix_map.get(key, f"AGL_{key.upper()}")
        literals[env_key] = str(value)

    # Build kubectl command
    cm_cmd = f"kubectl -n {ns} create configmap agl-lite-config"
    cm_cmd += f" --from-file=config.yaml={CONFIG_PATH}"
    for key, value in literals.items():
        cm_cmd += f' --from-literal={key}="{value}"'
    cm_cmd += " --dry-run=client -o yaml | kubectl apply -f -"
    run(["bash", "-c", cm_cmd])

    # 4. RBAC
    print("\n--- Applying RBAC ---")
    kubectl("apply", "-n", ns, "-f", str(DEPLOY_DIR / "controller" / "rbac.yaml"))

    # 5. Deployments
    print("\n--- Deploying agl-lite serve ---")
    kubectl("apply", "-n", ns, "-f", str(DEPLOY_DIR / "agl-lite" / "k8s.yaml"))

    print("\n--- Deploying controller ---")
    kubectl("apply", "-n", ns, "-f", str(DEPLOY_DIR / "controller" / "k8s.yaml"))

    # 6. Wait
    print("\n--- Waiting for pods ---")
    kubectl("-n", ns, "wait", "--for=condition=available", "deployment/agl-lite", "--timeout=120s")
    kubectl("-n", ns, "wait", "--for=condition=available", "deployment/agl-controller", "--timeout=120s")

    # 7. Summary
    print(f"\n=== agl-lite deployed to namespace: {ns} ===")
    kubectl("-n", ns, "get", "pods")
    print(f"\nTo access from host:")
    print(f"  kubectl -n {ns} port-forward svc/agl-lite 8080:8080")
    print(f"  export AGL_LITE_URL=http://localhost:8080 AGL_KEY=<your-key>")
    print(f"  agl-client health")


def teardown() -> None:
    config = load_config()
    ns = config.get("k8s", {}).get("AGL_K8S_NAMESPACE", "agl")
    print(f"=== Tearing down namespace: {ns} ===")
    kubectl("delete", "namespace", ns, "--ignore-not-found", "--wait")
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy agl-lite to K8s")
    parser.add_argument("--teardown", action="store_true", help="Tear down the deployment")
    args = parser.parse_args()

    if args.teardown:
        teardown()
    else:
        deploy()


if __name__ == "__main__":
    main()
