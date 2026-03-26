#!/bin/bash
# Deploy agl-lite infrastructure to K8s.
#
# Usage:
#   scripts/deploy.sh                    # deploy agl-lite server + controller into K8s
#   scripts/deploy.sh --controller-only  # deploy controller only (server runs externally)
#   scripts/deploy.sh --cleanup          # remove everything
#
# Modes:
#   Default (in-cluster):
#     Both agl-lite server and controller run inside K8s.
#     AGL_LITE_URL is auto-set to http://agl-lite.<namespace>.svc:8080.
#     Pods reach the server via cluster-internal DNS — no special networking.
#
#   --controller-only:
#     Only the K8s controller (and agent infrastructure) is deployed.
#     The agl-lite server runs externally on the compute backend (host machine),
#     typically colocated with model servers (vLLM) that have internal endpoints.
#     Set AGL_LITE_URL in deploy/.env to the external server address.
#     On minikube, this script auto-patches CoreDNS so pods can resolve
#     host.minikube.internal.
#
# Reads: deploy/.env (config), AGL_KEY env var or from .env (secret).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$REPO_ROOT/deploy/.env"

# --- Load config ---
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: deploy/.env not found. Copy from deploy/.env.example and edit."
    exit 1
fi
source "$ENV_FILE"

NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set in .env}"

# --- Cleanup mode ---
if [[ "${1:-}" == "--cleanup" || "${1:-}" == "--teardown" ]]; then
    echo "=== Cleaning up namespace: $NS ==="
    kubectl delete namespace "$NS" --ignore-not-found --wait
    echo "Done."
    exit 0
fi

# --- Parse flags ---
CONTROLLER_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --controller-only) CONTROLLER_ONLY=true ;;
        # Keep --no-serve as hidden alias for backwards compat
        --no-serve) CONTROLLER_ONLY=true ;;
    esac
done

# --- Helper: ensure host.minikube.internal resolves from pods ---
ensure_minikube_host_dns() {
    # Only relevant on minikube — check by context name
    local ctx
    ctx=$(kubectl config current-context 2>/dev/null || echo "")
    if [[ "$ctx" != "minikube" ]]; then
        return 0
    fi

    # Check if CoreDNS already has a hosts block for host.minikube.internal
    local corefile
    corefile=$(kubectl -n kube-system get configmap coredns -o jsonpath='{.data.Corefile}' 2>/dev/null || echo "")
    if echo "$corefile" | grep -q "host.minikube.internal"; then
        echo "✓ CoreDNS already resolves host.minikube.internal"
        return 0
    fi

    echo "⚠ Patching CoreDNS so pods can resolve host.minikube.internal..."
    local host_ip
    host_ip=$(minikube ssh "ip route | grep default | awk '{print \$3}'" 2>/dev/null | tr -d '\r\n')
    if [ -z "$host_ip" ]; then
        echo "WARNING: Could not detect minikube host IP. Pods may not reach the external server."
        return 1
    fi

    # Inject hosts block into Corefile (after the health check block)
    local new_corefile
    new_corefile=$(echo "$corefile" | sed "/^    ready$/a\\
    hosts {\\
       $host_ip host.minikube.internal\\
       fallthrough\\
    }")

    kubectl -n kube-system get configmap coredns -o json | \
        python3 -c "
import sys, json
cm = json.load(sys.stdin)
cm['data']['Corefile'] = '''$new_corefile'''
json.dump(cm, sys.stdout)
" | kubectl apply -f - >/dev/null

    # Restart CoreDNS to pick up changes
    kubectl -n kube-system rollout restart deployment coredns >/dev/null 2>&1
    kubectl -n kube-system wait --for=condition=available deployment/coredns --timeout=30s >/dev/null 2>&1
    echo "✓ CoreDNS patched: host.minikube.internal → $host_ip"
}

# --- Check AGL_KEY ---
if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Either:"
    echo "  1. export AGL_KEY=\$(openssl rand -hex 32)"
    echo "  2. Uncomment AGL_KEY in deploy/.env"
    exit 1
fi

# --- Determine AGL_LITE_URL ---
if [ "$CONTROLLER_ONLY" = true ]; then
    # Server runs externally — user must set AGL_LITE_URL in .env
    if [ -z "${AGL_LITE_URL:-}" ]; then
        echo "ERROR: --controller-only requires AGL_LITE_URL in deploy/.env"
        echo "  e.g., AGL_LITE_URL=http://host.minikube.internal:8080"
        exit 1
    fi
    echo "=== Mode: controller-only (server external at $AGL_LITE_URL) ==="
else
    # Server runs in-cluster — auto-set URL to cluster-internal service DNS
    AGL_LITE_URL="http://agl-lite.${NS}.svc:8080"
    echo "=== Mode: in-cluster (server at $AGL_LITE_URL) ==="
fi

echo "=== Deploying to namespace: $NS ==="

# 1. Namespace
echo "--- Creating namespace ---"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# 2. Secret (AGL_KEY — from env var, never written to disk by this script)
echo "--- Creating secret ---"
kubectl -n "$NS" create secret generic "${AGL_SECRET_NAME:-agl-lite-keys}" \
    --from-literal=AGL_KEY="$AGL_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. ConfigMap (from .env, excluding AGL_KEY and comments, with correct AGL_LITE_URL)
echo "--- Creating configmap ---"
(grep -v '^AGL_KEY=' "$ENV_FILE" | grep -v '^AGL_LITE_URL=' | grep -v '^#' | grep -v '^$'; \
 echo "AGL_LITE_URL=$AGL_LITE_URL") | \
    kubectl -n "$NS" create configmap agl-lite-config \
    --from-env-file=/dev/stdin \
    --dry-run=client -o yaml | kubectl apply -f -

# 4. RBAC
echo "--- Applying RBAC ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/rbac.yaml"

# 5. Deployments
if [ "$CONTROLLER_ONLY" = false ]; then
    echo "--- Deploying agl-lite server ---"
    kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/agl-lite/k8s.yaml"
fi

echo "--- Deploying controller ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/k8s.yaml"

# 6. Minikube connectivity fix (--controller-only mode only)
#    Pods need to resolve host.minikube.internal to reach the external server.
#    CoreDNS doesn't read the node's /etc/hosts, so we patch it.
if [ "$CONTROLLER_ONLY" = true ]; then
    ensure_minikube_host_dns
fi

# 7. Wait
if [ "$CONTROLLER_ONLY" = false ]; then
    echo "--- Waiting for pods ---"
    kubectl -n "$NS" wait --for=condition=available deployment/agl-lite --timeout=120s
    kubectl -n "$NS" wait --for=condition=available deployment/agl-controller --timeout=120s
else
    echo "--- Waiting for controller ---"
    kubectl -n "$NS" wait --for=condition=available deployment/agl-controller --timeout=120s
fi

echo ""
echo "=== agl-lite deployed to namespace: $NS ==="
kubectl -n "$NS" get pods
echo ""
if [ "$CONTROLLER_ONLY" = true ]; then
    echo "Server is external at: $AGL_LITE_URL"
    echo "Start it on the host with:"
    echo "  AGL_KEY=\$AGL_KEY uv run agl-lite serve --host 0.0.0.0 --port 8080 [--hooks ...]"
else
    echo "Server is in-cluster at: $AGL_LITE_URL"
    echo "To access from host (for rl_loop.py or debugging):"
    echo "  kubectl -n $NS port-forward svc/agl-lite 8080:8080"
    echo "  export AGL_LITE_URL=http://localhost:8080 AGL_KEY=\$AGL_KEY"
fi
