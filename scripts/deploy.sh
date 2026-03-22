#!/bin/bash
# Deploy agl-lite infrastructure to K8s.
# Usage:
#   scripts/deploy.sh              # deploy
#   scripts/deploy.sh --cleanup    # remove everything
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
SKIP_SERVE=false
for arg in "$@"; do
    case "$arg" in
        --no-serve) SKIP_SERVE=true ;;
    esac
done

# --- Check AGL_KEY ---
if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Either:"
    echo "  1. export AGL_KEY=\$(openssl rand -hex 32)"
    echo "  2. Uncomment AGL_KEY in deploy/.env"
    exit 1
fi

echo "=== Deploying agl-lite to namespace: $NS ==="

# 1. Namespace
echo "--- Creating namespace ---"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# 2. Secret (AGL_KEY — from env var, never written to disk by this script)
echo "--- Creating secret ---"
kubectl -n "$NS" create secret generic "${AGL_SECRET_NAME:-agl-lite-keys}" \
    --from-literal=AGL_KEY="$AGL_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. ConfigMap (from .env, excluding AGL_KEY and comments)
echo "--- Creating configmap ---"
grep -v '^AGL_KEY=' "$ENV_FILE" | grep -v '^#' | grep -v '^$' | \
    kubectl -n "$NS" create configmap agl-lite-config \
    --from-env-file=/dev/stdin \
    --dry-run=client -o yaml | kubectl apply -f -

# 4. RBAC
echo "--- Applying RBAC ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/rbac.yaml"

# 5. Deployments
if [ "$SKIP_SERVE" = false ]; then
    echo "--- Deploying agl-lite serve ---"
    kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/agl-lite/k8s.yaml"
fi

echo "--- Deploying controller ---"
kubectl apply -n "$NS" -f "$REPO_ROOT/deploy/controller/k8s.yaml"

# 6. Wait
if [ "$SKIP_SERVE" = false ]; then
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
echo "To access from host:"
echo "  kubectl -n $NS port-forward svc/agl-lite 8080:8080"
echo "  export AGL_LITE_URL=http://localhost:8080 AGL_KEY=\$AGL_KEY"
echo "  agl-client health"
