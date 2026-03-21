#!/bin/bash
# Deploy agl-lite infrastructure to K8s.
# Usage: scripts/deploy.sh [--teardown]
#
# Reads: deploy/.env (secrets + bootstrap), deploy/config.yaml (non-secret config)
# Creates: namespace, secret, configmap, RBAC, agl-lite serve, controller
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPLOY_DIR="$REPO_ROOT/deploy"

# --- Load .env ---
if [ ! -f "$DEPLOY_DIR/.env" ]; then
    echo "ERROR: deploy/.env not found. Copy from deploy/.env.example and edit."
    exit 1
fi
source "$DEPLOY_DIR/.env"

NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set in .env}"
AGL_KEY="${AGL_KEY:?AGL_KEY not set in .env}"

# --- Teardown mode ---
if [[ "${1:-}" == "--teardown" ]]; then
    echo "=== Tearing down namespace $NS ==="
    kubectl delete namespace "$NS" --ignore-not-found --wait
    echo "Done."
    exit 0
fi

# --- Check config.yaml ---
if [ ! -f "$DEPLOY_DIR/config.yaml" ]; then
    echo "ERROR: deploy/config.yaml not found. Copy from deploy/config.example.yaml and edit."
    exit 1
fi

echo "=== Deploying agl-lite to namespace: $NS ==="

# 1. Namespace
echo "--- Creating namespace ---"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# 2. Secret (AGL_KEY — never written to disk, passed via --from-literal)
echo "--- Creating secret ---"
kubectl -n "$NS" create secret generic agl-lite-keys \
    --from-literal=AGL_KEY="$AGL_KEY" \
    --dry-run=client -o yaml | kubectl apply -f -

# 3. ConfigMap (config.yaml as file + extracted values as literal keys)
#    The YAML file is mounted into pods. Individual keys are used as env vars by controller.
echo "--- Creating configmap ---"
AGL_LITE_URL=$(grep '^agl_lite_url:' "$DEPLOY_DIR/config.yaml" | awk '{print $2}')
SECRET_NAME=$(grep '^secret_name:' "$DEPLOY_DIR/config.yaml" | awk '{print $2}')
kubectl -n "$NS" create configmap agl-lite-config \
    --from-file=config.yaml="$DEPLOY_DIR/config.yaml" \
    --from-literal=AGL_K8S_NAMESPACE="$NS" \
    --from-literal=AGL_LITE_URL="${AGL_LITE_URL:?agl_lite_url not set in config.yaml}" \
    --from-literal=AGL_SECRET_NAME="${SECRET_NAME:-agl-lite-keys}" \
    --dry-run=client -o yaml | kubectl apply -f -

# 4. RBAC
echo "--- Applying RBAC ---"
kubectl apply -n "$NS" -f "$DEPLOY_DIR/controller/rbac.yaml"

# 5. Deployments
echo "--- Deploying agl-lite serve ---"
kubectl apply -n "$NS" -f "$DEPLOY_DIR/agl-lite/k8s.yaml"

echo "--- Deploying controller ---"
kubectl apply -n "$NS" -f "$DEPLOY_DIR/controller/k8s.yaml"

# 6. Wait for ready
echo "--- Waiting for pods ---"
kubectl -n "$NS" wait --for=condition=available deployment/agl-lite --timeout=120s
kubectl -n "$NS" wait --for=condition=available deployment/agl-controller --timeout=120s

echo ""
echo "=== agl-lite deployed to namespace: $NS ==="
kubectl -n "$NS" get pods
echo ""
echo "To access from host:"
echo "  kubectl -n $NS port-forward svc/agl-lite 8080:8080"
echo "  export AGL_LITE_URL=http://localhost:8080 AGL_KEY=$AGL_KEY"
echo "  agl-client health"
