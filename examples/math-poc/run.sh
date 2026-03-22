#!/bin/bash
# Run the math-poc end-to-end on minikube.
# Usage: examples/math-poc/run.sh
#
# Prerequisites:
#   - minikube running
#   - deploy/.env configured
#   - AGL_KEY set in environment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- Load config ---
if [ ! -f deploy/.env ]; then
    echo "ERROR: deploy/.env not found. Run:"
    echo "  cp deploy/.env.example deploy/.env"
    echo "  # edit deploy/.env"
    exit 1
fi
source deploy/.env
NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set}"

if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set."
    exit 1
fi

# --- Build images ---
echo "=== Building images ==="
scripts/build_images.sh --math-poc

# --- Deploy infra ---
echo ""
echo "=== Deploying agl-lite infra ==="
scripts/deploy.sh

# --- Deploy mockai ---
echo ""
echo "=== Deploying mockai ==="
kubectl apply -n "$NS" -f examples/math-poc/k8s-mockai.yaml
kubectl -n "$NS" wait --for=condition=available deployment/mockai --timeout=120s

# --- Port forward ---
echo ""
echo "=== Starting port-forward ==="
kubectl -n "$NS" port-forward svc/agl-lite 8080:8080 &
PF_PID=$!
sleep 2

# Cleanup on exit
cleanup() {
    echo ""
    echo "=== Cleanup ==="
    kill $PF_PID 2>/dev/null || true
    echo "Port-forward stopped. Cluster resources left running."
    echo "To tear down: scripts/deploy.sh --cleanup"
}
trap cleanup EXIT

# --- Run algorithm ---
echo ""
echo "=== Running mock RL loop ==="
export AGL_LITE_URL=http://localhost:8080
export AGL_K8S_NAMESPACE="$NS"
export AGL_KEY

uv run python examples/math-poc/mock_rl_loop.py
