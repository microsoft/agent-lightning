#!/bin/bash
# Run the math-poc end-to-end on minikube.
# Usage: examples/math-poc/run.sh
#
# Prerequisites:
#   - minikube running
#   - deploy/.env configured
#   - AGL_KEY set in environment
#
# Logs are saved to examples/math-poc/logs/<timestamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# --- Setup log directory ---
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
echo "=== Logs → $LOG_DIR ==="

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
scripts/build_images.sh --math-poc 2>&1 | tee "$LOG_DIR/build.log"

# --- Deploy infra ---
echo ""
echo "=== Deploying agl-lite infra ==="
scripts/deploy.sh 2>&1 | tee "$LOG_DIR/deploy.log"

# --- Deploy mockai ---
echo ""
echo "=== Deploying mockai ==="
kubectl apply -n "$NS" -f examples/math-poc/k8s-mockai.yaml 2>&1 | tee -a "$LOG_DIR/deploy.log"
kubectl -n "$NS" wait --for=condition=available deployment/mockai --timeout=120s 2>&1 | tee -a "$LOG_DIR/deploy.log"

# --- Port forward ---
echo ""
echo "=== Starting port-forward ==="
kubectl -n "$NS" port-forward svc/agl-lite 8080:8080 &
PF_PID=$!
sleep 2

# Cleanup on exit — collect K8s logs
cleanup() {
    echo ""
    echo "=== Collecting K8s logs ==="
    kubectl -n "$NS" logs deployment/agl-lite --tail=200 > "$LOG_DIR/agl-lite.log" 2>&1 || true
    kubectl -n "$NS" logs deployment/agl-controller --tail=200 > "$LOG_DIR/controller.log" 2>&1 || true
    kubectl -n "$NS" logs deployment/mockai --tail=200 > "$LOG_DIR/mockai.log" 2>&1 || true
    # Collect logs from any agent pods (Jobs)
    for pod in $(kubectl -n "$NS" get pods -l managed-by=agl-controller -o name 2>/dev/null); do
        name=$(basename "$pod")
        kubectl -n "$NS" logs "$pod" --all-containers > "$LOG_DIR/agent-$name.log" 2>&1 || true
    done
    kubectl -n "$NS" get pods -o wide > "$LOG_DIR/pods.log" 2>&1 || true
    kubectl -n "$NS" get jobs -o wide > "$LOG_DIR/jobs.log" 2>&1 || true
    echo "=== Cleanup ==="
    kill $PF_PID 2>/dev/null || true
    echo "Port-forward stopped. Cluster resources left running."
    echo "Logs saved to: $LOG_DIR"
    echo "To tear down: scripts/deploy.sh --cleanup"
}
trap cleanup EXIT

# --- Run algorithm ---
echo ""
echo "=== Running mock RL loop ==="
export AGL_LITE_URL=http://localhost:8080
export AGL_K8S_NAMESPACE="$NS"
export AGL_KEY

uv run python examples/math-poc/mock_rl_loop.py 2>&1 | tee "$LOG_DIR/mock_rl_loop.log"
