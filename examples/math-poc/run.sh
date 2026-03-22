#!/bin/bash
# Run the math-poc end-to-end on minikube.
#
# Usage:
#   cp examples/math-poc/.env.mockai.example deploy/.env   # or .env.vllm.example
#   export AGL_KEY=$(openssl rand -hex 32)
#   examples/math-poc/run.sh
#
# Prerequisites:
#   - minikube running
#   - deploy/.env configured (copy from examples/math-poc/.env.*.example)
#   - AGL_KEY set in environment
#   - For vLLM mode: vLLM serving on host (see .env.vllm.example)
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
    echo "ERROR: deploy/.env not found. Run one of:"
    echo "  cp examples/math-poc/.env.mockai.example deploy/.env"
    echo "  cp examples/math-poc/.env.vllm.example deploy/.env"
    exit 1
fi
source deploy/.env
NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set}"
MODE="${AGL_MODEL_MODE:-mock}"

if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Run: export AGL_KEY=\$(openssl rand -hex 32)"
    exit 1
fi

echo "=== Mode: $MODE ==="

# --- For vLLM mode, verify vLLM is reachable ---
if [ "$MODE" = "vllm" ]; then
    VLLM_PORT="${AGL_VLLM_PORT:-8001}"
    echo "Checking vLLM on localhost:$VLLM_PORT ..."
    if ! curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "ERROR: vLLM not reachable at localhost:$VLLM_PORT"
        echo "Start it with: vllm serve ${AGL_MODEL_NAME:-model} --port $VLLM_PORT --host 0.0.0.0"
        exit 1
    fi
    echo "  vLLM OK"
fi

# --- Build images ---
echo "=== Building images ==="
scripts/build_images.sh --math-poc 2>&1 | tee "$LOG_DIR/build.log"

# --- Deploy infra ---
echo ""
echo "=== Deploying agl-lite infra ==="
scripts/deploy.sh 2>&1 | tee "$LOG_DIR/deploy.log"

# --- Apply gateway config (vllm mode) ---
if [ "$MODE" = "vllm" ]; then
    GATEWAY_CONFIG="$SCRIPT_DIR/gateway-config.yaml"
    if [ -f "$GATEWAY_CONFIG" ]; then
        echo ""
        echo "=== Applying gateway config ==="
        kubectl -n "$NS" create configmap agl-gateway-config \
            --from-file=gateway-config.yaml="$GATEWAY_CONFIG" \
            --dry-run=client -o yaml | kubectl apply -f - 2>&1 | tee -a "$LOG_DIR/deploy.log"
        # Patch agl-lite deployment to mount gateway config and add --gateway-config flag
        kubectl -n "$NS" patch deployment agl-lite --type=json -p='[
          {"op": "add", "path": "/spec/template/spec/volumes", "value": [
            {"name": "gateway-config", "configMap": {"name": "agl-gateway-config"}}
          ]},
          {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts", "value": [
            {"name": "gateway-config", "mountPath": "/etc/agl-lite", "readOnly": true}
          ]},
          {"op": "replace", "path": "/spec/template/spec/containers/0/command", "value": [
            "agl-lite", "serve", "--host", "0.0.0.0", "--port", "8080",
            "--gateway-config", "/etc/agl-lite/gateway-config.yaml"
          ]}
        ]' 2>&1 | tee -a "$LOG_DIR/deploy.log"
        kubectl -n "$NS" rollout status deployment/agl-lite --timeout=60s 2>&1 | tee -a "$LOG_DIR/deploy.log"
    fi
fi

# --- Deploy mockai (mock mode only) ---
if [ "$MODE" = "mock" ]; then
    echo ""
    echo "=== Deploying mockai ==="
    kubectl apply -n "$NS" -f examples/math-poc/k8s-mockai.yaml 2>&1 | tee -a "$LOG_DIR/deploy.log"
    kubectl -n "$NS" wait --for=condition=available deployment/mockai --timeout=120s 2>&1 | tee -a "$LOG_DIR/deploy.log"
fi

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
    if [ "$MODE" = "mock" ]; then
        kubectl -n "$NS" logs deployment/mockai --tail=200 > "$LOG_DIR/mockai.log" 2>&1 || true
    fi
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

# --- Export env for algorithm script ---
export AGL_LITE_URL=http://localhost:8080
export AGL_K8S_NAMESPACE="$NS"
export AGL_KEY
export AGL_MODEL_MODE="$MODE"
export AGL_MODEL_NAME="${AGL_MODEL_NAME:-mock-llm}"
export AGL_MODEL_ENDPOINT="${AGL_MODEL_ENDPOINT:-}"
export AGL_BATCH_SIZE="${AGL_BATCH_SIZE:-5}"
export AGL_NUM_ITERATIONS="${AGL_NUM_ITERATIONS:-2}"
export AGL_VLLM_PORT="${AGL_VLLM_PORT:-8001}"

# --- Run algorithm ---
echo ""
if [ "$MODE" = "mock" ]; then
    echo "=== Running mock RL loop ==="
    uv run python examples/math-poc/mock_rl_loop.py 2>&1 | tee "$LOG_DIR/mock_rl_loop.log"
else
    echo "=== Running RL loop (vLLM) ==="
    uv run python examples/math-poc/rl_loop.py 2>&1 | tee "$LOG_DIR/rl_loop.log"
fi
