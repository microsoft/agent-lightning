#!/bin/bash
# Run the math-poc end-to-end.
#
# Usage:
#   examples/math-poc/run.sh [mock|vllm]   # default: vllm
#
# Topology:
#   mock mode: all services in minikube (agl-lite, controller, mockai, agents)
#   vllm mode: agl-lite + vLLM on host, controller + agents in minikube
#
# Prerequisites:
#   - minikube running
#   - AGL_KEY set in environment
#   - For vLLM mode: vLLM serving on host (scripts/start_vllm.sh)
#
# Logs are saved to examples/math-poc/logs/<timestamp>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE="${1:-vllm}"
MODE_DIR="$SCRIPT_DIR/$MODE"

cd "$REPO_ROOT"

# --- Validate mode ---
if [ ! -d "$MODE_DIR" ]; then
    echo "ERROR: Unknown mode '$MODE'. Available: mock, vllm"
    exit 1
fi

# --- Setup log directory ---
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)-$MODE"
mkdir -p "$LOG_DIR"
echo "=== Logs → $LOG_DIR ==="

# --- Load config ---
if [ ! -f "$MODE_DIR/.env.example" ]; then
    echo "ERROR: $MODE_DIR/.env.example not found"
    exit 1
fi
source "$MODE_DIR/.env.example"
NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set}"

if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Run: export AGL_KEY=\$(openssl rand -hex 32)"
    exit 1
fi

echo "=== Mode: $MODE ==="

# --- For vLLM mode, verify vLLM is reachable ---
if [ "$MODE" = "vllm" ]; then
    VLLM_PORT="${AGL_VLLM_PORT:-8010}"
    echo "Checking vLLM on localhost:$VLLM_PORT ..."
    if ! curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
        echo "ERROR: vLLM not reachable at localhost:$VLLM_PORT"
        echo "Start it with: scripts/start_vllm.sh"
        exit 1
    fi
    echo "  vLLM OK"
fi

# --- Build images ---
echo "=== Building images ==="
scripts/build_images.sh --math-poc 2>&1 | tee "$LOG_DIR/build.log"

# --- Deploy K8s infra ---
echo ""
if [ "$MODE" = "vllm" ]; then
    echo "=== Deploying controller to K8s (agl-lite will run on host) ==="
    scripts/deploy.sh --no-serve 2>&1 | tee "$LOG_DIR/deploy.log"
else
    echo "=== Deploying agl-lite infra to K8s ==="
    scripts/deploy.sh 2>&1 | tee "$LOG_DIR/deploy.log"
fi

# --- Deploy mockai (mock mode only) ---
if [ "$MODE" = "mock" ]; then
    echo ""
    echo "=== Deploying mockai ==="
    kubectl apply -n "$NS" -f "$MODE_DIR/k8s-mockai.yaml" 2>&1 | tee -a "$LOG_DIR/deploy.log"
    kubectl -n "$NS" wait --for=condition=available deployment/mockai --timeout=120s 2>&1 | tee -a "$LOG_DIR/deploy.log"
fi

# --- Start agl-lite serve ---
SERVE_PID=""
PF_PID=""

GATEWAY_CONFIG="$MODE_DIR/gateway-config.yaml"
HOOKS_PATH="${AGL_HOOKS:-}"

if [ "$MODE" = "vllm" ]; then
    SERVE_CMD=(uv run agl-lite serve --host 0.0.0.0 --port 8080)
    [ -f "$GATEWAY_CONFIG" ] && SERVE_CMD+=(--gateway-config "$GATEWAY_CONFIG")
    [ -n "$HOOKS_PATH" ] && SERVE_CMD+=(--hooks "$HOOKS_PATH")
    echo ""
    echo "=== Starting agl-lite serve on host ==="
    echo "  ${SERVE_CMD[*]}"
    AGL_KEY="$AGL_KEY" "${SERVE_CMD[@]}" > "$LOG_DIR/agl-lite.log" 2>&1 &
    SERVE_PID=$!

    for i in $(seq 1 30); do
        if curl -sf http://localhost:8080/healthz > /dev/null 2>&1; then
            echo "  agl-lite ready"
            break
        fi
        if ! kill -0 $SERVE_PID 2>/dev/null; then
            echo "ERROR: agl-lite process died"
            tail -20 "$LOG_DIR/agl-lite.log"
            exit 1
        fi
        sleep 1
    done
else
    echo ""
    echo "=== Starting port-forward ==="
    kubectl -n "$NS" port-forward svc/agl-lite 8080:8080 &
    PF_PID=$!
    sleep 2
fi

# Cleanup on exit
cleanup() {
    echo ""
    echo "=== Collecting K8s logs ==="
    if [ "$MODE" = "mock" ]; then
        kubectl -n "$NS" logs deployment/agl-lite --tail=200 > "$LOG_DIR/agl-lite.log" 2>&1 || true
        kubectl -n "$NS" logs deployment/mockai --tail=200 > "$LOG_DIR/mockai.log" 2>&1 || true
    fi
    kubectl -n "$NS" logs deployment/agl-controller --tail=200 > "$LOG_DIR/controller.log" 2>&1 || true
    for pod in $(kubectl -n "$NS" get pods -l managed-by=agl-controller -o name 2>/dev/null); do
        name=$(basename "$pod")
        kubectl -n "$NS" logs "$pod" --all-containers > "$LOG_DIR/agent-$name.log" 2>&1 || true
    done
    kubectl -n "$NS" get pods -o wide > "$LOG_DIR/pods.log" 2>&1 || true
    kubectl -n "$NS" get jobs -o wide > "$LOG_DIR/jobs.log" 2>&1 || true
    echo "=== Cleanup ==="
    [ -n "$PF_PID" ] && kill $PF_PID 2>/dev/null || true
    [ -n "$SERVE_PID" ] && kill $SERVE_PID 2>/dev/null || true
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
export AGL_NUM_ITERATIONS="${AGL_NUM_ITERATIONS:-1}"

# --- Run algorithm ---
echo ""
echo "=== Running RL loop ($MODE) ==="
uv run python examples/math-poc/rl_loop_v2.py 2>&1 | tee "$LOG_DIR/rl_loop.log"
