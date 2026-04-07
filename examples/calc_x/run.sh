#!/bin/bash
# Run the Calc-X VERL training end-to-end.
#
# Usage:
#   examples/calc_x/run.sh                     # full training
#   examples/calc_x/run.sh --ci-fast           # single PPO step
#
# Topology (agl-in-host mode):
#   Host:     agl-lite serve + vLLM inference
#   Minikube: controller + agent pods (AutoGen + MCP calculator)
#
# Prerequisites:
#   - minikube running
#   - AGL_KEY set in environment
#   - vLLM serving on host (scripts/start_vllm.sh)
#   - Calc-X dataset downloaded to examples/calc_x/data/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE_DIR="$SCRIPT_DIR/vllm"

cd "$REPO_ROOT"

# --- Setup log directory ---
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
export AGL_LOG_DIR="$LOG_DIR"
echo "=== Logs → $LOG_DIR ==="

# --- Load config ---
if [ ! -f "$MODE_DIR/.env.example" ]; then
    echo "ERROR: $MODE_DIR/.env.example not found"
    exit 1
fi
source "$MODE_DIR/.env.example"
DEPLOY_CONFIG="$MODE_DIR/.env.example"

# --- Validate ---
if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Run: export AGL_KEY=\$(openssl rand -hex 32)"
    exit 1
fi

VLLM_PORT="${AGL_VLLM_PORT:-8010}"
echo "=== Checking vLLM on localhost:$VLLM_PORT ==="
if ! curl -sf "http://localhost:$VLLM_PORT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM not reachable at localhost:$VLLM_PORT"
    echo "Start it with: scripts/start_vllm.sh"
    exit 1
fi
echo "  vLLM OK"

# --- Build images ---
echo ""
echo "=== Building images ==="
scripts/build_images.sh --include-example calc-x

# --- Deploy K8s infra + start host agl-lite ---
NS="$(grep -E '^AGL_NAMESPACE=' "$DEPLOY_CONFIG" | cut -d= -f2 | tr -d '[:space:]')"
echo ""
# Clean up previous deployment if any — avoids stale Jobs/pods from prior runs.
if kubectl get namespace "$NS" > /dev/null 2>&1; then
    echo "=== Cleaning up previous deployment in namespace: $NS ==="
    uv run agl-lite deploy --env-file "$DEPLOY_CONFIG" --cleanup
fi
echo "=== Deploying (agl-in-host) ==="
uv run agl-lite deploy --env-file "$DEPLOY_CONFIG"

# --- Wait for agl-lite ---
echo ""
echo "=== Waiting for agl-lite ==="
READY=false
for i in $(seq 1 40); do
    if curl -sf http://localhost:8080/healthz > /dev/null 2>&1; then
        echo "  agl-lite ready"
        READY=true
        break
    fi
    sleep 1
done
if [ "$READY" != true ]; then
    echo "ERROR: agl-lite did not become ready"
    exit 1
fi

# --- Export env for training script ---
export AGL_BASE_URL=http://localhost:8080
export AGL_KEY

# --- Run training ---
echo ""
echo "=== Running VERL training ==="
exec uv run python examples/calc_x/train_calc_agent.py \
    --train-file "${AGL_TRAIN_FILE:-examples/calc_x/data/train.parquet}" \
    --val-file "${AGL_VAL_FILE:-examples/calc_x/data/test.parquet}" \
    "$@"
