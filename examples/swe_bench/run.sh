#!/bin/bash
# SWE-bench example — one-command E2E runner.
#
# Usage:
#   examples/swe_bench/run.sh
#
# Prerequisites:
#   - K8s cluster running (minikube for local dev)
#   - SWE-bench Docker images pre-built for the sample instances
#   - deploy/.env configured (copy from examples/swe_bench/.env.example)
#   - AGL_KEY exported
#   - vLLM running on host with a code-capable model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="$REPO_ROOT/deploy/.env"

# --- Load config ---
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: deploy/.env not found."
    echo "  cp examples/swe_bench/.env.example deploy/.env"
    exit 1
fi
source "$ENV_FILE"

NS="${AGL_K8S_NAMESPACE:?AGL_K8S_NAMESPACE not set}"

if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set."
    echo "  export AGL_KEY=\$(openssl rand -hex 32)"
    exit 1
fi

export AGL_MODEL_NAME="${AGL_MODEL_NAME:-}"
export AGL_CODING_AGENT="${AGL_CODING_AGENT:-claude_code}"

echo "=== SWE-bench Example ==="
echo "  Namespace: $NS"
echo "  Model: ${AGL_MODEL_NAME:-not set}"
echo "  Agent: $AGL_CODING_AGENT"

# --- Check vLLM availability ---
if [ -n "${AGL_MODEL_ENDPOINT:-}" ]; then
    # Extract host:port from endpoint URL (e.g., http://localhost:8010/v1 → localhost:8010)
    VLLM_HOST_PORT=$(echo "$AGL_MODEL_ENDPOINT" | sed 's|https\?://||' | sed 's|/.*||')
    echo ""
    echo "--- Checking model server ---"
    echo "  Endpoint: $AGL_MODEL_ENDPOINT"

    if ! curl -sf "http://${VLLM_HOST_PORT}/v1/models" > /dev/null 2>&1; then
        echo "ERROR: Model server not reachable at $VLLM_HOST_PORT"
        echo "  Start vLLM: scripts/start_vllm.sh"
        exit 1
    fi

    # Check that the expected model is served
    if [ -n "$AGL_MODEL_NAME" ]; then
        SERVED_MODELS=$(curl -sf "http://${VLLM_HOST_PORT}/v1/models" | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(' '.join(m['id'] for m in data.get('data', [])))
" 2>/dev/null || echo "")
        if echo "$SERVED_MODELS" | grep -qF "$AGL_MODEL_NAME"; then
            echo "  ✓ Model '$AGL_MODEL_NAME' available"
        else
            echo "  WARNING: Model '$AGL_MODEL_NAME' not found in served models: $SERVED_MODELS"
            echo "  The gateway will route requests to '$AGL_MODEL_NAME' but vLLM may reject them."
        fi
    fi
    echo "  ✓ Model server reachable"
fi

# --- Create ConfigMap for agent scripts ---
echo ""
echo "--- Creating agent scripts ConfigMap ---"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -

# kubectl create configmap doesn't support nested dirs directly,
# so we specify files with path-based keys.
kubectl -n "$NS" delete configmap swe-agent-scripts --ignore-not-found
kubectl -n "$NS" create configmap swe-agent-scripts \
    --from-file=entrypoint.sh="$SCRIPT_DIR/agents/entrypoint.sh" \
    --from-file=claude_code/install.sh="$SCRIPT_DIR/agents/claude_code/install.sh" \
    --from-file=claude_code/run.sh="$SCRIPT_DIR/agents/claude_code/run.sh" \
    --from-file=claude_code/CLAUDE.md="$SCRIPT_DIR/agents/claude_code/CLAUDE.md" \
    --from-file=claude_code/handle_hook.sh="$SCRIPT_DIR/agents/claude_code/handle_hook.sh"
echo "  ConfigMap swe-agent-scripts created"

# --- Deploy infrastructure ---
echo ""
echo "--- Deploying infrastructure ---"
"$REPO_ROOT/scripts/deploy.sh" --controller-only

# --- Start agl-lite server on host ---
echo ""
echo "--- Starting agl-lite server ---"

AGL_KEY="$AGL_KEY" \
AGL_MODEL_NAME="$AGL_MODEL_NAME" \
AGL_CODING_AGENT="$AGL_CODING_AGENT" \
  uv run agl-lite serve \
    --host 0.0.0.0 --port 8080 \
    --gateway-config "$SCRIPT_DIR/gateway-config.yaml" \
    --hooks "$SCRIPT_DIR/hooks.py" \
    --artifact-dir "/tmp/agl-artifacts" \
    > /tmp/agl-lite-swebench.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready.
for i in $(seq 1 15); do
    if curl -sf http://localhost:8080/healthz > /dev/null 2>&1; then
        echo "  agl-lite ready (PID: $SERVER_PID)"
        break
    fi
    sleep 1
done

if ! curl -sf http://localhost:8080/healthz > /dev/null 2>&1; then
    echo "ERROR: agl-lite server failed to start"
    cat /tmp/agl-lite-swebench.log
    exit 1
fi

# --- Run the RL loop ---
echo ""
echo "--- Running SWE-bench RL loop ---"
export AGL_LITE_URL="http://localhost:8080"

AGL_KEY="$AGL_KEY" \
AGL_LITE_URL="$AGL_LITE_URL" \
AGL_MODEL_NAME="$AGL_MODEL_NAME" \
AGL_MODEL_ENDPOINT="${AGL_MODEL_ENDPOINT:-}" \
AGL_BATCH_SIZE="${AGL_BATCH_SIZE:-5}" \
AGL_NUM_ITERATIONS="${AGL_NUM_ITERATIONS:-1}" \
  uv run python "$SCRIPT_DIR/rl_loop.py"

EXIT_CODE=$?

# --- Cleanup ---
echo ""
echo "--- Cleanup ---"
kill $SERVER_PID 2>/dev/null || true
echo "  Server stopped"

exit $EXIT_CODE
