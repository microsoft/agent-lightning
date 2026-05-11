#!/bin/bash
# SWE-bench example — one-command E2E runner.
#
# Usage:
#   examples/swe_bench/run.sh
#
# Prerequisites:
#   - K8s cluster running (minikube for local dev)
#   - SWE-bench Docker images pre-built for the sample instances
#   - AGL_KEY exported
#   - vLLM running on host with a code-capable model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_CONFIG="$SCRIPT_DIR/.env.example"
cd "$REPO_ROOT"

# --- Load config ---
if [ ! -f "$DEPLOY_CONFIG" ]; then
    echo "ERROR: $DEPLOY_CONFIG not found."
    exit 1
fi
source "$DEPLOY_CONFIG"
CONFIG_NAMESPACE="${AGL_NAMESPACE:?AGL_NAMESPACE not set}"

STATE_ENV="${AGL_LOCAL_STATE_DIR:-.local}/agl-lite.env"
if [ -z "${AGL_KEY:-}" ] && [ -f "$STATE_ENV" ]; then
    echo "=== Loading AGL_KEY from $STATE_ENV ==="
    source "$STATE_ENV"
fi

NS="$CONFIG_NAMESPACE"

if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set."
    echo "  export AGL_KEY=\$(openssl rand -hex 32)"
    exit 1
fi

export AGL_MODEL_NAME="${AGL_MODEL_NAME:-}"
export AGL_CODING_AGENT="${AGL_CODING_AGENT:-claude_code}"
export AGL_SWEBENCH_IMAGE_NAMESPACE="${AGL_SWEBENCH_IMAGE_NAMESPACE:-swebench}"

LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
export AGL_LOG_DIR="$LOG_DIR"

echo "=== SWE-bench Example ==="
echo "  Namespace: $NS"
echo "  Model: ${AGL_MODEL_NAME:-not set}"
echo "  Agent: $AGL_CODING_AGENT"
echo "  Logs: $LOG_DIR"

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

# --- Check SWE-bench images before launching long-running rollouts ---
echo ""
echo "--- Checking SWE-bench images ---"
REQUIRED_IMAGES=$(AGL_BATCH_SIZE="${AGL_BATCH_SIZE:-5}" \
  AGL_NUM_ITERATIONS="${AGL_NUM_ITERATIONS:-1}" \
  AGL_SWEBENCH_IMAGE_NAMESPACE="$AGL_SWEBENCH_IMAGE_NAMESPACE" \
  .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

from swebench.harness.test_spec.test_spec import make_test_spec

dataset_path = Path("examples/swe_bench/swebench_samples.jsonl")
batch_size = int(os.environ.get("AGL_BATCH_SIZE", "5"))
num_iterations = int(os.environ.get("AGL_NUM_ITERATIONS", "1"))
namespace = os.environ.get("AGL_SWEBENCH_IMAGE_NAMESPACE", "swebench")
needed = batch_size * num_iterations
items = [json.loads(line) for line in dataset_path.open()]
images = []
for index in range(needed):
    item = items[index % len(items)]
    image = make_test_spec(item, namespace=namespace).instance_image_key
    if image not in images:
        images.append(image)
for image in images:
    print(image)
PY
)

MISSING_IMAGES=()
if command -v minikube &>/dev/null && minikube status --format='{{.Host}}' 2>/dev/null | grep -q Running; then
    for image in $REQUIRED_IMAGES; do
        if ! minikube ssh -- docker image inspect "$image" >/dev/null 2>&1; then
            MISSING_IMAGES+=("$image")
        fi
    done
else
    for image in $REQUIRED_IMAGES; do
        if ! docker image inspect "$image" >/dev/null 2>&1; then
            MISSING_IMAGES+=("$image")
        fi
    done
fi

if [ ${#MISSING_IMAGES[@]} -gt 0 ]; then
    echo "ERROR: Missing SWE-bench evaluation images:"
    printf '  %s\n' "${MISSING_IMAGES[@]}"
    echo ""
    echo "Build them for minikube with:"
    echo '  eval "$(minikube -p minikube docker-env)"'
    echo "  .venv/bin/python examples/swe_bench/build_images.py --limit ${AGL_BATCH_SIZE:-1}"
    echo '  eval "$(minikube -p minikube docker-env -u)"'
    exit 1
fi
echo "  SWE-bench images available"

# --- Mount artifact directory (minikube only) ---
# hostPath inside minikube's VM is not on the host machine.
# minikube mount creates a bidirectional mount so artifacts are accessible.
ARTIFACT_HOST_DIR="${AGL_ARTIFACT_DIR:-$REPO_ROOT/artifacts}"
ARTIFACT_VM_DIR="/data/agl-artifacts"
mkdir -p "$ARTIFACT_HOST_DIR"

if command -v minikube &>/dev/null && minikube status --format='{{.Host}}' 2>/dev/null | grep -q Running; then
    echo ""
    echo "--- Mounting artifact directory ---"
    # Kill any existing mounts.
    pkill -f "minikube mount.*${ARTIFACT_VM_DIR}" 2>/dev/null || true
    sleep 1

    # Artifact directory — host ↔ VM.
    minikube mount "${ARTIFACT_HOST_DIR}:${ARTIFACT_VM_DIR}" &>/dev/null &
    MOUNT_PID=$!
    sleep 2
    if kill -0 $MOUNT_PID 2>/dev/null; then
        echo "  Mounted: $ARTIFACT_HOST_DIR → $ARTIFACT_VM_DIR (PID: $MOUNT_PID)"
    else
        echo "  WARNING: minikube mount failed — artifacts only accessible via 'minikube ssh'"
        MOUNT_PID=""
    fi
else
    MOUNT_PID=""
fi

# Cleanup local minikube mount on exit. agl-lite itself is managed by
# `agl-lite deploy`; use the printed teardown command when you want to stop it.
cleanup() {
    if [ -n "${MOUNT_PID:-}" ]; then
        kill $MOUNT_PID 2>/dev/null || true
        echo "  Artifact mount stopped"
    fi
    echo "  Logs: $LOG_DIR"
    echo "  Artifacts: $ARTIFACT_HOST_DIR"
    echo "  To tear down: uv run agl-lite deploy --env-file $DEPLOY_CONFIG --cleanup"
}
trap cleanup EXIT

# --- Deploy infrastructure ---
echo ""
echo "--- Deploying infrastructure ---"
scripts/build_images.sh
if kubectl get namespace "$NS" >/dev/null 2>&1; then
    echo "--- Cleaning up previous deployment in namespace: $NS ---"
    uv run agl-lite deploy --env-file "$DEPLOY_CONFIG" --cleanup
fi
uv run agl-lite deploy --env-file "$DEPLOY_CONFIG"

# --- Wait for agl-lite ---
echo ""
AGL_HOST_URL="http://localhost:${AGL_HOST_PORT:-8080}"
echo "--- Waiting for agl-lite at $AGL_HOST_URL ---"
READY=false
for i in $(seq 1 40); do
    if curl -sf "$AGL_HOST_URL/healthz" > /dev/null 2>&1; then
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

if [ -f "$STATE_ENV" ]; then
    source "$STATE_ENV"
fi

export AGL_BASE_URL=${AGL_BASE_URL:-$AGL_HOST_URL}
export AGL_KEY
export AGL_NAMESPACE="$NS"
export AGL_MODEL_NAME
export AGL_MODEL_ENDPOINT="${AGL_MODEL_ENDPOINT:-}"
export AGL_BATCH_SIZE="${AGL_BATCH_SIZE:-5}"
export AGL_NUM_ITERATIONS="${AGL_NUM_ITERATIONS:-1}"
export AGL_CODING_AGENT
export AGL_TIMEOUT="${AGL_TIMEOUT:-5400}"
export AGL_POLL_INTERVAL_SEC="${AGL_POLL_INTERVAL_SEC:-${AGL_POLL_INTERVAL:-10}}"

# --- Create ConfigMap for agent scripts ---
echo ""
echo "--- Creating agent scripts ConfigMap ---"

# kubectl create configmap doesn't support nested dirs directly,
# so we specify files with path-based keys.
kubectl -n "$NS" delete configmap swe-agent-scripts --ignore-not-found
kubectl -n "$NS" create configmap swe-agent-scripts \
    --from-file=entrypoint.sh="$SCRIPT_DIR/agents/entrypoint.sh" \
    --from-file=grade.py="$SCRIPT_DIR/agents/grade.py" \
    --from-file=claude_code--install.sh="$SCRIPT_DIR/agents/claude_code/install.sh" \
    --from-file=claude_code--run.sh="$SCRIPT_DIR/agents/claude_code/run.sh" \
    --from-file=claude_code--CLAUDE.md="$SCRIPT_DIR/agents/claude_code/CLAUDE.md" \
    --from-file=claude_code--handle_hook.sh="$SCRIPT_DIR/agents/claude_code/handle_hook.sh"
echo "  ConfigMap swe-agent-scripts created"

# --- Run the RL loop ---
echo ""
echo "--- Running SWE-bench RL loop ---"
.venv/bin/python "$SCRIPT_DIR/rl_loop.py"

EXIT_CODE=$?

exit $EXIT_CODE
