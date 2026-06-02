#!/bin/bash
# Run the Calc-X VERL training end-to-end (async-rollout path).
#
# Usage:
#   examples/async_calc_x/run.sh                     # full training
#   examples/async_calc_x/run.sh --ci-fast           # single PPO step
#   examples/async_calc_x/run.sh --local --ci-fast   # local runner smoke test
#
# Topology:
#   K8s mode:   Host agl-lite serve + Minikube controller/agent pods
#   Local mode: Host agl-lite serve + local controller subprocess agents
#
# Prerequisites:
#   - minikube running
#   - AGL_KEY set in environment
#   - vLLM is managed internally by VERL
#   - Calc-X dataset downloaded to examples/async_calc_x/data/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODE_DIR="$SCRIPT_DIR"

cd "$REPO_ROOT"

RUNNER_MODE="k8s"
TRAIN_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --local)
            RUNNER_MODE="local"
            ;;
        --k8s)
            RUNNER_MODE="k8s"
            ;;
        *)
            TRAIN_ARGS+=("$arg")
            ;;
    esac
done

# --- Setup log directory ---
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR/agents"
export AGL_LOG_DIR="$LOG_DIR"
echo "=== Logs → $LOG_DIR ==="

# --- Mount agent log directory into minikube ---
# Agent pods write to hostPath /tmp/agl-lite/logs inside minikube VM.
# minikube mount bridges that to the host filesystem so logs survive pod deletion.
MOUNT_SRC="$LOG_DIR/agents"
MOUNT_DST="/tmp/agl-lite/logs"

if [ "$RUNNER_MODE" = "k8s" ]; then
    # Kill any stale mount to the same destination (e.g., from a previous run
    # that was killed without cleanup).
    STALE_PIDS=$(pgrep -f "minikube mount.*:$MOUNT_DST" || true)
    if [ -n "$STALE_PIDS" ]; then
        echo "=== Cleaning up stale minikube mount (PIDs: $STALE_PIDS) ==="
        kill $STALE_PIDS 2>/dev/null || true
        sleep 1
    fi

    echo "=== Mounting $MOUNT_SRC → minikube:$MOUNT_DST ==="
    minikube mount "$MOUNT_SRC:$MOUNT_DST" &
    MOUNT_PID=$!
    sleep 2
    if ! kill -0 $MOUNT_PID 2>/dev/null; then
        echo "ERROR: minikube mount failed"
        exit 1
    fi
    cleanup_mount() {
        local status=$?
        kill "$MOUNT_PID" 2>/dev/null || true
        wait "$MOUNT_PID" 2>/dev/null || true
        exit "$status"
    }
    trap cleanup_mount EXIT
fi

# --- Load config ---
USER_AGL_VAL_FILE_SET=false
if [ -n "${AGL_VAL_FILE+x}" ]; then
    USER_AGL_VAL_FILE_SET=true
    USER_AGL_VAL_FILE_VALUE="$AGL_VAL_FILE"
fi
USER_AGL_HOST_PORT_SET=false
if [ -n "${AGL_HOST_PORT+x}" ]; then
    USER_AGL_HOST_PORT_SET=true
    USER_AGL_HOST_PORT_VALUE="$AGL_HOST_PORT"
fi

if [ ! -f "$MODE_DIR/.env.example" ]; then
    echo "ERROR: $MODE_DIR/.env.example not found"
    exit 1
fi
set -a
source "$MODE_DIR/.env.example"
set +a
DEPLOY_CONFIG="$MODE_DIR/.env.example"

if [ "$USER_AGL_VAL_FILE_SET" = true ]; then
    export AGL_VAL_FILE="$USER_AGL_VAL_FILE_VALUE"
elif printf ' %s ' "${TRAIN_ARGS[@]}" | grep -q ' --ci-fast '; then
    export AGL_VAL_FILE="examples/async_calc_x/data/test_mini.parquet"
fi
if [ "$USER_AGL_HOST_PORT_SET" = true ]; then
    export AGL_HOST_PORT="$USER_AGL_HOST_PORT_VALUE"
fi

# --- Validate AGL_KEY ---
# AGL_KEY can come from: (1) environment, (2) state file from previous deploy.
STATE_ENV=".local/agl-lite.env"
if [ -z "${AGL_KEY:-}" ] && [ -f "$STATE_ENV" ]; then
    echo "=== Loading AGL_KEY from $STATE_ENV ==="
    source "$STATE_ENV"
fi
if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Either:"
    echo "  export AGL_KEY=\$(openssl rand -hex 32)"
    echo "  or run 'agl-lite deploy' first (stores key in $STATE_ENV)"
    exit 1
fi

# --- Validate AGL_ADMIN_KEY (required for async-rollout) ---
# AGL_ADMIN_KEY gates the /proxy/{pause,resume,state} surface used by the trainer to
# pause/resume the gateway during async rollouts. Must differ from AGL_KEY:
# agent pods carry AGL_KEY and must NOT be able to reach the admin surface.
if [ -z "${AGL_ADMIN_KEY:-}" ] && [ -f "$STATE_ENV" ]; then
    echo "=== Loading AGL_ADMIN_KEY from $STATE_ENV ==="
    source "$STATE_ENV"
fi
if [ -z "${AGL_ADMIN_KEY:-}" ]; then
    echo "=== Generating AGL_ADMIN_KEY ==="
    export AGL_ADMIN_KEY=$(openssl rand -hex 32)
fi
if [ "$AGL_ADMIN_KEY" = "$AGL_KEY" ]; then
    echo "ERROR: AGL_ADMIN_KEY must differ from AGL_KEY"
    exit 1
fi

# Note: vLLM is managed internally by VERL (hybrid mode), not started externally.
# The AglLiteAgentLoopManager registers VERL's vLLM server addresses with the
# agl-lite gateway at runtime.

if [ "$RUNNER_MODE" = "k8s" ]; then
    # --- Build images ---
    echo ""
    echo "=== Building images ==="
    scripts/build_images.sh --include-example async_calc_x

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

    # Source the state file generated by deploy (has AGL_KEY, AGL_BASE_URL).
    if [ -f "$STATE_ENV" ]; then
        source "$STATE_ENV"
    fi
    export AGL_BASE_URL=${AGL_BASE_URL:-http://localhost:${AGL_HOST_PORT:-8080}}
else
    export AGL_BASE_URL="http://localhost:${AGL_HOST_PORT:-8080}"
    SERVER_LOG="$LOG_DIR/server.log"
    echo ""
    echo "=== Starting agl-lite serve on :${AGL_HOST_PORT:-8080} (log: $SERVER_LOG) ==="
    AGL_GATEWAY_CONFIG="${AGL_GATEWAY_CONFIG:-examples/async_calc_x/gateway-config.yaml}" \
    AGL_HOOKS="${AGL_HOOKS:-examples/async_calc_x/hooks.py}" \
    AGL_LOG_DIR="$LOG_DIR" \
    uv run agl-lite-server \
        host="${AGL_HOST_IP_BIND:-0.0.0.0}" \
        port="${AGL_HOST_PORT:-8080}" \
        key="${AGL_KEY:-}" \
        > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!

    cleanup_local() {
        local status=$?
        set +e
        echo "=== Cleanup: stopping controller (PID ${CONTROLLER_PID:-?}) and server (PID ${SERVER_PID:-?}) ==="
        if [ -n "${CONTROLLER_PID:-}" ]; then
            kill -TERM "$CONTROLLER_PID" 2>/dev/null || true
        fi
        if [ -n "${SERVER_PID:-}" ]; then
            kill -TERM "$SERVER_PID" 2>/dev/null || true
        fi
        sleep 2
        if [ -n "${CONTROLLER_PID:-}" ]; then
            kill -KILL "$CONTROLLER_PID" 2>/dev/null || true
        fi
        if [ -n "${SERVER_PID:-}" ]; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        exit "$status"
    }
    trap cleanup_local EXIT
fi

# --- Wait for agl-lite ---
echo ""
echo "=== Waiting for agl-lite ==="
READY=false
for i in $(seq 1 40); do
    if [ "$RUNNER_MODE" = "local" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: agl-lite serve exited before becoming ready. See $SERVER_LOG."
        exit 1
    fi
    if curl -sf "$AGL_BASE_URL/healthz" > /dev/null 2>&1; then
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

if [ "$RUNNER_MODE" = "local" ]; then
    echo "=== Checking local agent dependencies ==="
    if ! .venv/bin/python - <<'PY'
import importlib.util
import sys

missing = [
    name
    for name in (
        "autogen_agentchat",
        "autogen_core",
        "autogen_ext",
        "mcp_server_calculator",
    )
    if importlib.util.find_spec(name) is None
]
if missing:
    print("Missing local Calc-X agent dependencies: " + ", ".join(missing), file=sys.stderr)
    print("Install them with: uv sync --extra verl", file=sys.stderr)
    sys.exit(1)
PY
    then
        exit 1
    fi

    CONTROLLER_LOG="$LOG_DIR/controller.log"
    echo "=== Starting agl-lite controller (local runner, pool=${AGL_LOCAL_POOL_SIZE:-8}, log: $CONTROLLER_LOG) ==="
    uv run agl-lite-controller \
        base_url="$AGL_BASE_URL" \
        namespace="${AGL_NAMESPACE:-local}" \
        runner_type=local \
        local_pool_size="${AGL_LOCAL_POOL_SIZE:-8}" \
        local_agent_class=examples.async_calc_x.agents.calc_agent:CalcXAgent \
        > "$CONTROLLER_LOG" 2>&1 &
    CONTROLLER_PID=$!
    sleep 1
    if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
        echo "ERROR: controller died on startup. See $CONTROLLER_LOG."
        exit 1
    fi
    echo "  controller running (PID $CONTROLLER_PID)"
fi

# --- Export env for training script ---
export AGL_KEY
export AGL_ADMIN_KEY
export AGL_NAMESPACE
export AGL_MODEL_NAME

# Avoid conflicts with existing Ray clusters on shared machines.
export RAY_GCS_SERVER_PORT=${RAY_GCS_SERVER_PORT:-0}
export RAY_tmpdir=${RAY_tmpdir:-/tmp/ray_agl_lite_$$}

# Clean up any leftover Ray processes from previous runs.
.venv/bin/ray stop --force 2>/dev/null || true

# --- Run training ---
# Training output goes to both stdout and a log file in the run's log directory.
echo ""
echo "=== Running VERL training ==="
echo "  Training log: $LOG_DIR/training.log"
echo "  Server log:   $LOG_DIR/server.log"
exec .venv/bin/python examples/async_calc_x/train_calc_agent.py \
    --train-file "${AGL_TRAIN_FILE:-examples/async_calc_x/data/train.parquet}" \
    --val-file "${AGL_VAL_FILE:-examples/async_calc_x/data/test.parquet}" \
    "${TRAIN_ARGS[@]}" \
    2>&1 | tee "$LOG_DIR/training.log"
