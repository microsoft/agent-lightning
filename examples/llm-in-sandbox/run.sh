#!/bin/bash
# Run llm-in-sandbox local K8s deployment and VERL training end to end.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_CONFIG="$SCRIPT_DIR/.env.example"

cd "$REPO_ROOT"

if [ ! -f "$DEPLOY_CONFIG" ]; then
  echo "ERROR: $DEPLOY_CONFIG not found"
  exit 1
fi

source "$DEPLOY_CONFIG"
CONFIG_NAMESPACE="${AGL_NAMESPACE:?AGL_NAMESPACE not set}"
STATE_ENV="${AGL_LOCAL_STATE_DIR:-.local}/agl-lite.env"
WANDB_ENV="${AGL_LOCAL_STATE_DIR:-.local}/wandb.env"

if [ -z "${AGL_LOG_DIR:-}" ]; then
  export AGL_LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$AGL_LOG_DIR/agents"
echo "=== Logs: $AGL_LOG_DIR ==="

if [ -z "${AGL_KEY:-}" ] && [ -f "$STATE_ENV" ]; then
  echo "=== Loading AGL_KEY from $STATE_ENV ==="
  source "$STATE_ENV"
fi

if [ -f "$WANDB_ENV" ]; then
  echo "=== Loading W&B config from $WANDB_ENV ==="
  set -a
  source "$WANDB_ENV"
  set +a
fi

export AGL_KEY="${AGL_KEY:-}"
if [ -z "$AGL_KEY" ]; then
  echo "ERROR: AGL_KEY not set. Either:"
  echo "  export AGL_KEY=\$(openssl rand -hex 32)"
  echo "  or run agl-lite deploy once so $STATE_ENV exists"
  exit 1
fi

if [ -z "${AGL_ADMIN_KEY:-}" ] && [ -f "$STATE_ENV" ]; then
  echo "=== Loading AGL_ADMIN_KEY from $STATE_ENV ==="
  source "$STATE_ENV"
fi

if [ -z "${AGL_ADMIN_KEY:-}" ]; then
  echo "=== Generating AGL_ADMIN_KEY ==="
  export AGL_ADMIN_KEY="$(openssl rand -hex 32)"
fi

if [ "$AGL_ADMIN_KEY" = "$AGL_KEY" ]; then
  echo "ERROR: AGL_ADMIN_KEY must differ from AGL_KEY"
  exit 1
fi
export AGL_ADMIN_KEY

MOUNT_SRC="$AGL_LOG_DIR/agents"
MOUNT_DST="/tmp/agl-lite/logs"
STALE_PIDS=$(pgrep -f "minikube mount.*:$MOUNT_DST" || true)
if [ -n "$STALE_PIDS" ]; then
  echo "=== Cleaning up stale minikube mount (PIDs: $STALE_PIDS) ==="
  kill $STALE_PIDS 2>/dev/null || true
  sleep 1
fi

echo "=== Mounting $MOUNT_SRC to minikube:$MOUNT_DST ==="
minikube mount "$MOUNT_SRC:$MOUNT_DST" &
MOUNT_PID=$!
sleep 2
if ! kill -0 "$MOUNT_PID" 2>/dev/null; then
  echo "ERROR: minikube mount failed"
  exit 1
fi
trap 'kill $MOUNT_PID 2>/dev/null; wait $MOUNT_PID 2>/dev/null' EXIT

echo "=== Building images ==="
scripts/build_images.sh --include-example llm-in-sandbox

echo "=== Deploying agl-lite namespace: $CONFIG_NAMESPACE ==="
if kubectl get namespace "$CONFIG_NAMESPACE" >/dev/null 2>&1; then
  echo "=== Cleaning up previous deployment in namespace: $CONFIG_NAMESPACE ==="
  uv run agl-lite deploy --env-file "$DEPLOY_CONFIG" --cleanup
fi
uv run agl-lite deploy --env-file "$DEPLOY_CONFIG"

export AGL_BASE_URL="${AGL_BASE_URL:-http://localhost:${AGL_HOST_PORT:-8080}}"
echo "=== Waiting for agl-lite at $AGL_BASE_URL ==="
READY=false
for _ in $(seq 1 60); do
  if curl -sf "$AGL_BASE_URL/healthz" >/dev/null 2>&1; then
    echo "  agl-lite ready"
    READY=true
    break
  fi
  sleep 1
done

if [ "$READY" != true ]; then
  echo "ERROR: agl-lite did not become ready. See $AGL_LOG_DIR/server.log"
  exit 1
fi

export AGL_NAMESPACE="$CONFIG_NAMESPACE"
export AGL_HOOKS
export AGL_POD_SPEC_TEMPLATE
export AGL_MODEL_NAME
export AGL_OPENAI_MODEL_PREFIX
export AGL_LLM_TEMPERATURE
export OPENAI_TIMEOUT
export MAX_TOKENS_PER_CALL
export AGL_TRAIN_DATA_DIR
export AGL_TEST_DATA_DIR

export RAY_GCS_SERVER_PORT="${RAY_GCS_SERVER_PORT:-0}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_agl_lite_llm_sandbox_$$}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/wandb}"

for wandb_var in WANDB_API_KEY WANDB_ENTITY WANDB_PROJECT WANDB_RUN_ID WANDB_RESUME; do
  if [ -n "${!wandb_var:-}" ]; then
    export "$wandb_var"
  else
    unset "$wandb_var"
  fi
done


PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -n "${VIRTUAL_ENV:-}" ]; then
  PYTHON_BIN="$VIRTUAL_ENV/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PYTHON_BIN" ] || [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: Python executable not found. Run scripts/setup_verl.sh first or set PYTHON_BIN."
  exit 1
fi

RAY_BIN="$(dirname "$PYTHON_BIN")/ray"
if [ -x "$RAY_BIN" ]; then
  "$RAY_BIN" stop --force 2>/dev/null || true
else
  "$PYTHON_BIN" -m ray stop --force 2>/dev/null || true
fi

echo "=== Running llm-in-sandbox VERL training ==="
echo "  agl-lite: $AGL_BASE_URL"
echo "  logs:     $AGL_LOG_DIR"
echo "  python:   $PYTHON_BIN"
echo "  wandb:    mode=${WANDB_MODE:-unset}, project=${AGL_VERL_PROJECT_NAME:-unset}, run=${AGL_VERL_EXPERIMENT_NAME:-unset}"
echo "  wandb:    entity=${WANDB_ENTITY:-default}, dir=${WANDB_DIR:-unset}, api_key=$([ -n "${WANDB_API_KEY:-}" ] && echo set || echo missing)"
"$PYTHON_BIN" examples/llm-in-sandbox/train_llm_in_sandbox.py "$@" 2>&1 | tee "$AGL_LOG_DIR/training.log"