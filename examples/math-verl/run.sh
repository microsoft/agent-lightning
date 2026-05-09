#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DEPLOY_CONFIG="$SCRIPT_DIR/.env.example"
if [ ! -f "$DEPLOY_CONFIG" ]; then
  echo "ERROR: $DEPLOY_CONFIG not found"
  exit 1
fi
source "$DEPLOY_CONFIG"
CONFIG_NAMESPACE="${AGL_NAMESPACE:?AGL_NAMESPACE not set}"

STATE_ENV="${AGL_LOCAL_STATE_DIR:-.local}/agl-lite.env"
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

LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
export AGL_LOG_DIR="$LOG_DIR"

echo "=== math-verl training ==="
echo "logs: $LOG_DIR"

if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'agl-vllm'; then
  echo "=== Stopping external agl-vllm; math-verl starts its own VERL vLLM server ==="
  scripts/start_vllm.sh --stop
fi

echo "=== Building images ==="
scripts/build_images.sh --include-example math-poc

NS="$CONFIG_NAMESPACE"
echo "=== Deploying agl-lite namespace: $NS ==="
if kubectl get namespace "$NS" >/dev/null 2>&1; then
  echo "=== Cleaning up previous deployment in namespace: $NS ==="
  uv run agl-lite deploy --env-file "$DEPLOY_CONFIG" --cleanup
fi
uv run agl-lite deploy --env-file "$DEPLOY_CONFIG"

AGL_HOST_URL="http://localhost:${AGL_HOST_PORT:-8080}"
echo "=== Waiting for agl-lite at $AGL_HOST_URL ==="
READY=false
for i in $(seq 1 40); do
  if curl -sf "$AGL_HOST_URL/healthz" >/dev/null 2>&1; then
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
export AGL_MODEL_NAME
export AGL_KEY
export AGL_NAMESPACE="$NS"

SMOKE_ARGS=()
if [ "${AGL_VERL_SMOKE_ROLLOUT_CHECK:-false}" = "true" ]; then
  SMOKE_ARGS+=(--smoke-rollout-check)
fi

echo "Running VERL training..."
.venv/bin/python examples/math-verl/train.py \
  --total-steps "${AGL_VERL_TOTAL_STEPS:-1}" \
  --rollout-n "${AGL_VERL_ROLLOUT_N:-2}" \
  --val-size "${AGL_VERL_VAL_SIZE:-5}" \
  --experiment-name "${AGL_VERL_EXPERIMENT_NAME:-math_verl_smoke}" \
  "${SMOKE_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/train.log"

echo "Done. Logs: $LOG_DIR"
echo "To tear down: uv run agl-lite deploy --env-file $DEPLOY_CONFIG --cleanup"
