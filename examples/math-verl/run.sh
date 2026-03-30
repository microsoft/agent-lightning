#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

source "$SCRIPT_DIR/.env.example"

if [ -z "${AGL_KEY:-}" ]; then
  echo "ERROR: AGL_KEY not set. Run: export AGL_KEY=\$(openssl rand -hex 32)"
  exit 1
fi

LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=== math-verl training ==="
echo "logs: $LOG_DIR"

echo "Reminder: controller/minikube should already be running (e.g. scripts/deploy.sh --agl-in-host)"

SERVE_PID=""
cleanup() {
  [ -n "$SERVE_PID" ] && kill "$SERVE_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Start agl-lite serve on host
SERVE_CMD=(uv run agl-lite serve --host 0.0.0.0 --port 8080)
[ -n "${AGL_GATEWAY_CONFIG:-}" ] && SERVE_CMD+=(--gateway-config "$AGL_GATEWAY_CONFIG")
[ -n "${AGL_HOOKS:-}" ] && SERVE_CMD+=(--hooks "$AGL_HOOKS")

echo "Starting agl-lite serve: ${SERVE_CMD[*]}"
AGL_KEY="$AGL_KEY" "${SERVE_CMD[@]}" > "$LOG_DIR/agl-lite.log" 2>&1 &
SERVE_PID=$!

for i in $(seq 1 30); do
  if curl -sf "$AGL_LITE_URL/healthz" >/dev/null 2>&1; then
    echo "agl-lite ready"
    break
  fi
  if ! kill -0 "$SERVE_PID" 2>/dev/null; then
    echo "ERROR: agl-lite serve exited"
    tail -40 "$LOG_DIR/agl-lite.log" || true
    exit 1
  fi
  sleep 1
done

export AGL_LITE_URL
export AGL_MODEL_NAME
export AGL_KEY

echo "Running VERL training..."
uv run python examples/math-verl/train.py \
  --total-steps "${AGL_VERL_TOTAL_STEPS:-1}" \
  --rollout-n "${AGL_VERL_ROLLOUT_N:-2}" \
  --val-size "${AGL_VERL_VAL_SIZE:-5}" \
  --experiment-name "${AGL_VERL_EXPERIMENT_NAME:-math_verl_smoke}" \
  --smoke-rollout-check \
  2>&1 | tee "$LOG_DIR/train.log"

echo "Done. Logs: $LOG_DIR"
