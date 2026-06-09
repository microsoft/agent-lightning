#!/usr/bin/env bash
# Run Calc-X VERL training with agl-lite's local controller.
set -euo pipefail

AGL_SERVER_PORT=8080
AGL_KEY=dummy
LOG_SUFFIX="$(date +%Y%m%d-%H%M%S)-$$"
SERVER_LOG="/tmp/agl-lite-server-$LOG_SUFFIX.log"
CONTROLLER_LOG="/tmp/agl-lite-controller-$LOG_SUFFIX.log"

cleanup() {
    pkill -f agl-lite-server 2>/dev/null || true
    pkill -f agl-lite-controller 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}

cleanup
trap cleanup EXIT INT TERM

export PYTHONPATH="$(pwd)/examples/calc_x:${PYTHONPATH:-}"

printf 'agl-lite-server log: %s\n' "$SERVER_LOG"
printf 'agl-lite-controller log: %s\n' "$CONTROLLER_LOG"

env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    ray start --head --dashboard-host=0.0.0.0

agl-lite-server \
    port="$AGL_SERVER_PORT" \
    key="$AGL_KEY" \
    default_proxy.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    >"$SERVER_LOG" 2>&1 &

for _ in $(seq 1 60); do
    curl -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null && break
    sleep 1
done

agl-lite-controller \
    runner_type=local \
    agl_server.url="http://localhost:$AGL_SERVER_PORT" \
    agl_server.key="$AGL_KEY" \
    >"$CONTROLLER_LOG" 2>&1 &

python examples/calc_x/train_calc_agent.py \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --run-name local \
    "$@"
