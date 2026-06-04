#!/usr/bin/env bash
# Run Search-R1 VERL training with agl-lite's local controller.
set -euo pipefail

AGL_SERVER_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY="${AGL_KEY:-dummy}"
SEARCH_R1_MODEL="${SEARCH_R1_MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"

cleanup() {
    pkill -f agl-lite-server 2>/dev/null || true
    pkill -f agl-lite-controller 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}

cleanup
trap cleanup EXIT INT TERM

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

agl-lite-server \
    port="$AGL_SERVER_PORT" \
    key="$AGL_KEY" \
    default_proxy.model_name="$SEARCH_R1_MODEL" &

for _ in $(seq 1 60); do
    curl -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null && break
    sleep 1
done

agl-lite-controller \
    runner_type=local \
    agl_server.url="http://localhost:$AGL_SERVER_PORT" \
    agl_server.key="$AGL_KEY" &

python examples/search_r1/train_search_r1_agent.py \
    --model "$SEARCH_R1_MODEL" \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --run-name local \
    "$@"

