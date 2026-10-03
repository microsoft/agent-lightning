#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

# Run Search-R1 VERL training with Agent Lightning's local controller.
set -euo pipefail

AGL_SERVER_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY="${AGL_KEY:-dummy}"
SEARCH_R1_MODEL="${SEARCH_R1_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"

cleanup() {
    pkill -f agl-server 2>/dev/null || true
    pkill -f agl-controller 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}

cleanup
trap cleanup EXIT INT TERM

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

agl-server \
    port="$AGL_SERVER_PORT" \
    key="$AGL_KEY" \
    default_proxy.model_name="$SEARCH_R1_MODEL" &

server_ready=0
for _ in $(seq 1 60); do
    if curl --max-time 1 -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null; then
        server_ready=1
        break
    fi
    sleep 1
done
if [[ "$server_ready" -ne 1 ]]; then
    echo "Agent Lightning server did not become ready on port $AGL_SERVER_PORT after 60 attempts" >&2
    exit 1
fi

agl-controller \
    runner_type=local \
    agl_server.url="http://localhost:$AGL_SERVER_PORT" \
    agl_server.key="$AGL_KEY" &

python examples/search_r1/train_search_r1_agent.py \
    --model "$SEARCH_R1_MODEL" \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --run-name local \
    "$@"
