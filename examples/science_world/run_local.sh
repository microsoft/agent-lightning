#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

# Run ScienceWorld VERL training with Agent Lightning's local controller.
set -euo pipefail

AGL_SERVER_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY=dummy

cleanup() {
    pkill -f agl-server 2>/dev/null || true
    pkill -f agl-controller 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}

cleanup
trap cleanup EXIT INT TERM

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# SWAgent runtime knobs (read by the rollout subprocess via the environment).
export SW_MAX_STEPS="${SW_MAX_STEPS:-30}"
export SW_ENV_STEP_LIMIT="${SW_ENV_STEP_LIMIT:-100}"
export SW_MAX_VALID_ACTIONS_SHOWN="${SW_MAX_VALID_ACTIONS_SHOWN:-50}"
export SW_OBS_SNIPPET_CHARS="${SW_OBS_SNIPPET_CHARS:-240}"
export AGL_MAX_TOKENS="${AGL_MAX_TOKENS:-256}"

agl-server \
    port="$AGL_SERVER_PORT" \
    key="$AGL_KEY" \
    default_proxy.model_name=Qwen/Qwen2.5-7B-Instruct &

for _ in $(seq 1 60); do
    curl -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null && break
    sleep 1
done

agl-controller \
    runner_type=local \
    agl_server.url="http://localhost:$AGL_SERVER_PORT" \
    agl_server.key="$AGL_KEY" &

python examples/science_world/train_sw_agent.py \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --run-name local \
    "$@"
