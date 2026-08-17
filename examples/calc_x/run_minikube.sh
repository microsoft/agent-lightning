#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

# Run Calc-X VERL training with Agent Lightning's k8s controller on minikube.
set -euo pipefail

AGL_SERVER_PORT=8181
AGL_KEY=dummy
LOG_SUFFIX="$(date +%Y%m%d-%H%M%S)-$$"
SERVER_LOG="/tmp/agl-server-$LOG_SUFFIX.log"
CONTROLLER_LOG="/tmp/agl-controller-$LOG_SUFFIX.log"

cleanup() {
    minikube stop >/dev/null 2>&1 || true
    pkill -f agl-server 2>/dev/null || true
    pkill -f agl-controller 2>/dev/null || true
    ray stop --force >/dev/null 2>&1 || true
}

cleanup
trap cleanup EXIT INT TERM

printf 'agl-server log: %s\n' "$SERVER_LOG"
printf 'agl-controller log: %s\n' "$CONTROLLER_LOG"

env LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    ray start --head --dashboard-host=0.0.0.0

minikube delete -p minikube >/dev/null 2>&1 || true
minikube start --memory=65536 --cpus=16 --driver=docker
minikube image build -t calc-x-agent:dev -f Dockerfile .

agl-server \
    port="$AGL_SERVER_PORT" \
    key="$AGL_KEY" \
    default_proxy.model_name=Qwen/Qwen2.5-1.5B-Instruct \
    >"$SERVER_LOG" 2>&1 &

for _ in $(seq 1 60); do
    curl -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null && break
    sleep 1
done

agl-controller \
    runner_type=k8s \
    agl_server.url="http://host.minikube.internal:$AGL_SERVER_PORT" \
    agl_server.key="$AGL_KEY" \
    k8s_runner.ttl_after_finished=600 \
    >"$CONTROLLER_LOG" 2>&1 &

python train_calc_agent.py \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --run-name minikube \
    "$@"
