#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.
set -euo pipefail

SC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="$SC_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$SC_ROOT:${PYTHONPATH:-}"
cd "$SC_ROOT"
export SC_GPU="${SC_GPU:-0}"
export CUDA_VISIBLE_DEVICES="$SC_GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export SC_MODEL="${SC_MODEL:-Qwen/Qwen3.5-9B}"
SC_PORT="${SC_PORT:-8282}"
export SC_BASE_URL="http://localhost:$SC_PORT"
export SC_KEY="${SC_KEY:-sc-local}"
SC_LOG_DIR="${SC_LOG_DIR:-/tmp/score-centering-$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$SC_LOG_DIR"
python -m examples.intellect2_math.train "$@" --print-config > "$SC_LOG_DIR/config.yaml"
mapfile -t SC_LAUNCH_SETTINGS < <(python - "$SC_LOG_DIR/config.yaml" <<'PY'
import sys
from omegaconf import OmegaConf

config = OmegaConf.load(sys.argv[1])
rollout = config.actor_rollout_ref.rollout
print(str(config.actor_rollout_ref.actor.policy_loss.loss_mode == "score_centering").lower())
for sampling in (rollout, rollout.val_kwargs):
    for key in ("temperature", "top_p", "top_k"):
        print(sampling[key])
print(config.agentlightning.get("score_centering", {}).get("top_logprobs_num", 128))
print(config.data.max_response_length)
PY
)
export AGL_MAX_TOKENS="${SC_LAUNCH_SETTINGS[8]}"
if [[ "${SC_LAUNCH_SETTINGS[0]}" == true ]]; then
    export VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=1
fi
SC_SERVER_PID=""
SC_CONTROLLER_PID=""
cleanup() {
    [[ -z "$SC_CONTROLLER_PID" ]] || kill -- "-$SC_CONTROLLER_PID" 2>/dev/null || true
    [[ -z "$SC_SERVER_PID" ]] || kill -- "-$SC_SERVER_PID" 2>/dev/null || true
    wait || true
}
trap cleanup EXIT INT TERM

setsid agl-server host=127.0.0.1 port="$SC_PORT" key="$SC_KEY" \
    upstream_timeout_seconds="${SC_UPSTREAM_TIMEOUT_SECONDS:-3600}" \
    default_proxy.model_name="$SC_MODEL" \
    default_proxy.score_centering.enabled="${SC_LAUNCH_SETTINGS[0]}" \
    default_proxy.score_centering.top_logprobs_num="${SC_LAUNCH_SETTINGS[7]}" \
    default_proxy.train.temperature="${SC_LAUNCH_SETTINGS[1]}" \
    default_proxy.train.top_p="${SC_LAUNCH_SETTINGS[2]}" \
    default_proxy.train.top_k="${SC_LAUNCH_SETTINGS[3]}" default_proxy.train.min_p=0 \
    default_proxy.val.temperature="${SC_LAUNCH_SETTINGS[4]}" \
    default_proxy.val.top_p="${SC_LAUNCH_SETTINGS[5]}" \
    default_proxy.val.top_k="${SC_LAUNCH_SETTINGS[6]}" default_proxy.val.min_p=0 \
    >"$SC_LOG_DIR/server.log" 2>&1 &
SC_SERVER_PID=$!
for _ in $(seq 1 60); do
    curl -sf "$SC_BASE_URL/healthz" >/dev/null && break
    sleep 1
done
curl -sf "$SC_BASE_URL/healthz" >/dev/null
setsid agl-controller runner_type=local \
    agl_server.url="$SC_BASE_URL" agl_server.key="$SC_KEY" \
    local_runner.poll_interval=1 \
    >"$SC_LOG_DIR/controller.log" 2>&1 &
SC_CONTROLLER_PID=$!
printf 'Logs: %s\n' "$SC_LOG_DIR"
python -m examples.intellect2_math.train "$@" 2>&1 | tee "$SC_LOG_DIR/train.log"
