#!/bin/bash
# Launch the ScienceWorld VERL training end-to-end on a single host.
#
# Topology (all on this host — no K8s, no Docker):
#   - agl-lite serve (HTTP + gateway + store)            background pid 1
#   - agl-lite controller --runner-type=local            background pid 2
#       └ spawns one Python subprocess per rollout, each runs SWAgent
#   - train_sw_agent.py                                   foreground
#       └ VERL boots Ray + vLLM internally and drives the loop
#
# Prerequisites:
#   - .venv with agl-lite, VERL, vLLM, torch, scienceworld, Java 1.8+
#   - AGL_KEY set in environment (or via .local/agl-lite.env)
#
# Usage:
#   examples/science_world/run.sh              # full training
#   examples/science_world/run.sh --ci-fast    # 1 PPO step smoke test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- Log directory ---
LOG_DIR="$SCRIPT_DIR/logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"
export AGL_LOG_DIR="$LOG_DIR"
echo "=== Logs → $LOG_DIR ==="

# --- Load config ---
if [ ! -f "$SCRIPT_DIR/.env.example" ]; then
    echo "ERROR: $SCRIPT_DIR/.env.example not found"
    exit 1
fi
# shellcheck disable=SC1091
source "$SCRIPT_DIR/.env.example"

# --- Auth keys ---
STATE_ENV=".local/agl-lite.env"
if [ -z "${AGL_KEY:-}" ] && [ -f "$STATE_ENV" ]; then
    # shellcheck disable=SC1090
    source "$STATE_ENV"
fi
if [ -z "${AGL_KEY:-}" ]; then
    echo "ERROR: AGL_KEY not set. Run: export AGL_KEY=\$(openssl rand -hex 32)"
    exit 1
fi
if [ -z "${AGL_ADMIN_KEY:-}" ]; then
    AGL_ADMIN_KEY="$(openssl rand -hex 32)"
    echo "=== Generated AGL_ADMIN_KEY ==="
fi
if [ "$AGL_ADMIN_KEY" = "$AGL_KEY" ]; then
    echo "ERROR: AGL_ADMIN_KEY must differ from AGL_KEY"
    exit 1
fi
export AGL_KEY AGL_ADMIN_KEY AGL_MODEL_NAME AGL_HOST_PORT AGL_LOCAL_POOL_SIZE
export AGL_TASK_NAMES AGL_VARIATIONS_PER_TASK AGL_SIMPLIFICATION
export AGL_N_GPUS_PER_NODE
export SW_MAX_STEPS SW_ENV_STEP_LIMIT SW_MAX_VALID_ACTIONS_SHOWN SW_OBS_SNIPPET_CHARS
export AGL_TEMPERATURE_TRAIN AGL_TEMPERATURE_VAL AGL_MAX_TOKENS
export AGL_BASE_URL="http://localhost:${AGL_HOST_PORT}"

# --- NUMA topology ---
# On the 8×A100 host the GPUs are split across 4 NUMA nodes:
#   GPU0/1 → NUMA 1 (CPU 24-47), GPU2/3 → NUMA 0 (CPU 0-23),
#   GPU4/5 → NUMA 3 (CPU 72-95), GPU6/7 → NUMA 2 (CPU 48-71).
# Pin agl-lite serve to NUMA 2 so it stops bouncing across nodes; leave
# the controller un-pinned so the 64 SWAgent JVM subprocesses spread over
# all cores instead of crowding one node. Skip pinning entirely on hosts
# with a single NUMA node (e.g. dev boxes).
NUMA_NODES=$(numactl --hardware 2>/dev/null | awk '/^available:/ {print $2}')
if [ "${NUMA_NODES:-1}" -ge 2 ]; then
    SERVER_NUMA_PREFIX=(numactl --cpunodebind=2 --membind=2)
else
    SERVER_NUMA_PREFIX=()
fi

# --- Clean up any leftover Ray from prior runs ---
.venv/bin/ray stop --force 2>/dev/null || true

# --- Start agl-lite serve ---
SERVER_LOG="$LOG_DIR/server.log"
echo "=== Starting agl-lite serve on :$AGL_HOST_PORT (log: $SERVER_LOG) ==="
"${SERVER_NUMA_PREFIX[@]}" uv run agl-lite serve \
    --host 0.0.0.0 \
    --port "$AGL_HOST_PORT" \
    --gateway-config "$SCRIPT_DIR/gateway-config.yaml" \
    --hooks "$SCRIPT_DIR/hooks.py" \
    --log-dir "$LOG_DIR" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Cleanup trap — kill server + controller on any exit path.
cleanup() {
    set +e
    echo "=== Cleanup: stopping controller (PID ${CONTROLLER_PID:-?}) and server (PID $SERVER_PID) ==="
    if [ -n "${CONTROLLER_PID:-}" ]; then
        kill -TERM "$CONTROLLER_PID" 2>/dev/null || true
    fi
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    sleep 2
    if [ -n "${CONTROLLER_PID:-}" ]; then
        kill -KILL "$CONTROLLER_PID" 2>/dev/null || true
    fi
    kill -KILL "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for /healthz.
echo "=== Waiting for agl-lite ==="
READY=false
for _ in $(seq 1 40); do
    if curl -sf "http://localhost:$AGL_HOST_PORT/healthz" > /dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 1
done
if [ "$READY" != true ]; then
    echo "ERROR: agl-lite did not become ready in 40s. See $SERVER_LOG."
    exit 1
fi
echo "  agl-lite ready"

# --- Start the local-runner controller ---
CONTROLLER_LOG="$LOG_DIR/controller.log"
echo "=== Starting agl-lite controller (local runner, pool=$AGL_LOCAL_POOL_SIZE, log: $CONTROLLER_LOG) ==="
uv run agl-lite controller \
    --base-url "$AGL_BASE_URL" \
    --namespace "${AGL_NAMESPACE:-local}" \
    --runner-type local \
    --local-pool-size "$AGL_LOCAL_POOL_SIZE" \
    --local-agent-class examples.science_world.agents.sw_agent:SWAgent \
    > "$CONTROLLER_LOG" 2>&1 &
CONTROLLER_PID=$!
sleep 1
if ! kill -0 "$CONTROLLER_PID" 2>/dev/null; then
    echo "ERROR: controller died on startup. See $CONTROLLER_LOG."
    exit 1
fi
echo "  controller running (PID $CONTROLLER_PID)"

# --- Run training (foreground) ---
# AGL_HOOKS tells the trainer's rollout bridge where to load hooks so the
# on_succeeded hook can transform episode_result events into reward events.
# Without this, bridge._hooks=None and reward events are never written, which
# manifests as "Reward is None" warnings and final_reward=0.0 for every rollout.
export AGL_HOOKS="$SCRIPT_DIR/hooks.py"
TRAIN_LOG="$LOG_DIR/training.log"
echo "=== Running VERL training (log: $TRAIN_LOG) ==="
.venv/bin/python examples/science_world/train_sw_agent.py "$@" 2>&1 | tee "$TRAIN_LOG"
