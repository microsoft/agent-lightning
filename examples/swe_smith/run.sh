#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail
ROLE="${1:-}"
if [ "$ROLE" != "server" ] && [ "$ROLE" != "controller" ] && [ "$ROLE" != "trainer" ]; then
  echo "Usage: $0 {server|controller|trainer} [extra args passed to trainer]"
  echo "  server      → Machine B: agl-lite-server (store + API only, no model backend)"
  echo "  controller  → Machine A: agent ConfigMap + agl-lite-controller (k8s runner)"
  echo "  trainer     → Machine B: VERL trainer (enqueues rollouts)"
  echo ""
  echo "Start order: server → controller → trainer."
  exit 1
fi
shift || true
cd "$(dirname "$0")/../.."
EXAMPLE_DIR="examples/swe_smith"
AGL_SERVER_PORT="${AGL_SERVER_PORT:-8080}"
AGL_KEY="${AGL_KEY:-dummy}"
AGL_MODEL_NAME="${AGL_MODEL_NAME:-Qwen/Qwen3-8B}"
AGL_NAMESPACE="${AGL_NAMESPACE:-default}"
PUBLIC_HOST="${AGL_SERVER_PUBLIC_HOST:-0.0.0.0}"
SERVER_URL="http://${PUBLIC_HOST}:${AGL_SERVER_PORT}"
TRAIN_DATASET_PATH="${AGL_TRAIN_DATASET_PATH-$EXAMPLE_DIR/train_dataset_mixed.jsonl}"
VAL_DATASET_PATH="${AGL_VAL_DATASET_PATH-$EXAMPLE_DIR/val_dataset_filtered.jsonl}"
export VLLM_USE_FLASHINFER_MOE_FP16="${VLLM_USE_FLASHINFER_MOE_FP16:-0}"

if { [ "$ROLE" = "controller" ] || [ "$ROLE" = "trainer" ]; } && \
   { [ ! -f "$TRAIN_DATASET_PATH" ] || [ ! -f "$VAL_DATASET_PATH" ]; }; then
  echo "ERROR: pre-split SWE-smith datasets are required:" >&2
  echo "  train: $TRAIN_DATASET_PATH" >&2
  echo "  val:   $VAL_DATASET_PATH" >&2
  echo "Set AGL_TRAIN_DATASET_PATH and AGL_VAL_DATASET_PATH to existing JSONL files." >&2
  exit 1
fi

if [ "$ROLE" = "server" ]; then
  echo "=== SWE-smith :: server (Machine B) ==="
  echo "  Public server URL: $SERVER_URL"
  echo "  No backend model is started here — this is just the store + REST API."
  echo "  Readiness criterion: GET ${SERVER_URL%/}/healthz returns 200."
  echo "=== Starting agl-lite server (no model backend) ==="
  agl-lite-server \
    port="$AGL_SERVER_PORT" \
    host="${AGL_SERVER_BIND:-0.0.0.0}" \
    key="$AGL_KEY" \
    default_proxy.model_name="$AGL_MODEL_NAME" &
  SERVER_PID=$!
  cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill "$SERVER_PID"
    fi
  }
  trap cleanup EXIT INT TERM
  ready=false
  for _ in $(seq 1 30); do
    if curl -sf "http://localhost:${AGL_SERVER_PORT}/healthz" >/dev/null 2>&1; then
      ready=true
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "ERROR: agl-lite-server exited before becoming healthy." >&2
      exit 1
    fi
    sleep 1
  done
  if [ "$ready" != true ]; then
    echo "ERROR: server not healthy after 30s (http://localhost:${AGL_SERVER_PORT}/healthz)." >&2
    exit 1
  fi
  echo "  server ready: http://localhost:${AGL_SERVER_PORT}/healthz → 200"
  echo "  Next: start the controller on Machine A, then './run.sh trainer' here."
  wait "$SERVER_PID"
elif [ "$ROLE" = "trainer" ]; then
  echo "=== SWE-smith :: trainer (Machine B) ==="
  echo "  Server: http://localhost:$AGL_SERVER_PORT"
  echo "  Model: $AGL_MODEL_NAME"
  if ! curl -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null 2>&1; then
    echo "ERROR: server not reachable at http://localhost:$AGL_SERVER_PORT — start './run.sh server' first."
    exit 1
  fi
  echo "=== Running SWE-smith training ==="
  python "$EXAMPLE_DIR/train_smith_agent.py" \
    --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
    --agl-key "$AGL_KEY" \
    --train-dataset-path "$TRAIN_DATASET_PATH" \
    --val-dataset-path "$VAL_DATASET_PATH" \
    --model "$AGL_MODEL_NAME" \
    --run-name distributed \
    "$@"
elif [ "$ROLE" = "controller" ]; then
  echo "=== SWE-smith :: controller (Machine A) ==="
  echo "  Connecting to server: $SERVER_URL"
  echo "  Namespace: $AGL_NAMESPACE"
  if ! curl -sf "${SERVER_URL}/healthz" >/dev/null 2>&1; then
    echo "WARNING: server not reachable at $SERVER_URL — start './run.sh server' on Machine B first."
  fi
  echo "=== Ensuring namespace '$AGL_NAMESPACE' exists ==="
  kubectl create namespace "$AGL_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
  echo "=== Creating agent scripts ConfigMap ==="
  kubectl -n "$AGL_NAMESPACE" create configmap swe-smith-agent-scripts \
    --from-file=smith_agent.py="$EXAMPLE_DIR/agents/smith_agent.py" \
    --dry-run=client -o yaml | kubectl -n "$AGL_NAMESPACE" apply -f -
  echo "=== Starting agl-lite controller (runner_type=k8s) ==="
  echo "  Next: start the trainer on Machine B with './run.sh trainer'."
  agl-lite-controller \
    runner_type=k8s \
    agl_server.url="$SERVER_URL" \
    agl_server.key="$AGL_KEY" \
    k8s_runner.namespace="$AGL_NAMESPACE" \
    k8s_runner.ttl_after_finished=600
fi
