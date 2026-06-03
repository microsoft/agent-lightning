#!/usr/bin/env bash
# Run llm-in-sandbox on local minikube with agl-lite.
set -euo pipefail

cd "$(dirname "$0")/../.."

cleanup() {
  pkill -f agl-lite-server 2>/dev/null || true
  pkill -f agl-lite-controller 2>/dev/null || true
  pkill -f "python .*train_llm_in_sandbox.py" 2>/dev/null || true
  ray stop --force >/dev/null 2>&1 || true
}

cleanup
trap cleanup EXIT INT TERM

AGL_SERVER_PORT=8080
AGL_KEY=dummy

echo "=== Starting minikube ==="
minikube delete -p minikube >/dev/null 2>&1 || true
minikube start --memory=65536 --cpus=16 --driver=docker

echo "=== Building llm-in-sandbox image ==="
minikube image build -t llm-in-sandbox-agent:dev -f examples/llm-in-sandbox/Dockerfile.agent examples/llm-in-sandbox

echo "=== Starting agl-lite server ==="
agl-lite-server \
  port="$AGL_SERVER_PORT" \
  key="$AGL_KEY" \
  default_proxy.model_name=Qwen/Qwen3-4B-Instruct-2507 &

echo "=== Waiting for agl-lite server ==="
for _ in $(seq 1 60); do
  if curl -sf "http://localhost:$AGL_SERVER_PORT/healthz" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "=== Starting agl-lite controller ==="
agl-lite-controller \
  runner_type=k8s \
  agl_server.url="http://localhost:$AGL_SERVER_PORT" \
  agl_server.agent_url="http://host.minikube.internal:$AGL_SERVER_PORT" \
  agl_server.key="$AGL_KEY" \
  k8s_runner.ttl_after_finished=600 &

echo "=== Running llm-in-sandbox training ==="
python examples/llm-in-sandbox/train_llm_in_sandbox.py \
  --agl-base-url "http://localhost:$AGL_SERVER_PORT" \
  --agl-key "$AGL_KEY" \
  --run-name minikube \
  "$@"