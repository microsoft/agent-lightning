#!/bin/bash
# Start vLLM serving in a Docker container.
#
# Usage:
#   scripts/start_vllm.sh                           # defaults
#   scripts/start_vllm.sh --model Qwen/Qwen2.5-7B-Instruct --gpu 1
#   scripts/start_vllm.sh --stop                     # stop and remove
#
# Requires: docker, nvidia-container-toolkit
set -euo pipefail

# Defaults (override via args or env)
MODEL="${AGL_MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
PORT="${AGL_VLLM_PORT:-8010}"
GPU="${AGL_VLLM_GPU:-0}"
MAX_MODEL_LEN="${AGL_VLLM_MAX_MODEL_LEN:-2048}"
GPU_MEM_UTIL="${AGL_VLLM_GPU_MEM_UTIL:-0.2}"
CONTAINER_NAME="agl-vllm"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --gpu-mem) GPU_MEM_UTIL="$2"; shift 2 ;;
        --stop)
            echo "Stopping $CONTAINER_NAME..."
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME" 2>/dev/null || true
            echo "Done."
            exit 0
            ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Stop existing container if running
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "=== Starting vLLM ==="
echo "  Model:    $MODEL"
echo "  Port:     $PORT (host) → 8000 (container)"
echo "  GPU:      $GPU"
echo "  Max len:  $MAX_MODEL_LEN"
echo "  GPU mem:  $GPU_MEM_UTIL"

docker run -d --name "$CONTAINER_NAME" \
    --gpus "\"device=$GPU\"" \
    -p "$PORT:8000" \
    --shm-size=1g \
    -e HF_HOME=/root/.cache/huggingface \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    --model "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_UTIL"

echo ""
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
    if curl -sf "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "vLLM ready after $((i*2))s"
        echo ""
        curl -sf "http://localhost:$PORT/v1/models" | python3 -m json.tool
        echo ""
        echo "=== vLLM is serving ==="
        echo "  Host:      http://localhost:$PORT/v1"
        echo "  Minikube:  http://host.minikube.internal:$PORT/v1"
        echo "  Logs:      docker logs $CONTAINER_NAME"
        echo "  Stop:      scripts/start_vllm.sh --stop"
        exit 0
    fi
    STATUS=$(docker inspect --format='{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "unknown")
    if [ "$STATUS" = "exited" ]; then
        echo "ERROR: vLLM container exited after $((i*2))s"
        docker logs "$CONTAINER_NAME" --tail 20
        exit 1
    fi
    sleep 2
done

echo "ERROR: vLLM did not become ready within 240s"
docker logs "$CONTAINER_NAME" --tail 20
exit 1
