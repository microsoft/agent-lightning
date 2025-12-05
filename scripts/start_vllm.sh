#!/bin/bash
# Start vLLM server with Qwen2.5-0.5B-Instruct model
# This script starts a vLLM server that provides an OpenAI-compatible API

set -e

# Configuration
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
PORT=8000
HOST="0.0.0.0"

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vLLM is not installed."
    echo "Please install it with: pip install vllm"
    exit 1
fi

echo "Starting vLLM server..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Host: $HOST"
echo ""
echo "The server will be available at: http://localhost:$PORT"
echo "OpenAI API endpoint: http://localhost:$PORT/v1"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start vLLM with OpenAI-compatible server
# Additional options for the small 0.5B model:
# --max-model-len: Reduced from default to save memory
# --gpu-memory-utilization: How much GPU memory to use (0.9 = 90%)
# --dtype: Use half precision for better performance on small models

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --host "$HOST" \
    --max-model-len 2048 \
    --gpu-memory-utilization 0.9 \
    --dtype half
