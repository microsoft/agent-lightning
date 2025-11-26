#!/usr/bin/env bash
set -euo pipefail
export

# Configurable port (first CLI argument, or default to 12306)
PORT="${1:-12306}"

# Launch LiteLLM Proxy in background
echo "Starting LiteLLM Proxy on port ${PORT}..."
nohup uv run litellm --config scripts/litellm_ci.yaml --port "${PORT}" &

# Wait for the server to be ready
echo "Waiting for LiteLLM Proxy to start..."
for i in {1..30}; do
  if curl -s "http://localhost:${PORT}/v1/models" > /dev/null; then
    echo "LiteLLM Proxy is up!"
    break
  fi
  echo "Waiting... (${i})"
  # Wait for 2 seconds before checking again
  sleep 2
done

# Run sanity check
echo "Running sanity check..."
export OPENAI_BASE_URL="http://localhost:${PORT}/"
export OPENAI_API_KEY="dummy"
uv run scripts/litellm_sanity_check.py

echo "Sanity check complete!"
