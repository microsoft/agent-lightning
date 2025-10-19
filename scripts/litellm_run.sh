#!/usr/bin/env bash
set -euo pipefail
export

# Launch LiteLLM Proxy in background
echo "Starting LiteLLM Proxy on port 12306..."
nohup uv run litellm --config scripts/litellm_ci.yaml --port 12306 &

# Wait for the server to be ready
echo "Waiting for LiteLLM Proxy to start..."
for i in {1..30}; do
  if curl -s http://localhost:12306/v1/models > /dev/null; then
    echo "LiteLLM Proxy is up!"
    break
  fi
  echo "Waiting... ($i)"
  # Wait for 2 seconds before checking again
  sleep 2
done

# Run sanity check
echo "Running sanity check..."
export OPENAI_BASE_URL="http://localhost:12306/"
export OPENAI_API_KEY="dummy"
uv run scripts/litellm_sanity_check.py

echo "Sanity check complete!"
