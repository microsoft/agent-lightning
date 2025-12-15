#!/bin/bash

# Run agent against a locally deployed Qwen2.5-Coder-0.5B vLLM OpenAI-compatible API.
# Only the API endpoint/model/key are overridden to point to local service.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Local OpenAI-compatible endpoint started by vLLM (deploy_qwen2.5_coder_0.5b_single.sh)
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_MODEL="qwen2.5-coder-0.5b"
# vLLM 默认不校验 key，给一个占位
export OPENAI_API_KEY="local-placeholder"

# 运行评估（可按需调整参数）
python examples/spider/sql_agent.py \
  --mode eval \
  --num-samples -1 \
  --output outputs/qwen2.5-coder-0.5b-local.jsonl \
  --concurrency 8