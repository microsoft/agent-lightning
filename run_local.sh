#!/bin/bash

# Run spider SQL agent against a locally deployed vLLM OpenAI-compatible API.
# Only the API endpoint/model/key are overridden to point to local service.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Local OpenAI-compatible endpoint started by vLLM (deploy_qwen3_4b_gpu4567_4gpu.sh)
export OPENAI_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_MODEL="qwen3-4b-40k-quad"
# vLLM 默认不校验 key，给一个占位
export OPENAI_API_KEY="local-placeholder"

# Spider 数据目录（默认已在 .env 中设置，可按需覆盖为绝对路径）
# export VERL_SPIDER_DATA_DIR="$SCRIPT_DIR/examples/spider/data"

# 运行评估（可按需调整参数）
python examples/spider/sql_agent.py \
  --mode eval \
  --num-samples -1 \
  --output outputs/qwen3-4b-instruct-dev-local.jsonl \
  --concurrency 12
