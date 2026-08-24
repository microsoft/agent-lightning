#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

# Launch the Search-R1 retrieval endpoint used by SearchR1Agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SEARCH_R1_DATA_DIR:-$SCRIPT_DIR/data}"
ENV_NAME="${SEARCH_R1_RETRIEVER_ENV:-retriever}"
HOST="${SEARCH_R1_RETRIEVAL_HOST:-0.0.0.0}"
PORT="${SEARCH_R1_RETRIEVAL_PORT:-8000}"
TOPK="${SEARCH_R1_TOPK:-3}"
RETRIEVER_NAME="${SEARCH_R1_RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${SEARCH_R1_RETRIEVER_MODEL:-intfloat/e5-base-v2}"
DEVICE="${SEARCH_R1_RETRIEVER_DEVICE:-auto}"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to activate the Search-R1 retriever environment" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python "$SCRIPT_DIR/retrieval_server.py" \
    --index-path "$DATA_DIR/e5_Flat.index" \
    --corpus-path "$DATA_DIR/wiki-18.jsonl" \
    --topk "$TOPK" \
    --retriever-name "$RETRIEVER_NAME" \
    --retriever-model "$RETRIEVER_MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE" \
    --faiss-gpu
