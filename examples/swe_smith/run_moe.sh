#!/usr/bin/env bash
# Copyright (c) Microsoft. All rights reserved.

set -euo pipefail
EXAMPLE_DIR="$(cd "$(dirname "$0")" && pwd)"
export AGL_MODEL_NAME="${AGL_MODEL_NAME:-Qwen/Qwen3.5-35B-A3B}"
export AGL_TRAIN_SCRIPT="$EXAMPLE_DIR/train_smith_agent_moe.py"
export AGL_INCLUDE_ROUTED_EXPERTS=true
exec "$EXAMPLE_DIR/run.sh" "$@"
