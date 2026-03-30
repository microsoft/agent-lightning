#!/bin/bash
# Thin wrapper for Python deploy entrypoint.
#
# Examples:
#   scripts/deploy.sh --agl-in-k8s
#   scripts/deploy.sh --agl-in-host --config examples/math-poc/vllm/.env.example
#   scripts/deploy.sh --agl-external --config deploy/.env
#   scripts/deploy.sh --cleanup --config deploy/.env
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

exec uv run agl-lite deploy "$@"
