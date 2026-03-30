#!/bin/bash
# Thin wrapper for Python deploy entrypoint.
#
# Usage:
#   export AGL_KEY=$(openssl rand -hex 32)
#   scripts/deploy.sh --config deploy/agl-lite.yaml
#   scripts/deploy.sh --config deploy/agl-lite.yaml --cleanup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

exec uv run agl-lite deploy "$@"
