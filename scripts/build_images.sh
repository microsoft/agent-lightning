#!/bin/bash
# Build Docker images into minikube.
# Usage: scripts/build_images.sh [--math-poc]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Building agl-lite:dev ==="
minikube image build -t agl-lite:dev -f deploy/agl-lite/Dockerfile "$REPO_ROOT"

# Build PoC images if requested.
if [[ "${1:-}" == "--math-poc" || "${1:-}" == "all" ]]; then
    echo "=== Building mockai:dev ==="
    minikube image build -t mockai:dev ~/mockai

    if [ -f "$REPO_ROOT/examples/math-poc/Dockerfile.agent" ]; then
        echo "=== Building math-agent:dev ==="
        minikube image build -t math-agent:dev \
            "$REPO_ROOT/examples/math-poc"
    fi
fi

echo "=== Done ==="
minikube image ls --format='{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E "agl-lite|mockai|math-agent" || true
