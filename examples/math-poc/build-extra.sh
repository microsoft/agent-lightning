#!/bin/bash
# Build additional images for math-poc (beyond the convention-driven Dockerfile.agent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building mockai:dev ==="
minikube image build -t mockai:dev "$SCRIPT_DIR/mockai"
