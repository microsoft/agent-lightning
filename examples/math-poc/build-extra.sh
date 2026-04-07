#!/bin/bash
# Build additional images for math-poc (beyond the convention-driven Dockerfile.agent).
set -euo pipefail

echo "=== Building mockai:dev ==="
minikube image build -t mockai:dev ~/mockai
