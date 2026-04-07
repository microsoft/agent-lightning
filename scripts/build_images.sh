#!/bin/bash
# Build Docker images into minikube.
#
# Usage:
#   scripts/build_images.sh                                    # core only
#   scripts/build_images.sh --include-example math-poc         # core + math-poc
#   scripts/build_images.sh --include-example math-poc --include-example calc-x
#
# Legacy (still supported):
#   scripts/build_images.sh --math-poc
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parse arguments — collect example names.
EXAMPLES=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --include-example)
            EXAMPLES+=("$2")
            shift 2
            ;;
        --math-poc)
            # Legacy flag — map to new form.
            EXAMPLES+=("math-poc")
            shift
            ;;
        all)
            EXAMPLES+=("math-poc" "calc-x")
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--include-example <name>]..."
            exit 1
            ;;
    esac
done

# --- Core image (always built) ---
echo "=== Building agl-lite:dev ==="
minikube image build -t agl-lite:dev -f deploy/agl-lite/Dockerfile "$REPO_ROOT"

# --- Example images ---
for example in "${EXAMPLES[@]}"; do
    case "$example" in
        math-poc)
            echo "=== Building mockai:dev ==="
            minikube image build -t mockai:dev ~/mockai

            if [ -f "$REPO_ROOT/examples/math-poc/Dockerfile.agent" ]; then
                echo "=== Building math-agent:dev ==="
                minikube image build -t math-agent:dev \
                    -f Dockerfile.agent \
                    "$REPO_ROOT/examples/math-poc"
            fi
            ;;
        calc-x)
            if [ -f "$REPO_ROOT/examples/calc_x/Dockerfile.agent" ]; then
                echo "=== Building calc-agent:dev ==="
                minikube image build -t calc-agent:dev \
                    -f Dockerfile.agent \
                    "$REPO_ROOT/examples/calc_x"
            else
                echo "WARNING: examples/calc_x/Dockerfile.agent not found, skipping"
            fi
            ;;
        *)
            echo "WARNING: Unknown example '$example', skipping"
            ;;
    esac
done

echo "=== Done ==="
minikube image ls --format='{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -E "agl-lite|mockai|math-agent|calc-agent" || true
