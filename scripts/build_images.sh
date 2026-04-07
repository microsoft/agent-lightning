#!/bin/bash
# Build Docker images into minikube.
#
# Usage:
#   scripts/build_images.sh                                    # core only
#   scripts/build_images.sh --include-example math-poc         # core + math-poc
#   scripts/build_images.sh --include-example math-poc --include-example calc_x
#
# Convention for examples:
#   examples/<name>/Dockerfile.agent  → image "<name-normalized>-agent:dev"
#   examples/<name>/build-extra.sh    → optional hook for additional images
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
            # Legacy flag.
            EXAMPLES+=("math-poc")
            shift
            ;;
        all)
            # Auto-discover: every dir under examples/ with a Dockerfile.agent.
            for dir in "$REPO_ROOT"/examples/*/; do
                if [ -f "$dir/Dockerfile.agent" ]; then
                    EXAMPLES+=("$(basename "$dir")")
                fi
            done
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

# --- Example images (convention-driven) ---
for example in "${EXAMPLES[@]}"; do
    example_dir="$REPO_ROOT/examples/$example"
    # Normalize name for Docker tag: underscores → hyphens.
    tag_name="${example//_/-}"

    if [ ! -d "$example_dir" ]; then
        echo "WARNING: examples/$example/ not found, skipping"
        continue
    fi

    # Convention: Dockerfile.agent → <name>-agent:dev
    if [ -f "$example_dir/Dockerfile.agent" ]; then
        echo "=== Building ${tag_name}-agent:dev ==="
        minikube image build -t "${tag_name}-agent:dev" \
            -f Dockerfile.agent \
            "$example_dir"
    fi

    # Optional hook for additional images (e.g., mockai for math-poc).
    if [ -x "$example_dir/build-extra.sh" ]; then
        echo "=== Running build-extra.sh for $example ==="
        "$example_dir/build-extra.sh"
    fi
done

echo "=== Done ==="
minikube image ls --format='{{.Repository}}:{{.Tag}}' 2>/dev/null \
    | grep -E "agl-lite|agent:dev|mockai" || true
