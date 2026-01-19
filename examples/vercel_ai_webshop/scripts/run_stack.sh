#!/usr/bin/env bash
# WebShop Training Stack Orchestrator
#
# Runs all three services (WebShop, AGL Coordinator, Runners) as processes
# within a single container. This pattern is recommended for Azure ML jobs
# where all services communicate via localhost.
#
# Usage:
#   ./run_stack.sh qwen         # GPU training with Qwen model
#   ./run_stack.sh fast         # Fast training mode
#
# Environment Variables:
#   N_RUNNERS          Number of runner processes (default: 1)
#   HF_TOKEN           HuggingFace token for model access
#   WANDB_API_KEY      Weights & Biases API key for logging

set -euo pipefail

# Parse arguments
MODE="${1:-qwen}"

# Configuration
N_RUNNERS="${N_RUNNERS:-1}"

# Local URLs for inter-service communication
export WEBSHOP_URL="${WEBSHOP_URL:-http://127.0.0.1:3000}"
export AGENT_LIGHTNING_STORE_URL="${AGENT_LIGHTNING_STORE_URL:-http://127.0.0.1:4747}"
export AGENT_LIGHTNING_OTLP_ENDPOINT="${AGENT_LIGHTNING_OTLP_ENDPOINT:-http://127.0.0.1:4747/v1/traces}"
export AGENT_LIGHTNING_MODE="${AGENT_LIGHTNING_MODE:-train}"
export AGENT_LIGHTNING_SERVICE_NAME="${AGENT_LIGHTNING_SERVICE_NAME:-webshop-runner}"

# PID tracking for cleanup
PIDS=()

cleanup() {
    echo ""
    echo ">> Shutting down services..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "   Stopping PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait
    echo ">> All services stopped."
}

trap cleanup EXIT INT TERM

wait_for_health() {
    local url="$1"
    local name="$2"
    local max_wait="${3:-120}"

    echo ">> Waiting for $name to be healthy ($url)..."
    local count=0
    until curl -sf "$url" >/dev/null 2>&1; do
        sleep 2
        count=$((count + 2))
        if [ $count -ge $max_wait ]; then
            echo "   ERROR: $name did not become healthy within ${max_wait}s"
            return 1
        fi
    done
    echo "   $name is healthy."
}

echo "========================================"
echo "WebShop Training Stack"
echo "========================================"
echo "Mode: $MODE"
echo "Runners: $N_RUNNERS"
echo ""

# Determine base directory (support both Docker /app and Azure ML relative paths)
if [[ -d "/app/webshop" ]]; then
    # Docker container
    BASE_DIR="/app"
    WEBSHOP_BASE="/app/webshop"
else
    # Azure ML or local - use current directory
    BASE_DIR="$(pwd)"
    WEBSHOP_BASE="$BASE_DIR/server/webshop"
fi

echo "Base directory: $BASE_DIR"
echo "WebShop base: $WEBSHOP_BASE"
echo ""

# Add WebShop to PYTHONPATH so web_agent_site module can be imported
export PYTHONPATH="$WEBSHOP_BASE:${PYTHONPATH:-}"
echo "PYTHONPATH: $PYTHONPATH"

# ==============================================================================
# Step 1: Initialize WebShop Data (if needed)
# ==============================================================================

DATA_DIR="$WEBSHOP_BASE/data"
SEARCH_DIR="$WEBSHOP_BASE/search_engine"
INDEX_DIR="$SEARCH_DIR/indexes_1k"

if [[ ! -f "$DATA_DIR/items_shuffle_1000.json" ]]; then
    echo ">> Downloading WebShop dataset (first run only)..."
    mkdir -p "$DATA_DIR"

    gdown --quiet "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" -O "$DATA_DIR/items_shuffle_1000.json"
    gdown --quiet "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" -O "$DATA_DIR/items_ins_v2_1000.json"
    gdown --quiet "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O" -O "$DATA_DIR/items_human_ins.json"

    echo "   Dataset downloaded."
else
    echo ">> Dataset found. Skipping download."
fi

if [[ ! -d "$INDEX_DIR" ]]; then
    echo ">> Building search index (first run only)..."

    mkdir -p "$SEARCH_DIR/resources"
    mkdir -p "$SEARCH_DIR/resources_100"
    mkdir -p "$SEARCH_DIR/resources_1k"
    mkdir -p "$SEARCH_DIR/resources_100k"

    cd "$SEARCH_DIR"
    python convert_product_file_format.py

    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input resources_1k \
        --index indexes_1k \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1 \
        --storePositions --storeDocvectors --storeRaw

    cd "$BASE_DIR"
    echo "   Search index built."
else
    echo ">> Search index found. Skipping build."
fi

# ==============================================================================
# Step 2: Start WebShop Server
# ==============================================================================

echo ""
echo ">> Starting WebShop server on port 3000..."

# Run WebShop without GPU (CUDA_VISIBLE_DEVICES="")
CUDA_VISIBLE_DEVICES="" python "$BASE_DIR/server/webshop_server.py" --host 127.0.0.1 --port 3000 &
WEBSHOP_PID=$!
PIDS+=("$WEBSHOP_PID")

echo "   WebShop PID: $WEBSHOP_PID"

# ==============================================================================
# Step 3: Start Agent Lightning Coordinator
# ==============================================================================

echo ""
echo ">> Starting Agent Lightning coordinator on port 4747..."

cd "$BASE_DIR/agl"

echo "   Mode: Training ($MODE)"
python run_training.py "$MODE" &

AGL_PID=$!
PIDS+=("$AGL_PID")

echo "   Coordinator PID: $AGL_PID"

cd "$BASE_DIR"

# ==============================================================================
# Step 4: Wait for Services
# ==============================================================================

echo ""
wait_for_health "http://127.0.0.1:3000/health" "WebShop" 120
wait_for_health "http://127.0.0.1:4747/v1/agl/health" "Coordinator" 60

# ==============================================================================
# Step 5: Build and Start Runners
# ==============================================================================

echo ""
echo ">> Building headless runner..."

# Ensure we're in the right directory for pnpm
cd "$BASE_DIR"

# Build the headless runner (compiles TypeScript, resolves path aliases)
pnpm build:headless || {
    echo "   ERROR: Failed to build headless runner"
    exit 1
}
echo "   Build complete."

echo ""
echo ">> Starting $N_RUNNERS runner(s)..."

for i in $(seq 1 "$N_RUNNERS"); do
    export WORKER_ID="runner-$i"
    echo "   Starting runner-$i..."
    pnpm headless -- --worker-id "runner-$i" &
    RUNNER_PID=$!
    PIDS+=("$RUNNER_PID")
    echo "   Runner-$i PID: $RUNNER_PID"
done

# ==============================================================================
# Step 6: Wait for Training to Complete
# ==============================================================================

echo ""
echo "========================================"
echo "All services started. Training in progress..."
echo "========================================"
echo ""
echo "Services:"
echo "  - WebShop:     http://127.0.0.1:3000"
echo "  - Coordinator: http://127.0.0.1:4747"
echo "  - Runners:     $N_RUNNERS process(es)"
echo ""
echo "Press Ctrl+C to stop all services."
echo ""

# Wait for the coordinator to finish (it drives the training)
wait "$AGL_PID" || true

echo ""
echo ">> Training completed or coordinator exited."
