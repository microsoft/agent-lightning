#!/bin/bash
#
# Agent Lightning Training Setup Script
#
# This script:
# 1. Creates a Python virtual environment
# 2. Installs Agent Lightning and dependencies
# 3. Creates an activation script
#
# Usage:
#   ./setup.sh           # Default: dev mode (CPU-only, minimal deps)
#   ./setup.sh --gpu     # Full training with VERL (GPU required)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Parse arguments
INSTALL_MODE="dev"
if [[ "$1" == "--gpu" ]]; then
    INSTALL_MODE="gpu"
    echo "Installing for GPU training with VERL"
else
    echo "Installing for CPU-only dev mode (use --gpu for full training)"
fi

echo ""
echo "=== Agent Lightning Training Setup ==="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Error: Python 3.10+ is required"
    echo "Please install Python 3.10 or later"
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install dependencies based on mode
echo ""
if [[ "$INSTALL_MODE" == "gpu" ]]; then
    echo "Installing GPU training dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"

    # Install Agent Lightning with VERL extras from repo root
    echo ""
    echo "Installing Agent Lightning with VERL support..."
    cd "$REPO_ROOT"
    pip install -e ".[verl]" || {
        echo "VERL extras failed, installing base package..."
        pip install -e .
    }
else
    echo "Installing dev mode dependencies..."
    pip install -r "$SCRIPT_DIR/requirements.txt"

    # Install Agent Lightning from repo root (no VERL)
    echo ""
    echo "Installing Agent Lightning (dev mode)..."
    cd "$REPO_ROOT"
    pip install -e .
fi

cd "$SCRIPT_DIR"

# Create activation script
echo ""
echo "Creating activation script..."
cat > "$SCRIPT_DIR/activate.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
echo "Agent Lightning training environment activated"
echo ""
echo "Commands:"
echo "  python run_training.py --dev          # Dev mode (CPU-only)"
echo "  python run_training.py fast           # Fast training (GPU)"
echo "  python run_training.py qwen           # Full Qwen training (GPU)"
echo ""
echo "Remember to start the headless runner in another terminal:"
echo "  cd ../  # Go to vercel_ai_webshop directory"
echo "  pnpm headless -- --worker-id runner-1"
EOF
chmod +x "$SCRIPT_DIR/activate.sh"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run training:"
echo "  cd $SCRIPT_DIR"
echo "  source activate.sh"
echo "  python run_training.py --dev    # For CPU-only dev mode"
echo ""
echo "Make sure to also start:"
echo "  1. WebShop server:  cd .. && docker compose up webshop"
echo "  2. Headless runner: cd .. && pnpm headless -- --worker-id runner-1"
echo ""
