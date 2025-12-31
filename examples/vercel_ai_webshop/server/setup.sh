#!/bin/bash
#
# WebShop Server Setup Script
#
# This script:
# 1. Creates a Python virtual environment
# 2. Clones the WebShop repository
# 3. Installs dependencies
# 4. Downloads the small dataset (1,000 products)
#
# Usage:
#   ./setup.sh           # Default: small dataset
#   ./setup.sh --all     # Full dataset (1.18M products, requires 4GB+ RAM)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBSHOP_DIR="$SCRIPT_DIR/webshop"
VENV_DIR="$SCRIPT_DIR/venv"

# Parse arguments
DATASET_SIZE="small"
if [[ "$1" == "--all" ]]; then
    DATASET_SIZE="all"
    echo "Will download FULL dataset (1.18M products) - this requires 4GB+ RAM"
else
    echo "Will download SMALL dataset (1,000 products) - fast and lightweight"
fi

echo ""
echo "=== WebShop Server Setup ==="
echo ""

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    if python3 -c 'import sys; exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
        PYTHON_CMD="python3"
    fi
fi

if [[ -z "$PYTHON_CMD" ]]; then
    echo "Error: Python 3.8+ is required"
    echo "Please install Python 3.8 or later"
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
pip install --index-url https://pypi.org/simple --upgrade pip wheel setuptools

# Install Flask server dependencies
echo ""
echo "Installing Flask server dependencies..."
pip install --index-url https://pypi.org/simple -r "$SCRIPT_DIR/requirements.txt"

# Clone WebShop if not present
if [[ ! -d "$WEBSHOP_DIR" ]]; then
    echo ""
    echo "Cloning WebShop repository..."
    git clone https://github.com/princeton-nlp/WebShop.git "$WEBSHOP_DIR"
else
    echo ""
    echo "WebShop directory exists, pulling latest changes..."
    cd "$WEBSHOP_DIR"
    git pull || true
    cd "$SCRIPT_DIR"
fi

# Install WebShop dependencies
echo ""
echo "Installing WebShop dependencies..."
cd "$WEBSHOP_DIR"

# Install requirements (skip some heavy optional dependencies)
pip install --index-url https://pypi.org/simple -r requirements.txt || {
    echo "Some dependencies failed, trying minimal install..."
    pip install --index-url https://pypi.org/simple flask gym beautifulsoup4 rank_bm25 thefuzz numpy pandas tqdm
}

# Add WebShop to Python path
echo ""
echo "Adding WebShop to Python path..."
export PYTHONPATH="$WEBSHOP_DIR:$PYTHONPATH"

# Download dataset
echo ""
echo "Downloading $DATASET_SIZE dataset..."

# Create data directory
mkdir -p "$WEBSHOP_DIR/data"
cd "$WEBSHOP_DIR"

# Download using WebShop's setup script
if [[ -f "setup.sh" ]]; then
    # Run WebShop's setup with dataset flag
    bash setup.sh -d "$DATASET_SIZE" 2>&1 || {
        echo "WebShop setup.sh failed, trying manual download..."

        # Manual download fallback
        echo "Downloading data manually..."

        # Download items data
        if [[ ! -f "data/items_shuffle.json" && ! -f "data/items_ins_v2.json" ]]; then
            echo "Downloading product data..."
            # Try to download from WebShop releases
            if command -v wget &> /dev/null; then
                wget -q --show-progress -O data/items_shuffle.json \
                    "https://github.com/princeton-nlp/WebShop/releases/download/v1.0/items_shuffle.json" || true
            elif command -v curl &> /dev/null; then
                curl -L -o data/items_shuffle.json \
                    "https://github.com/princeton-nlp/WebShop/releases/download/v1.0/items_shuffle.json" || true
            fi
        fi
    }
else
    echo "WebShop setup.sh not found, skipping dataset download"
    echo "The server will still work but with limited functionality"
fi

cd "$SCRIPT_DIR"

# Create activation script
echo ""
echo "Creating activation script..."
cat > "$SCRIPT_DIR/activate.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/webshop:$PYTHONPATH"
echo "WebShop environment activated"
echo "Run: python webshop_server.py"
EOF
chmod +x "$SCRIPT_DIR/activate.sh"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  cd $SCRIPT_DIR"
echo "  source activate.sh"
echo "  python webshop_server.py"
echo ""
echo "Or run directly:"
echo "  source $SCRIPT_DIR/activate.sh && python $SCRIPT_DIR/webshop_server.py"
echo ""
