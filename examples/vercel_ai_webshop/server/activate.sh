#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export PYTHONPATH="$SCRIPT_DIR/webshop:$PYTHONPATH"
echo "WebShop environment activated"
echo "Run: python webshop_server.py"
