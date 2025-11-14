#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.

set -e

echo "============================================"
echo "ChartQA Dataset Download Script"
echo "============================================"
echo
echo "This script downloads ChartQA from HuggingFace"
echo "and prepares it for training."
echo

# Check if datasets library is installed
echo "Checking dependencies..."
python -c "import datasets" 2>/dev/null || {
    echo ""
    echo "Error: 'datasets' library not found."
    echo "Please install it with:"
    echo "  pip install datasets"
    exit 1
}

python -c "import PIL" 2>/dev/null || {
    echo ""
    echo "Error: 'PIL' (Pillow) library not found."
    echo "Please install it with:"
    echo "  pip install Pillow"
    exit 1
}

echo "✓ Dependencies OK"
echo

# Download and prepare dataset
python prepare_data.py

echo
echo "============================================"
echo "✓ Dataset setup complete!"
echo "============================================"
echo "Dataset location: examples/chartqa/data/"
echo "- data/images/: Chart images saved locally"
echo "- data/train_chartqa.parquet: Training data"
echo "- data/test_chartqa.parquet: Test data"
echo
echo "You can now train the agent with:"
echo "  python train_chartqa_agent.py fast"
echo
