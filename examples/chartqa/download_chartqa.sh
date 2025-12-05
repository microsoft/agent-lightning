#!/bin/bash
set -e

uv run python -c "import datasets, PIL, pandas" 2>/dev/null || uv pip install datasets pillow pandas pyarrow

uv run python prepare_data.py
