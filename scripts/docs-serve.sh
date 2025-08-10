#!/bin/bash

echo "Installing documentation dependencies..."
pip install -e .[dev]

echo "Building and serving documentation locally..."
mkdocs serve