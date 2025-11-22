#!/bin/bash

set -euo pipefail

# Create data directories
mkdir -p data/prometheus data/mongo

# Change permissions
chmod 777 data/prometheus data/mongo
