#!/bin/bash
set -e

DATA_DIR="/app/webshop/data"
SEARCH_DIR="/app/webshop/search_engine"
INDEX_DIR="$SEARCH_DIR/indexes_1k"

echo ">> WebShop Server Entrypoint"

# Check if small dataset exists in the persistent volume
if [[ ! -f "$DATA_DIR/items_shuffle_1000.json" ]]; then
    echo ">> Dataset not found in volume. Downloading (this happens once)..."

    mkdir -p "$DATA_DIR"

    # Download small dataset (1000 products) from Google Drive using gdown
    # These are the pre-processed files expected by WebShop
    echo ">> Downloading items_shuffle_1000.json (~15MB)..."
    gdown --quiet "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib" -O "$DATA_DIR/items_shuffle_1000.json"

    echo ">> Downloading items_ins_v2_1000.json (~3MB)..."
    gdown --quiet "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu" -O "$DATA_DIR/items_ins_v2_1000.json"

    echo ">> Downloading items_human_ins.json (~1MB)..."
    gdown --quiet "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O" -O "$DATA_DIR/items_human_ins.json"

    echo ">> Download complete."
else
    echo ">> Dataset found in volume. Skipping download."
fi

# Build search index if it doesn't exist
if [[ ! -d "$INDEX_DIR" ]]; then
    echo ">> Building search index (this happens once)..."

    # Create all resources directories needed by convert_product_file_format.py
    mkdir -p "$SEARCH_DIR/resources"
    mkdir -p "$SEARCH_DIR/resources_100"
    mkdir -p "$SEARCH_DIR/resources_1k"
    mkdir -p "$SEARCH_DIR/resources_100k"

    # Convert product data to pyserini format
    echo ">> Converting product data to search format..."
    cd "$SEARCH_DIR"
    python convert_product_file_format.py

    # Build Lucene index using pyserini
    echo ">> Building Lucene index (this may take a minute)..."
    python -m pyserini.index.lucene \
        --collection JsonCollection \
        --input resources_1k \
        --index indexes_1k \
        --generator DefaultLuceneDocumentGenerator \
        --threads 1 \
        --storePositions --storeDocvectors --storeRaw

    echo ">> Search index built successfully."
    cd /app
else
    echo ">> Search index found. Skipping build."
fi

# Execute the main command
exec "$@"
