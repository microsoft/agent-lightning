#!/usr/bin/env bash
# Prepare Search-R1 retrieval data and the retriever conda environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SEARCH_R1_DATA_DIR:-$SCRIPT_DIR/data}"
ENV_NAME="${SEARCH_R1_RETRIEVER_ENV:-retriever}"
PYTHON_VERSION="${SEARCH_R1_RETRIEVER_PYTHON:-3.10}"
PYTORCH_CUDA="${SEARCH_R1_PYTORCH_CUDA:-12.1}"

WIKI_URL="https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz"
INDEX_A_URL="https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa"
INDEX_B_URL="https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab"
TRAIN_URL="https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/train.parquet"
TEST_URL="https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train/resolve/main/test.parquet"

download() {
    local url="$1"
    local output="$2"
    if [[ -s "$output" ]]; then
        echo "skip existing $output"
        return
    fi
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 5 -o "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$output" "$url"
    else
        echo "curl or wget is required to download $url" >&2
        exit 1
    fi
}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is required to create/use the Search-R1 retriever environment" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

if [[ "${SEARCH_R1_SKIP_RETRIEVER_INSTALL:-0}" != "1" ]]; then
    conda install -y \
        "pytorch==2.4.0" \
        "torchvision==0.19.0" \
        "torchaudio==2.4.0" \
        "pytorch-cuda=$PYTORCH_CUDA" \
        -c pytorch -c nvidia
    conda install -y -c pytorch -c nvidia "faiss-gpu=1.8.0"
    python -m pip install -U "transformers==4.57.1" datasets pyserini uvicorn fastapi tqdm
fi

mkdir -p "$DATA_DIR"

download "$WIKI_URL" "$DATA_DIR/wiki-18.jsonl.gz"
download "$INDEX_A_URL" "$DATA_DIR/part_aa"
download "$INDEX_B_URL" "$DATA_DIR/part_ab"
download "$TRAIN_URL" "$DATA_DIR/train.parquet"
download "$TEST_URL" "$DATA_DIR/test.parquet"

if [[ ! -s "$DATA_DIR/e5_Flat.index" ]]; then
    cat "$DATA_DIR"/part_* > "$DATA_DIR/e5_Flat.index"
fi

if [[ ! -s "$DATA_DIR/wiki-18.jsonl" ]]; then
    gzip -dk "$DATA_DIR/wiki-18.jsonl.gz"
fi

echo "Search-R1 data ready in $DATA_DIR"
echo "Retriever env: $ENV_NAME"
