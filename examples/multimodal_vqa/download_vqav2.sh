#!/bin/bash
# Copyright (c) Microsoft. All rights reserved.

# Download and setup VQAv2 dataset for training
#
# This script downloads:
# 1. VQAv2 questions and annotations
# 2. COCO train2014 and val2014 images
# 3. Converts to parquet format using data_utils.py
#
# Dataset Info:
# - VQAv2: https://visualqa.org/download.html
# - COCO: https://cocodataset.org/#download
#
# Total size: ~20GB (13GB images + ~200MB annotations)
#
# Usage:
#   bash download_vqav2.sh

set -e  # Exit on error

# Configuration
DATA_DIR="data/vqav2"
IMAGES_DIR="$DATA_DIR/images"

echo "===================================="
echo "VQAv2 Dataset Download Script"
echo "===================================="
echo "Data directory: $DATA_DIR"
echo ""

# Create directories
mkdir -p "$DATA_DIR"
mkdir -p "$IMAGES_DIR"
mkdir -p "$IMAGES_DIR/train2014"
mkdir -p "$IMAGES_DIR/val2014"

echo "[1/4] Downloading VQAv2 Questions..."
echo "-------------------------------------"

# TODO: Download VQAv2 questions
# Training questions
if [ ! -f "$DATA_DIR/v2_OpenEnded_mscoco_train2014_questions.json" ]; then
    echo "Downloading train2014 questions..."
    # wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip -O "$DATA_DIR/v2_Questions_Train_mscoco.zip"
    # unzip "$DATA_DIR/v2_Questions_Train_mscoco.zip" -d "$DATA_DIR"
    # rm "$DATA_DIR/v2_Questions_Train_mscoco.zip"
    echo "[TODO] Download VQAv2 train questions from: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
else
    echo "Train questions already exist, skipping..."
fi

# Validation questions
if [ ! -f "$DATA_DIR/v2_OpenEnded_mscoco_val2014_questions.json" ]; then
    echo "Downloading val2014 questions..."
    # wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -O "$DATA_DIR/v2_Questions_Val_mscoco.zip"
    # unzip "$DATA_DIR/v2_Questions_Val_mscoco.zip" -d "$DATA_DIR"
    # rm "$DATA_DIR/v2_Questions_Val_mscoco.zip"
    echo "[TODO] Download VQAv2 val questions from: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip"
else
    echo "Val questions already exist, skipping..."
fi

echo ""
echo "[2/4] Downloading VQAv2 Annotations..."
echo "-------------------------------------"

# TODO: Download VQAv2 annotations
# Training annotations
if [ ! -f "$DATA_DIR/v2_mscoco_train2014_annotations.json" ]; then
    echo "Downloading train2014 annotations..."
    # wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip -O "$DATA_DIR/v2_Annotations_Train_mscoco.zip"
    # unzip "$DATA_DIR/v2_Annotations_Train_mscoco.zip" -d "$DATA_DIR"
    # rm "$DATA_DIR/v2_Annotations_Train_mscoco.zip"
    echo "[TODO] Download VQAv2 train annotations from: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
else
    echo "Train annotations already exist, skipping..."
fi

# Validation annotations
if [ ! -f "$DATA_DIR/v2_mscoco_val2014_annotations.json" ]; then
    echo "Downloading val2014 annotations..."
    # wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip -O "$DATA_DIR/v2_Annotations_Val_mscoco.zip"
    # unzip "$DATA_DIR/v2_Annotations_Val_mscoco.zip" -d "$DATA_DIR"
    # rm "$DATA_DIR/v2_Annotations_Val_mscoco.zip"
    echo "[TODO] Download VQAv2 val annotations from: https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
else
    echo "Val annotations already exist, skipping..."
fi

echo ""
echo "[3/4] Downloading COCO Images..."
echo "-------------------------------------"
echo "Warning: This is ~13GB and may take a while"
echo ""

# TODO: Download COCO images
# Training images (train2014)
if [ ! -f "$IMAGES_DIR/train2014/COCO_train2014_000000000009.jpg" ]; then
    echo "Downloading COCO train2014 images (~13GB)..."
    # wget http://images.cocodataset.org/zips/train2014.zip -O "$IMAGES_DIR/train2014.zip"
    # unzip "$IMAGES_DIR/train2014.zip" -d "$IMAGES_DIR"
    # rm "$IMAGES_DIR/train2014.zip"
    echo "[TODO] Download COCO train2014 images from: http://images.cocodataset.org/zips/train2014.zip"
    echo "[TODO] This is ~13GB, consider using wget or aria2c for faster download"
else
    echo "COCO train2014 images already exist, skipping..."
fi

# Validation images (val2014)
if [ ! -f "$IMAGES_DIR/val2014/COCO_val2014_000000000042.jpg" ]; then
    echo "Downloading COCO val2014 images (~6GB)..."
    # wget http://images.cocodataset.org/zips/val2014.zip -O "$IMAGES_DIR/val2014.zip"
    # unzip "$IMAGES_DIR/val2014.zip" -d "$IMAGES_DIR"
    # rm "$IMAGES_DIR/val2014.zip"
    echo "[TODO] Download COCO val2014 images from: http://images.cocodataset.org/zips/val2014.zip"
else
    echo "COCO val2014 images already exist, skipping..."
fi

echo ""
echo "[4/4] Converting to Parquet Format..."
echo "-------------------------------------"

# TODO: Convert to parquet format using data_utils.py
# This requires pandas and pyarrow: pip install pandas pyarrow

# Training set
if [ ! -f "data/vqav2_train.parquet" ]; then
    echo "Converting training set to parquet..."
    # python data_utils.py convert \
    #     --questions "$DATA_DIR/v2_OpenEnded_mscoco_train2014_questions.json" \
    #     --annotations "$DATA_DIR/v2_mscoco_train2014_annotations.json" \
    #     --image-dir "$IMAGES_DIR/train2014" \
    #     --output "data/vqav2_train.parquet"
    echo "[TODO] Run data_utils.py to convert training set"
    echo "  python data_utils.py convert --questions $DATA_DIR/v2_OpenEnded_mscoco_train2014_questions.json --annotations $DATA_DIR/v2_mscoco_train2014_annotations.json --image-dir $IMAGES_DIR/train2014 --output data/vqav2_train.parquet"
else
    echo "Training parquet already exists, skipping..."
fi

# Validation set
if [ ! -f "data/vqav2_val.parquet" ]; then
    echo "Converting validation set to parquet..."
    # python data_utils.py convert \
    #     --questions "$DATA_DIR/v2_OpenEnded_mscoco_val2014_questions.json" \
    #     --annotations "$DATA_DIR/v2_mscoco_val2014_annotations.json" \
    #     --image-dir "$IMAGES_DIR/val2014" \
    #     --output "data/vqav2_val.parquet"
    echo "[TODO] Run data_utils.py to convert validation set"
    echo "  python data_utils.py convert --questions $DATA_DIR/v2_OpenEnded_mscoco_val2014_questions.json --annotations $DATA_DIR/v2_mscoco_val2014_annotations.json --image-dir $IMAGES_DIR/val2014 --output data/vqav2_val.parquet"
else
    echo "Validation parquet already exists, skipping..."
fi

echo ""
echo "===================================="
echo "Setup Instructions"
echo "===================================="
echo ""
echo "This script has TODO markers. To complete setup:"
echo ""
echo "1. Uncomment the download commands above (wget/unzip)"
echo "2. Or manually download from:"
echo "   - VQAv2: https://visualqa.org/download.html"
echo "   - COCO: https://cocodataset.org/#download"
echo ""
echo "3. Install dependencies:"
echo "   pip install pandas pyarrow"
echo ""
echo "4. Run conversion commands:"
echo "   python data_utils.py convert \\"
echo "     --questions $DATA_DIR/v2_OpenEnded_mscoco_train2014_questions.json \\"
echo "     --annotations $DATA_DIR/v2_mscoco_train2014_annotations.json \\"
echo "     --image-dir $IMAGES_DIR/train2014 \\"
echo "     --output data/vqav2_train.parquet"
echo ""
echo "   python data_utils.py convert \\"
echo "     --questions $DATA_DIR/v2_OpenEnded_mscoco_val2014_questions.json \\"
echo "     --annotations $DATA_DIR/v2_mscoco_val2014_annotations.json \\"
echo "     --image-dir $IMAGES_DIR/val2014 \\"
echo "     --output data/vqav2_val.parquet"
echo ""
echo "5. Verify parquet files:"
echo "   python data_utils.py verify --file data/vqav2_train.parquet"
echo "   python data_utils.py verify --file data/vqav2_val.parquet"
echo ""
echo "6. Start training:"
echo "   python train_vqa_agent.py --n-runners 4 --llm-proxy"
echo ""
echo "===================================="
