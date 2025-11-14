#!/usr/bin/env python3
"""
快速创建调试用的 VQA 数据集
使用简单的生成图片和问答对，用于验证训练流程
"""

import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def create_test_image(image_path: str, content: str, color: str):
    """创建一个简单的测试图片"""
    # 创建 640x480 的图片
    img = Image.new('RGB', (640, 480), color=color)
    draw = ImageDraw.Draw(img)

    # 在图片上绘制文本
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
    except:
        font = ImageFont.load_default()

    # 在中心绘制内容
    bbox = draw.textbbox((0, 0), content, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((640 - text_width) // 2, (480 - text_height) // 2)
    draw.text(position, content, fill='white', font=font)

    img.save(image_path)
    print(f"Created image: {image_path}")

def create_debug_dataset(output_dir: Path, num_samples: int = 10):
    """创建调试用的 VQA 数据集"""
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images" / "val2014"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 定义测试样本
    test_samples = [
        {"content": "CAT", "color": "brown", "question": "What animal is shown?", "answer": "cat"},
        {"content": "DOG", "color": "green", "question": "What animal is in the image?", "answer": "dog"},
        {"content": "BIRD", "color": "blue", "question": "What kind of animal is this?", "answer": "bird"},
        {"content": "FISH", "color": "navy", "question": "What animal do you see?", "answer": "fish"},
        {"content": "RED", "color": "red", "question": "What color is shown?", "answer": "red"},
        {"content": "BLUE", "color": "blue", "question": "What is the color in the image?", "answer": "blue"},
        {"content": "GREEN", "color": "green", "question": "What color do you see?", "answer": "green"},
        {"content": "SUNNY", "color": "yellow", "question": "What is the weather like?", "answer": "sunny"},
        {"content": "RAINY", "color": "gray", "question": "How is the weather?", "answer": "rainy"},
        {"content": "CLOUDY", "color": "lightgray", "question": "What kind of weather is it?", "answer": "cloudy"},
    ]

    # 创建数据
    data = []
    for idx, sample in enumerate(test_samples[:num_samples]):
        # 创建图片路径（模拟 COCO 命名格式）
        image_filename = f"COCO_val2014_{idx:012d}.jpg"
        image_path = images_dir / image_filename

        # 生成图片
        create_test_image(str(image_path), sample["content"], sample["color"])

        # 添加到数据列表
        data.append({
            "id": f"debug_{idx:03d}",
            "image_path": str(image_path.absolute()),
            "question": sample["question"],
            "answer": sample["answer"]
        })

    # 创建 parquet 文件
    df = pd.DataFrame(data)
    parquet_path = output_dir / "vqav2_val_debug.parquet"
    df.to_parquet(parquet_path, index=False)

    print(f"\n{'='*60}")
    print(f"Debug dataset created successfully!")
    print(f"{'='*60}")
    print(f"Parquet file: {parquet_path}")
    print(f"Images directory: {images_dir}")
    print(f"Number of samples: {len(data)}")
    print(f"\nSample data:")
    print(df.head())
    print(f"{'='*60}\n")

    return parquet_path

if __name__ == "__main__":
    import sys

    # 获取输出目录
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(__file__).parent / "data" / "vqav2"

    # 获取样本数量
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    # 创建数据集
    parquet_path = create_debug_dataset(output_dir, num_samples)

    # 验证数据集
    print("Verifying dataset...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} samples")

    # 检查图片是否存在
    missing_images = []
    for idx, row in df.iterrows():
        if not Path(row['image_path']).exists():
            missing_images.append(row['image_path'])

    if missing_images:
        print(f"WARNING: {len(missing_images)} images not found:")
        for img in missing_images[:5]:
            print(f"  - {img}")
    else:
        print("All images found! Dataset is ready for training.")
