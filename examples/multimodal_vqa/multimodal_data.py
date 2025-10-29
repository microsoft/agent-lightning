# Copyright (c) Microsoft. All rights reserved.

"""
Multimodal data processing utilities for Agent Lightning.

This module provides tools for loading, encoding, and processing images for
multimodal agent tasks, particularly for Visual Question Answering (VQA).
"""

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Union, cast

try:
    from PIL import Image
    import requests
except ImportError as e:
    raise ImportError(
        "Multimodal support requires Pillow and requests. "
        "Install with: pip install Pillow requests"
    ) from e


def encode_image_to_base64(image: Union[str, Path, Image.Image], max_size: int = 2048) -> str:
    """
    Encode an image to base64 string for OpenAI API.

    Args:
        image: Image source - can be a file path, URL, or PIL Image object
        max_size: Maximum dimension (width or height) for resizing

    Returns:
        Base64 encoded image string with data URI prefix

    Examples:
        >>> encoded = encode_image_to_base64("chart.png")
        >>> encoded[:30]
        'data:image/jpeg;base64,/9j/4A...'
    """
    # Load image based on input type
    if isinstance(image, (str, Path)):
        image_str = str(image)
        if image_str.startswith(("http://", "https://")):
            # Download from URL
            response = requests.get(image_str, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        else:
            # Load from local path
            img = Image.open(image_str)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert RGBA to RGB if necessary
    if img.mode == "RGBA":
        # Create white background
        background = Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
        img = background
    elif img.mode not in ["RGB", "L"]:
        img = img.convert("RGB")

    # Resize if too large
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Encode to base64
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/jpeg;base64,{img_str}"


def create_image_message(
    text: str,
    image: Union[str, Path, Image.Image],
    use_base64: bool = True
) -> Dict:
    """
    Create an OpenAI-compatible multimodal message with text and image.

    Args:
        text: The text prompt/question
        image: Image source (path, URL, or PIL Image)
        use_base64: If True, encode image as base64; if False, use URL directly

    Returns:
        Dict in OpenAI message format with role="user"

    Examples:
        >>> msg = create_image_message("What's in this image?", "photo.jpg")
        >>> msg["role"]
        'user'
        >>> len(msg["content"])
        2
    """
    content: List[Dict] = [{"type": "text", "text": text}]

    if isinstance(image, str) and image.startswith(("http://", "https://")) and not use_base64:
        # Use URL directly
        content.append({
            "type": "image_url",
            "image_url": {"url": image}
        })
    else:
        # Encode to base64
        encoded = encode_image_to_base64(image)
        content.append({
            "type": "image_url",
            "image_url": {"url": encoded}
        })

    return {"role": "user", "content": content}


def load_vqa_dataset(jsonl_path: Union[str, Path]) -> List[Dict]:
    """
    Load VQA dataset from JSONL file.

    Expected format:
    {
        "id": "vqa_001",
        "image": "data/sample_images/chart.png",
        "question": "What is the highest value?",
        "expected_answer": "42"
    }

    Args:
        jsonl_path: Path to JSONL file

    Returns:
        List of task dictionaries
    """
    tasks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def preprocess_image_for_training(
    image_path: Union[str, Path],
    target_size: tuple = (336, 336)
) -> Image.Image:
    """
    Preprocess image for training with vision models like Qwen-VL.

    Args:
        image_path: Path to image file
        target_size: Target (width, height) for resizing

    Returns:
        Processed PIL Image

    Note:
        This is a placeholder for future Qwen-VL integration.
        Different models may require different preprocessing.
    """
    img = Image.open(image_path)

    # Convert to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize while maintaining aspect ratio
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    # Pad to target size if needed
    if img.size != target_size:
        padded = Image.new("RGB", target_size, (255, 255, 255))
        offset = ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2)
        padded.paste(img, offset)
        img = padded

    return img


if __name__ == "__main__":
    # Example usage
    print("Testing multimodal data utilities...")

    # Test image encoding (using a simple test image)
    try:
        # Create a simple test image
        test_img = Image.new("RGB", (100, 100), color=(73, 109, 137))
        encoded = encode_image_to_base64(test_img)
        print(f"✓ Image encoding works. Encoded length: {len(encoded)}")

        # Test message creation
        msg = create_image_message("Test question", test_img)
        print(f"✓ Message creation works. Message keys: {list(msg.keys())}")

        print("\nAll tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
