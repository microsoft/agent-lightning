"""
Multimodal support utilities for Agent Lightning.

This module provides helper functions for working with multimodal agents,
particularly for vision-language tasks.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from PIL import Image as PILImage

    ImageType = PILImage.Image
else:
    ImageType = Any

# Check for optional dependencies
_multimodal_deps_available = False
try:
    import requests as _requests
    from PIL import Image as _Image

    _multimodal_deps_available = True
except ImportError:
    _requests = None  # type: ignore[assignment]
    _Image = None  # type: ignore[assignment]


def encode_image_to_base64(image: Union[str, Path, ImageType], max_size: int = 2048) -> str:
    """
    Encode an image to base64 string for multimodal LLM APIs.

    Args:
        image: Image source (file path, URL, or PIL Image object)
        max_size: Maximum dimension for resizing

    Returns:
        Base64 encoded image string with data URI prefix

    Raises:
        ImportError: If PIL (Pillow) is not installed
        TypeError: If image type is not supported

    Examples:
        >>> encoded = encode_image_to_base64("photo.jpg")
        >>> encoded[:30]
        'data:image/jpeg;base64,/9j/4A...'

        >>> from PIL import Image
        >>> img = Image.open("photo.jpg")
        >>> encoded = encode_image_to_base64(img)
    """
    if not _multimodal_deps_available or _requests is None or _Image is None:
        raise ImportError("Multimodal support requires Pillow and requests. Install with: pip install Pillow requests")

    # Load image
    if isinstance(image, (str, Path)):
        image_str = str(image)
        if image_str.startswith(("http://", "https://")):
            response = _requests.get(image_str, timeout=30)
            response.raise_for_status()
            img = _Image.open(BytesIO(response.content))
        else:
            img = _Image.open(image_str)
    elif hasattr(image, "mode"):
        # PIL Image object
        img = image
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # Convert to RGB
    if img.mode == "RGBA":
        background = _Image.new("RGB", img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if needed
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), _Image.Resampling.LANCZOS)

    # Encode
    buffered = BytesIO()
    img.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()

    return f"data:image/jpeg;base64,{img_str}"


def create_image_message(text: str, image: Union[str, Path, ImageType], use_base64: bool = True) -> dict[str, Any]:
    """
    Create an OpenAI-compatible multimodal message.

    Args:
        text: The text prompt/question
        image: Image source (path, URL, or PIL Image)
        use_base64: If True, encode as base64; if False, use URL directly

    Returns:
        Message dict with role="user" and multimodal content

    Examples:
        >>> msg = create_image_message("What's in the image?", "photo.jpg")
        >>> msg["role"]
        'user'
        >>> len(msg["content"])
        2
    """
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]

    if isinstance(image, str) and image.startswith(("http://", "https://")) and not use_base64:
        content.append({"type": "image_url", "image_url": {"url": image}})
    else:
        encoded = encode_image_to_base64(image)
        content.append({"type": "image_url", "image_url": {"url": encoded}})

    return {"role": "user", "content": content}


__all__ = [
    "encode_image_to_base64",
    "create_image_message",
]
