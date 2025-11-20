"""Utilities for multimodal model support in VERL training.

This module provides utilities for handling multimodal position embeddings,
specifically for models that use Multi-dimensional Rotary Position Embedding (M-RoPE).

Supported Models:
-----------------
- Qwen2-VL (Qwen2VLImageProcessor)
- Qwen3-VL (Qwen3VLImageProcessor)
- GLM4V (Glm4vImageProcessor)

Adding Support for New Models:
------------------------------
To add support for a new multimodal model that uses M-RoPE:

1. Add the image processor class name to `mrope_processors` list in `is_mrope_model()`
2. Add the corresponding `get_rope_index` import in `compute_mrope_position_ids()`
3. The model's VERL implementation should be in `verl/models/transformers/<model_name>.py`
"""

from typing import Any, Dict, List, Optional

import torch

__all__ = [
    "is_mrope_model",
    "get_image_grid_thw",
    "compute_mrope_position_ids",
]


def is_mrope_model(processor: Any) -> bool:
    """
    Check if the processor belongs to a model that requires M-RoPE position embeddings.

    M-RoPE (Multi-dimensional Rotary Position Embedding) is used by multimodal models
    like Qwen2-VL, Qwen3-VL, and GLM4V to encode spatial and temporal positions of
    vision tokens.

    Args:
        processor: The HuggingFace processor for the model.

    Returns:
        True if the model requires M-RoPE position embeddings, False otherwise.

    Note:
        To add support for a new model, add its image processor class name to
        the `mrope_processors` list below.
    """
    if processor is None:
        return False

    if not hasattr(processor, "image_processor"):
        return False

    image_processor_name = processor.image_processor.__class__.__name__

    # List of image processors that require M-RoPE
    # To add a new model: append its image processor class name here
    mrope_processors = [
        "Qwen2VLImageProcessor",  # Qwen2-VL
        "Qwen3VLImageProcessor",  # Qwen3-VL
        "Glm4vImageProcessor",    # GLM4V
    ]

    return any(name in image_processor_name for name in mrope_processors)


def get_image_grid_thw(
    processor: Any,
    original_sample: Dict[str, Any],
    image_keys: Optional[List[str]] = None,
    image_base_dir: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """
    Process images from a sample and return image_grid_thw for M-RoPE computation.

    This function loads images from the sample, processes them with the model's
    processor, and extracts the image_grid_thw tensor which describes the
    temporal (T), height (H), and width (W) dimensions of the processed image grid.

    Args:
        processor: The HuggingFace processor for the model.
        original_sample: The original sample dict containing image path or data.
        image_keys: List of keys to try for finding image data in the sample.
            Defaults to ["images", "image", "image_path"].
        image_base_dir: Base directory for resolving relative image paths.
            If None, paths are used as-is.

    Returns:
        image_grid_thw tensor if images exist and processing succeeds, None otherwise.
        The tensor has shape (num_images, 3) where each row is [T, H, W].
    """
    import os

    if processor is None:
        return None

    if image_keys is None:
        image_keys = ["images", "image", "image_path"]

    # Try different image keys that might be in the sample
    image_data = None
    for key in image_keys:
        if key in original_sample and original_sample[key]:
            image_data = original_sample[key]
            break

    if image_data is None:
        return None

    def resolve_path(path: str) -> str:
        """Resolve image path, handling relative paths with base_dir."""
        if os.path.isabs(path):
            return path
        if image_base_dir is not None:
            return os.path.join(image_base_dir, path)
        return path

    try:
        from PIL import Image

        # Handle different image formats
        images: List[Image.Image] = []
        if isinstance(image_data, str):
            # Single image path
            resolved_path = resolve_path(image_data)
            images = [Image.open(resolved_path).convert("RGB")]
        elif isinstance(image_data, list):
            # List of image paths or PIL images
            for img in image_data:
                if isinstance(img, str):
                    resolved_path = resolve_path(img)
                    images.append(Image.open(resolved_path).convert("RGB"))
                elif isinstance(img, Image.Image):
                    images.append(img.convert("RGB"))
                elif isinstance(img, dict) and "image" in img:
                    # Format like {"type": "image", "image": "path"}
                    resolved_path = resolve_path(img["image"])
                    images.append(Image.open(resolved_path).convert("RGB"))
        elif hasattr(image_data, "convert"):
            # PIL Image
            images = [image_data.convert("RGB")]

        if not images:
            return None

        # Use processor to get image_grid_thw
        # We use a dummy text since we only need the image grid info
        model_inputs = processor(
            text=["dummy"],
            images=images,
            return_tensors="pt",
        )
        return model_inputs.get("image_grid_thw")

    except Exception as e:
        print(f"Warning: Failed to process images for mrope: {e}")
        return None


def compute_mrope_position_ids(
    processor: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute 4D position_ids for models using M-RoPE (Multi-dimensional Rotary Position Embedding).

    M-RoPE uses 4 dimensions for position encoding:
    - Dimension 0: Text position IDs (standard sequential positions)
    - Dimension 1: Temporal position (T) for video frames
    - Dimension 2: Height position (H) in the image grid
    - Dimension 3: Width position (W) in the image grid

    Args:
        processor: The HuggingFace processor for the model.
        input_ids: (seq_length,) tensor of token ids for a single sample.
        attention_mask: (seq_length,) tensor indicating valid positions.
        image_grid_thw: Optional tensor of image grid dimensions with shape
            (num_images, 3) where each row is [T, H, W].

    Returns:
        position_ids: (4, seq_length) tensor containing the 4D position embeddings.

    Note:
        To add support for a new model:
        1. Add a new elif branch below to import the model's `get_rope_index` function
        2. The function should be in `verl/models/transformers/<model_name>.py`
        3. It should have signature: get_rope_index(processor, input_ids, image_grid_thw, attention_mask)
    """
    # Import the appropriate get_rope_index based on processor type
    # To add a new model: add an elif branch with the model's get_rope_index import
    image_processor_name = processor.image_processor.__class__.__name__

    if "Qwen3VL" in processor.__class__.__name__:
        from verl.models.transformers.qwen3_vl import get_rope_index
    elif "Glm4v" in image_processor_name:
        from verl.models.transformers.glm4v import get_rope_index
    else:
        # Default to Qwen2-VL
        from verl.models.transformers.qwen2_vl import get_rope_index

    # Get vision position_ids (3, seq_length) for t, h, w
    vision_position_ids = get_rope_index(
        processor,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        attention_mask=attention_mask,
    )  # (3, seq_length)

    # Compute text position_ids (1, seq_length)
    valid_mask = attention_mask.bool()
    text_position_ids = torch.zeros((1, len(input_ids)), dtype=torch.long, device=input_ids.device)
    text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)

    # Concatenate to form (4, seq_length)
    position_ids = torch.cat([text_position_ids, vision_position_ids], dim=0)

    return position_ids
