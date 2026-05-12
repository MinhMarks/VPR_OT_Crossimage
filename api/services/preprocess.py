"""Image preprocessing pipeline matching eval.py's input_transform()."""

import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import HTTPException, status

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Default inference size (must match export_model.py)
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)   # (H, W)

# Max upload size: 10 MB
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def preprocess_image(
    image_bytes: bytes,
    image_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Preprocess raw image bytes into a normalized float32 numpy array.

    Pipeline:
        1. Decode bytes → PIL RGB
        2. Resize to image_size using Pillow
        3. Normalize to [0, 1] using Numpy
        4. Apply ImageNet mean/std normalization using Numpy
        5. Transpose to [C, H, W] and add batch dim → [1, C, H, W]

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        image_size:  Target (H, W). Defaults to DEFAULT_IMAGE_SIZE.

    Returns:
        numpy array of shape [1, 3, H, W], dtype float32

    Raises:
        HTTPException 400 if image cannot be decoded or is too large.
    """
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Image too large ({len(image_bytes)//1024} KB). Max: {MAX_IMAGE_BYTES//1024} KB.",
        )

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot decode image: {str(e)}",
        )

    target_size = image_size or DEFAULT_IMAGE_SIZE
    img = img.resize(
        (target_size[1], target_size[0]),   # PIL takes (W, H)
        Image.Resampling.BILINEAR,
    )

    arr = np.array(img, dtype=np.float32) / 255.0         # [H, W, 3]  in [0,1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD            # normalize
    arr = arr.transpose(2, 0, 1)                           # [3, H, W]
    arr = np.expand_dims(arr, axis=0)                      # [1, 3, H, W]

    return arr
