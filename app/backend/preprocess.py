"""Decode uploaded bytes to a single CNN input batch — Colab eval: RGB, 224×224, divide by 255 once."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .inference_config import IMG_SIZE


def bytes_to_model_input(image_bytes: bytes) -> np.ndarray:
    """
    Match Colab `apply_preprocessing_eval`:
    tf.cast(image, float32) then Rescaling(1/255). No keras.applications.preprocess_input.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
