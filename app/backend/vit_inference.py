"""Isolated Vision Transformer (ViT) inference — separate from CNN + XGBoost hybrid pipeline."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from .inference_config import CNN_BINARY_THRESHOLD, IMG_SIZE

logger = logging.getLogger("skin_cancer_api")

_BACKEND_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BACKEND_DIR.parent / "models"
VIT_FILENAME = "best_vit_skin_cancer.h5"
VIT_LABELS: dict[int, str] = {0: "benign", 1: "malignant"}

vit_model: tf.keras.Model | None = None
_vit_load_error: str | None = None


def load_vit_model() -> None:
    """Load ViT weights from app/models/best_vit_skin_cancer.h5."""
    global vit_model, _vit_load_error

    vit_path = _MODELS_DIR / VIT_FILENAME
    if not vit_path.is_file():
        raise FileNotFoundError(f"Missing ViT weights: {vit_path}")

    logger.info("Loading ViT model from %s", vit_path)
    loaded = tf.keras.models.load_model(str(vit_path), compile=False, safe_mode=False)
    if not isinstance(loaded, tf.keras.Model):
        raise TypeError("Loaded ViT is not a tf.keras.Model")
    vit_model = loaded

    mi = tuple(int(x) for x in vit_model.input_shape[1:4])
    if mi != (224, 224, 3):
        logger.warning("ViT input shape is %s (expected (224,224,3)).", mi)

    _vit_load_error = None


def try_load_vit_model() -> None:
    """Load ViT on startup; record error without affecting CNN/XGBoost loading."""
    global vit_model, _vit_load_error
    try:
        load_vit_model()
    except Exception as e:
        vit_model = None
        _vit_load_error = str(e)
        logger.exception("ViT model load failed: %s", e)


def vit_ready() -> bool:
    return vit_model is not None


def vit_startup_error() -> str | None:
    return _vit_load_error


def preprocess_for_vit(image_bytes: bytes) -> np.ndarray:
    """
    ViT preprocessing pipeline (isolated from CNN bytes_to_model_input):
    RGB, 224×224, float32, scale to [0, 1] via /255.
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def predict_vit(image_bytes: bytes, *, threshold: float = CNN_BINARY_THRESHOLD) -> dict[str, Any]:
    """
    Run ViT binary classifier: sigmoid output = P(malignant).
    Labels: 0=benign, 1=malignant (Benign / Malignant).
    """
    if vit_model is None:
        raise RuntimeError(_vit_load_error or "ViT model not loaded")

    batch = preprocess_for_vit(image_bytes)
    raw_out = vit_model.predict(batch, verbose=0)
    vit_p_malignant = float(np.asarray(raw_out).reshape(-1)[0])

    malignant = vit_p_malignant >= threshold
    final_label = VIT_LABELS[1 if malignant else 0]
    confidence = float(vit_p_malignant if malignant else (1.0 - vit_p_malignant))

    return {
        "vit_prediction": final_label,
        "vit_confidence": round(confidence, 6),
        "vit_score": round(vit_p_malignant, 6),
    }
