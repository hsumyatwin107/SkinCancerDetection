"""Load CNN (.keras), XGBoost (.json), and labels — Colab-aligned paths and shapes."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import xgboost as xgb

from .inference_config import GAP_FEATURE_DIM

logger = logging.getLogger("skin_cancer_api")

_BACKEND_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BACKEND_DIR.parent / "models"

CNN_FILENAME = "best_skin_cancer_cnn.keras"
XGB_FILENAME = "best_skin_cancer_xgb.json"
LABELS_FILENAME = "labels.json"

cnn_model: tf.keras.Model | None = None
xgb_booster: xgb.Booster | None = None
feature_extractor: tf.keras.Model | None = None
labels_by_index: dict[int, str] | None = None
_load_error: str | None = None


def models_dir() -> Path:
    return _MODELS_DIR


def _require_labels(data: dict[str, Any]) -> dict[int, str]:
    """Strict mapping from Colab export: 0=benign, 1=malignant."""
    if data.get("0") != "benign" or data.get("1") != "malignant":
        raise ValueError(
            'labels.json must be exactly {"0": "benign", "1": "malignant"} '
            f"(got 0={data.get('0')!r}, 1={data.get('1')!r})."
        )
    return {0: "benign", 1: "malignant"}


def _find_gap_feature_extractor(model: tf.keras.Model) -> tf.keras.Model:
    """
    Colab: outputs=best_fine_tuned_model.layers[2].output  # GlobalAveragePooling2D
    Top-level layers: [0] Input, [1] MobileNetV2 functional, [2] GAP, [3] Dropout, [4] Dense.
    Resolve by layer type (no blind negative indexing on nested graphs).
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return tf.keras.Model(inputs=model.input, outputs=layer.output, name="gap_feature_extractor")
    raise ValueError("No top-level GlobalAveragePooling2D found (Colab expects GAP after MobileNetV2).")


def load_all() -> None:
    """Load CNN, XGBoost booster, labels, and build GAP feature extractor."""
    global cnn_model, xgb_booster, feature_extractor, labels_by_index, _load_error

    cnn_path = _MODELS_DIR / CNN_FILENAME
    xgb_path = _MODELS_DIR / XGB_FILENAME
    labels_path = _MODELS_DIR / LABELS_FILENAME

    if not cnn_path.is_file():
        raise FileNotFoundError(f"Missing CNN weights: {cnn_path}")
    if not xgb_path.is_file():
        raise FileNotFoundError(f"Missing XGBoost model: {xgb_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels: {labels_path}")

    with open(labels_path, encoding="utf-8") as f:
        raw = json.load(f)
    labels_by_index = _require_labels(raw)

    logger.info("Loading Keras model from %s", cnn_path)
    loaded = tf.keras.models.load_model(str(cnn_path), compile=False, safe_mode=False)
    if not isinstance(loaded, tf.keras.Model):
        raise TypeError("Loaded CNN is not a tf.keras.Model")
    cnn_model = loaded
    feature_extractor = _find_gap_feature_extractor(cnn_model)

    mi = tuple(int(x) for x in cnn_model.input_shape[1:4])
    if mi != (224, 224, 3):
        logger.warning("CNN input shape is %s (Colab uses (224,224,3)).", mi)

    logger.info("Loading XGBoost booster from %s", xgb_path)
    booster = xgb.Booster()
    booster.load_model(str(xgb_path))
    xgb_booster = booster

    dummy = xgb.DMatrix(np.zeros((1, GAP_FEATURE_DIM), dtype=np.float32))
    out = xgb_booster.predict(dummy)
    if out.shape != (1,) and out.size != 1:
        raise ValueError(f"Unexpected XGBoost predict shape: {out.shape}")

    _load_error = None


def try_load_all() -> None:
    global cnn_model, xgb_booster, feature_extractor, labels_by_index, _load_error
    try:
        load_all()
    except Exception as e:
        cnn_model = None
        xgb_booster = None
        feature_extractor = None
        labels_by_index = None
        _load_error = str(e)
        logger.exception("Model load failed: %s", e)
        raise


def ready() -> bool:
    return (
        cnn_model is not None
        and xgb_booster is not None
        and feature_extractor is not None
        and labels_by_index is not None
    )


def startup_error() -> str | None:
    return _load_error
