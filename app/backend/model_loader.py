"""Load models."""
from __future__ import annotations
import json
import h5py
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

cnn_model = None
xgb_booster = None
feature_extractor = None
labels_by_index = None
vit_model = None
_load_error = None

def models_dir():
    return _MODELS_DIR

def _require_labels(data):
    if data.get("0") != "benign" or data.get("1") != "malignant":
        raise ValueError("Invalid labels")
    return {0: "benign", 1: "malignant"}

def _find_gap_feature_extractor(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            return tf.keras.Model(inputs=model.input, outputs=layer.output)
    raise ValueError("No GAP layer found.")

def _load_vit(path):
    with h5py.File(path, 'r+') as f:
        mc = f.attrs.get('model_config')
        if isinstance(mc, bytes):
            mc = mc.decode('utf-8')
        cs = json.dumps(json.loads(mc)).replace('"quantization_config": null,', '').replace(', "quantization_config": null', '')
        f.attrs['model_config'] = cs.encode('utf-8')
    return tf.keras.models.load_model(path, compile=False)

def load_all():
    global cnn_model, xgb_booster, feature_extractor, labels_by_index, _load_error, vit_model
    cnn_path = _MODELS_DIR / CNN_FILENAME
    xgb_path = _MODELS_DIR / XGB_FILENAME
    labels_path = _MODELS_DIR / LABELS_FILENAME
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing labels: {labels_path}")
    with open(labels_path, encoding="utf-8") as f:
        raw = json.load(f)
    labels_by_index = _require_labels(raw)
    if cnn_path.is_file():
        logger.info("Loading CNN from %s", cnn_path)
        cnn_model = tf.keras.models.load_model(str(cnn_path), compile=False, safe_mode=False)
        feature_extractor = _find_gap_feature_extractor(cnn_model)
    else:
        logger.warning("CNN not found, skipping")
    if xgb_path.is_file():
        booster = xgb.Booster()
        booster.load_model(str(xgb_path))
        xgb_booster = booster
    else:
        logger.warning("XGBoost not found, skipping")
    vit_path = _MODELS_DIR / "best_vit_skin_cancer.h5"
    if vit_path.is_file():
        logger.info("Loading ViT from %s", vit_path)
        vit_model = _load_vit(str(vit_path))
        logger.info("ViT loaded.")
    else:
        logger.warning("ViT not found")
    _load_error = None

def try_load_all():
    global cnn_model, xgb_booster, feature_extractor, labels_by_index, _load_error, vit_model
    try:
        load_all()
    except Exception as e:
        cnn_model = None
        xgb_booster = None
        feature_extractor = None
        labels_by_index = None
        vit_model = None
        _load_error = str(e)
        logger.exception("Model load failed: %s", e)
        raise

def ready():
    return labels_by_index is not None

def startup_error():
    return _load_error
