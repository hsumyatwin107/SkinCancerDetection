from __future__ import annotations
import logging
from typing import Any
import numpy as np
import xgboost as xgb
from . import model_loader
from .inference_config import (
    CNN_BINARY_THRESHOLD,
    DECISION_THRESHOLD,
    ENSEMBLE_W_CNN,
    ENSEMBLE_W_XGB,
    ENSEMBLE_W_VIT,
    VIT_THRESHOLD,
)
from .preprocess import bytes_to_model_input

logger = logging.getLogger("skin_cancer_api")


def run_inference(image_bytes, model_type, *, debug=False, cnn_only=False):
    requested_model_type = (model_type or "").strip().lower()
    if not model_loader.ready():
        raise RuntimeError(model_loader.startup_error() or "Models not loaded")

    cnn = model_loader.cnn_model
    feat_model = model_loader.feature_extractor
    vit = model_loader.vit_model
    labels = model_loader.labels_by_index
    assert labels is not None

    batch = bytes_to_model_input(image_bytes)

    cnn_p_malignant = 0.0
    if cnn is not None:
        cnn_out = cnn.predict(batch, verbose=0)
        cnn_p_malignant = float(np.asarray(cnn_out).reshape(-1)[0])

    xgb_p_malignant = 0.0
    if model_loader.xgb_booster is not None and feat_model is not None:
        gap = feat_model.predict(batch, verbose=0)
        gap_np = np.asarray(gap, dtype=np.float32)
        row = gap_np[0].reshape(1, -1)
        dm = xgb.DMatrix(row)
        xgb_p_malignant = float(model_loader.xgb_booster.predict(dm)[0])

    vit_p_malignant = 0.0
    if vit is not None:
        vit_out = vit.predict(batch, verbose=0)
        vit_p_malignant = float(np.asarray(vit_out).reshape(-1)[-1])

    if requested_model_type == "vit" and vit is not None:
        combined = vit_p_malignant
        thr = VIT_THRESHOLD
        model_used = "vit"
    elif cnn_only:
        combined = cnn_p_malignant
        thr = CNN_BINARY_THRESHOLD
        model_used = "cnn"
    else:
        w_c = ENSEMBLE_W_CNN if cnn is not None else 0.0
        w_x = ENSEMBLE_W_XGB if model_loader.xgb_booster is not None else 0.0
        w_v = ENSEMBLE_W_VIT if vit is not None else 0.0
        total = w_c + w_x + w_v or 1.0
        combined = (w_c * cnn_p_malignant + w_x * xgb_p_malignant + w_v * vit_p_malignant) / total
        thr = DECISION_THRESHOLD
        model_used = "hybrid"

    malignant = combined >= thr
    final_label = labels[1 if malignant else 0]
    confidence = float(combined if malignant else (1.0 - combined))

    return {
        "model_used": model_used,
        "prediction": final_label,
        "confidence": round(confidence, 6),
        "cnn_score": round(cnn_p_malignant, 6),
        "xgb_score": round(xgb_p_malignant, 6),
        "vit_score": round(vit_p_malignant, 6),
        "combined_score": round(combined, 6),
    }