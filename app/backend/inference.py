"""Hybrid CNN + XGBoost + ViT inference."""
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


def _xgb_p_malignant_and_margin(features_1d: np.ndarray) -> tuple[float, float]:
    booster = model_loader.xgb_booster
    if booster is None:
        raise RuntimeError("XGBoost booster not loaded")
    row = np.asarray(features_1d, dtype=np.float32).reshape(1, -1)
    dm = xgb.DMatrix(row)
    p_mal = float(booster.predict(dm)[0])
    margin = float(booster.predict(dm, output_margin=True)[0])
    return p_mal, margin


def run_inference(
    image_bytes: bytes,
    model_type: str,
    *,
    debug: bool = False,
    cnn_only: bool = False,
) -> dict[str, Any]:
    requested_model_type = (model_type or "").strip().lower()
    if not model_loader.ready():
        raise RuntimeError(model_loader.startup_error() or "Models not loaded")

    cnn = model_loader.cnn_model
    feat_model = model_loader.feature_extractor
    vit = model_loader.vit_model
    labels = model_loader.labels_by_index
    assert cnn is not None and feat_model is not None and labels is not None

    batch = bytes_to_model_input(image_bytes)

    cnn_out = cnn.predict(batch, verbose=0)
    cnn_p_malignant = float(np.asarray(cnn_out).reshape(-1)[0])

    gap = feat_model.predict(batch, verbose=0)
    gap_np = np.asarray(gap, dtype=np.float32)
    if gap_np.ndim != 2 or gap_np.shape[1] != 1280:
        raise ValueError(f"Expected GAP features (1, 1280), got {gap_np.shape}")

    xgb_p_malignant, xgb_margin = _xgb_p_malignant_and_margin(gap_np[0])

    # ViT prediction
    vit_p_malignant = 0.0
    if vit is not None:
        vit_out = vit.predict(batch, verbose=0)
        vit_p_malignant = float(np.asarray(vit_out).reshape(-1)[-1])

    if cnn_only:
        combined = cnn_p_malignant
        thr = CNN_BINARY_THRESHOLD
        model_used = "cnn"
    elif requested_model_type == "vit" and vit is not None:
        combined = vit_p_malignant
        thr = VIT_THRESHOLD
        model_used = "vit"
    else:
        w_c = ENSEMBLE_W_CNN
        w_x = ENSEMBLE_W_XGB
        w_v = ENSEMBLE_W_VIT if vit is not None else 0.0
        total = w_c + w_x + w_v
        combined = (w_c * cnn_p_malignant + w_x * xgb_p_malignant + w_v * vit_p_malignant) / total
        thr = DECISION_THRESHOLD
        model_used = "hybrid"

    malignant = combined >= thr
    final_idx = 1 if malignant else 0
    final_label = labels[final_idx]
    confidence = float(combined if malignant else (1.0 - combined))

    result: dict[str, Any] = {
        "model_used": model_used,
        "prediction": final_label,
        "confidence": round(confidence, 6),
        "cnn_score": round(cnn_p_malignant, 6),
        "xgb_score": round(xgb_p_malignant, 6),
        "vit_score": round(vit_p_malignant, 6),
        "combined_score": round(combined, 6),
    }

    if debug:
        result["inference_debug"] = {
            "requested_model_type": requested_model_type or None,
            "cnn_p_malignant": cnn_p_malignant,
            "xgb_p_malignant": xgb_p_malignant,
            "vit_p_malignant": vit_p_malignant,
            "combined_score": combined,
            "decision_threshold": thr,
            "ensemble_w_cnn": ENSEMBLE_W_CNN,
            "ensemble_w_xgb": ENSEMBLE_W_XGB,
            "ensemble_w_vit": ENSEMBLE_W_VIT,
            "final_label": final_label,
            "cnn_only_mode": cnn_only,
        }

    return result 