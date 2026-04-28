"""Hybrid CNN + XGBoost inference — matches Colab evaluation (weights, threshold, preprocessing)."""

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
)
from .preprocess import bytes_to_model_input

logger = logging.getLogger("skin_cancer_api")


def _xgb_p_malignant_and_margin(features_1d: np.ndarray) -> tuple[float, float]:
    """XGBoost binary:logistic: predict() is P(malignant); output_margin=True gives raw margin."""
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
    """
    Colab test evaluation:
      cnn_prob = best_fine_tuned_model.predict(test_ds).ravel()  # P(malignant)
      xgb_prob = xgb_model.predict_proba(X_test_features)[:, 1]
      combined_prob = w_cnn * cnn_prob + w_xgb * xgb_prob
      y_pred = (combined_prob >= best_threshold).astype(int)  # 1 = malignant

    Default: w_cnn=0.60, w_xgb=0.40, threshold=0.15 (notebook grid search).

    model_type (cnn/vit) is accepted for UI compatibility; both use the same hybrid unless
    cnn_only=True (optional ablation: sigmoid head only, threshold 0.5).
    """
    requested_model_type = (model_type or "").strip().lower()
    if not model_loader.ready():
        raise RuntimeError(model_loader.startup_error() or "Models not loaded")

    cnn = model_loader.cnn_model
    feat_model = model_loader.feature_extractor
    labels = model_loader.labels_by_index
    assert cnn is not None and feat_model is not None and labels is not None

    batch = bytes_to_model_input(image_bytes)

    cnn_out = cnn.predict(batch, verbose=0)
    cnn_raw = float(np.asarray(cnn_out).reshape(-1)[0])
    cnn_p_malignant = cnn_raw  # Dense(1, sigmoid) with class 1 = malignant

    gap = feat_model.predict(batch, verbose=0)
    gap_np = np.asarray(gap, dtype=np.float32)
    if gap_np.ndim != 2 or gap_np.shape[1] != 1280:
        raise ValueError(f"Expected GAP features (1, 1280), got {gap_np.shape}")

    xgb_p_malignant, xgb_margin = _xgb_p_malignant_and_margin(gap_np[0])

    if cnn_only:
        combined = cnn_p_malignant
        thr = CNN_BINARY_THRESHOLD
        model_used = "cnn"
    else:
        w_c = ENSEMBLE_W_CNN
        w_x = ENSEMBLE_W_XGB
        if abs(w_c + w_x - 1.0) > 1e-3:
            logger.warning("Ensemble weights do not sum to 1.0: %s + %s", w_c, w_x)
        combined = w_c * cnn_p_malignant + w_x * xgb_p_malignant
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
        "combined_score": round(combined, 6),
    }

    if debug:
        result["inference_debug"] = {
            "requested_model_type": requested_model_type or None,
            "cnn_raw_output": cnn_raw,
            "cnn_p_malignant": cnn_p_malignant,
            "xgb_raw_output": xgb_margin,
            "xgb_p_malignant": xgb_p_malignant,
            "combined_score": combined,
            "decision_threshold": thr,
            "ensemble_w_cnn": ENSEMBLE_W_CNN,
            "ensemble_w_xgb": ENSEMBLE_W_XGB,
            "final_label": final_label,
            "cnn_only_mode": cnn_only,
        }

    return result
