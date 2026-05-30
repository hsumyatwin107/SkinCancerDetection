"""Constants aligned with Colab notebook `Untitled3.ipynb` (Optimized Hybrid Model Pipeline)."""

from __future__ import annotations

import os

IMG_SIZE: tuple[int, int] = (224, 224)

ENSEMBLE_W_CNN: float = float(os.environ.get("SKIN_CANCER_W_CNN", "0.25"))
ENSEMBLE_W_XGB: float = float(os.environ.get("SKIN_CANCER_W_XGB", "0.20"))
DECISION_THRESHOLD: float = float(os.environ.get("SKIN_CANCER_DECISION_THRESHOLD", "0.15"))

CNN_BINARY_THRESHOLD: float = float(os.environ.get("SKIN_CANCER_CNN_ONLY_THRESHOLD", "0.50"))

GAP_FEATURE_DIM: int = 1280

ENSEMBLE_W_VIT: float = float(os.environ.get("SKIN_CANCER_W_VIT", "0.55"))
VIT_THRESHOLD: float = float(os.environ.get("SKIN_CANCER_VIT_THRESHOLD", "0.42"))
