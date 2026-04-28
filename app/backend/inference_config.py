"""Constants aligned with Colab notebook `Untitled3.ipynb` (Optimized Hybrid Model Pipeline)."""

from __future__ import annotations

import os

# image_dataset_from_directory(..., image_size=IMG_SIZE) + Rescaling(1./255) — no preprocess_input
IMG_SIZE: tuple[int, int] = (224, 224)

# Grid search on test set (notebook output): Best Malignant F1, weights sum to 1.0
ENSEMBLE_W_CNN: float = float(os.environ.get("SKIN_CANCER_W_CNN", "0.60"))
ENSEMBLE_W_XGB: float = float(os.environ.get("SKIN_CANCER_W_XGB", "0.40"))
DECISION_THRESHOLD: float = float(os.environ.get("SKIN_CANCER_DECISION_THRESHOLD", "0.15"))

# Binary head: Dense(1, activation="sigmoid") with label_mode="binary", class 1 = malignant
CNN_BINARY_THRESHOLD: float = float(os.environ.get("SKIN_CANCER_CNN_ONLY_THRESHOLD", "0.50"))

GAP_FEATURE_DIM: int = 1280  # MobileNetV2 before top
