"""FastAPI skin cancer API — Colab-aligned hybrid CNN + XGBoost (no mock)."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import model_loader
from .inference import run_inference
from . import vit_inference
from .inference_config import CNN_BINARY_THRESHOLD

logger = logging.getLogger("skin_cancer_api")


class PredictResponse(BaseModel):
    model_used: str
    prediction: str
    confidence: float
    cnn_score: float
    xgb_score: float
    combined_score: float
    inference_debug: dict[str, Any] | None = None
    cnn_prediction: str | None = None
    xgb_prediction: str | None = None
    hybrid_prediction: str | None = None
    vit_prediction: str | None = None
    vit_confidence: float | None = None
    vit_score: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_loader.try_load_all()
        logger.info("Models loaded from %s", model_loader.models_dir())
    except Exception as e:
        logger.error("Startup model load failed: %s", e)
    # START VIT INTEGRATION
    try:
        vit_inference.try_load_vit_model()
        if vit_inference.vit_ready():
            logger.info("ViT model loaded from %s", model_loader.models_dir() / vit_inference.VIT_FILENAME)
    except Exception as e:
        logger.error("ViT startup load failed: %s", e)
    # END VIT INTEGRATION
    yield


app = FastAPI(
    title="Skin Cancer Detection API",
    description="Hybrid MobileNetV2 + XGBoost inference aligned with Colab (Rescale 1/255, GAP features, ensemble 0.6/0.4, threshold 0.15).",
    version="5.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _truthy(val: str | None) -> bool:
    if val is None:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


@app.get("/health")
def health() -> dict[str, Any]:
    err = model_loader.startup_error()
    response: dict[str, Any] = {
        "status": "ok" if model_loader.ready() else "degraded",
        "models_ready": model_loader.ready(),
        "models_dir": str(model_loader.models_dir()),
        "error": err,
    }
    # START VIT INTEGRATION
    response["vit_ready"] = vit_inference.vit_ready()
    response["vit_error"] = vit_inference.vit_startup_error()
    return response
    # END VIT INTEGRATION


@app.post("/predict", response_model=PredictResponse)
async def predict(
    image: Annotated[UploadFile, File(description="Skin lesion image (JPG/PNG).")],
    model_type: Annotated[str, Form()] = "cnn",
    debug: Annotated[str | None, Form()] = None,
    cnn_only: Annotated[str | None, Form()] = None,
) -> PredictResponse:
    if not model_loader.ready():
        raise HTTPException(
            status_code=503,
            detail=model_loader.startup_error() or "Models are not loaded.",
        )
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image file (content-type image/*).")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    dbg = _truthy(debug) or os.environ.get("SKIN_CANCER_INFERENCE_DEBUG", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    cnn_only_flag = _truthy(cnn_only)

    try:
        out = run_inference(raw, model_type, debug=dbg, cnn_only=cnn_only_flag)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    # START VIT INTEGRATION
    def _score_to_label(score: float, threshold: float) -> str:
        return "malignant" if score >= threshold else "benign"

    cnn_thr = CNN_BINARY_THRESHOLD
    cnn_pred = _score_to_label(float(out["cnn_score"]), cnn_thr)
    xgb_pred = _score_to_label(float(out["xgb_score"]), cnn_thr)
    hybrid_pred = str(out["prediction"])

    vit_pred: str | None = None
    vit_conf: float | None = None
    vit_scr: float | None = None
    if vit_inference.vit_ready():
        try:
            vit_out = vit_inference.predict_vit(raw)
            vit_pred = str(vit_out["vit_prediction"])
            vit_conf = float(vit_out["vit_confidence"])
            vit_scr = float(vit_out["vit_score"])
        except Exception as e:
            logger.exception("ViT predict failed: %s", e)
    # END VIT INTEGRATION

    return PredictResponse(
        model_used=out["model_used"],
        prediction=out["prediction"],
        confidence=float(out["confidence"]),
        cnn_score=float(out["cnn_score"]),
        xgb_score=float(out["xgb_score"]),
        combined_score=float(out["combined_score"]),
        inference_debug=out.get("inference_debug") if dbg else None,
        # START VIT INTEGRATION
        cnn_prediction=cnn_pred,
        xgb_prediction=xgb_pred,
        hybrid_prediction=hybrid_pred,
        vit_prediction=vit_pred,
        vit_confidence=vit_conf,
        vit_score=vit_scr,
        # END VIT INTEGRATION
    )
