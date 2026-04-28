# Skin Cancer Detection (Hybrid Inference)

Inference project for skin lesion prediction using a hybrid CNN + XGBoost pipeline.

## Stack

- FastAPI backend for inference API
- React + Vite frontend for UI
- TensorFlow `.keras` CNN model + XGBoost model for prediction
- OpenCV + NumPy for image preprocessing

## Clean project structure

```text
main.py
app/
  backend/
    main.py
    model_loader.py
    utils.py
  frontend-react/
    src/
    package.json
models/
  best_skin_cancer_cnn.keras
  best_skin_cancer_xgb.json
  labels.json
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run backend

```bash
uvicorn main:app --reload
```

- Health check: `http://127.0.0.1:8000/health`
- API docs: `http://127.0.0.1:8000/docs`

## Run frontend

```bash
cd app/frontend-react
npm install
npm run dev
```

Frontend runs at `http://127.0.0.1:5173`.

## Prediction endpoint

`POST /predict` (multipart form-data):

- field name: `image`
- accepted input: JPG/PNG image file
- output:
  - `prediction` (`benign`, `malignant`)
  - `confidence` (0.0 to 1.0)
  - `cnn_score` (0.0 to 1.0)
  - `xgb_score` (0.0 to 1.0)
  - `combined_score` (0.0 to 1.0)

## Notes

- Required model files:
  - `models/best_skin_cancer_cnn.keras`
  - `models/best_skin_cancer_xgb.json`
  - `models/labels.json`
- Ensemble rule:
  - `combined = (cnn_prob * 0.60) + (xgb_prob * 0.40)`
  - `combined >= 0.15 => malignant`, else `benign`
- This application is for educational purposes and not a medical diagnosis tool.
