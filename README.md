# Skin Cancer Detection — ViT Backend

> Graduation Project | Vision Transformer (ViT) + XGBoost Hybrid Model for Skin Cancer Classification

---

## Overview

This project implements a **Vision Transformer (ViT)** model for binary skin cancer classification (benign vs malignant), integrated into a FastAPI backend as part of a hybrid ensemble system.

The ViT model is combined with an XGBoost classifier to produce a robust ensemble prediction, achieving an overall accuracy of **82.6%** on the HAM10000 dataset.

---

## My Contribution

- Trained and exported the **Vision Transformer (ViT)** model
- Built the **backend integration** to load and serve the ViT model via FastAPI
- Implemented a **hybrid ensemble inference** pipeline combining:
  - Vision Transformer (ViT) — weight: 0.55
  - XGBoost — weight: 0.45
- Used a decision threshold of **0.42** for malignant classification

---

## Model Details

| Property | Value |
|---|---|
| Architecture | Vision Transformer (ViT) |
| Dataset | HAM10000 |
| Task | Binary classification (benign / malignant) |
| Input size | 224 × 224 × 3 |
| Accuracy | 82.6% |
| Decision threshold | 0.42 |

---

## Project Structure

```
SkinCancerDetection/
  app/
    backend/
      inference.py        # Hybrid ensemble inference (ViT + XGBoost)
      inference_config.py # Model weights and thresholds
      model_loader.py     # Loads ViT, CNN, and XGBoost models
      preprocess.py       # Image preprocessing
      main.py             # FastAPI app
    frontend-react/       # React frontend (teammate's contribution)
    models/               # Model files (not tracked in git)
  requirements.txt
  main.py
```

---

## How to Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/hsumyatwin107/SkinCancerDetection.git
cd SkinCancerDetection
```

**2. Set up environment**
```bash
conda create -n skinapp python=3.10
conda activate skinapp
pip install -r requirements.txt
```

**3. Add model files to `app/models/`**
- `best_vit_skin_cancer.h5`
- `best_skin_cancer_cnn.keras`
- `best_skin_cancer_xgb.json`
- `labels.json`

**4. Run the backend**
```bash
uvicorn app.backend.main:app --port 8000
```

**5. Test the API**

Open `http://127.0.0.1:8000/docs` and use the `/predict` endpoint with `model_type=vit`.

---

## API Usage

```bash
curl -X POST 'http://127.0.0.1:8000/predict' \
  -F 'image=@your_image.jpg;type=image/jpeg' \
  -F 'model_type=vit'
```

**Response:**
```json
{
  "model_used": "vit",
  "prediction": "benign",
  "confidence": 0.691052,
  "vit_score": 0.308948,
  "combined_score": 0.308948
}
```

---

## Team

| Name | Contribution |
|---|---|
| Najma Mohamed Mohamud | ViT model training + backend integration |
| Hsu Myat Win | MobileNetV2 CNN + React frontend |

---

## Tech Stack

- Python 3.10
- TensorFlow / Keras
- Vision Transformer (ViT)
- XGBoost
- FastAPI
- Apple M4 (Metal GPU acceleration)
