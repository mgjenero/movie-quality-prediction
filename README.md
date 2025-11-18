# 🎬 Movie Quality Prediction (IMDb Dataset)
## Table of Contents
- **Overview**
- **Dataset**
- **Features**
- **Modeling & Evaluation**
- **Deployment**
- **Quickstart**
- **Contributing**

## Overview
Predict whether a movie is "good" or "bad" from tabular IMDb metadata. The target label is `is_good` (1 if rating >= 7, else 0).

**Problem type:** Binary classification



## Dataset
- **Source:** IMDb (tabular metadata)
- **Typical fields:** `title`, `genre`, `director`, `cast`, `runtime`, `budget`, `box_office`, `release_year`, `language`, `country`, `rating`, etc.

**Target:**
```
is_good = 1 if rating >= 7 else 0 s
```

## Features
- **Numerical:** `budget`, `runtime`, `box_office`, `release_year`, etc.
- **Categorical:** `genre`, `director`, `cast`, `language`, `country`

Feature engineering should include sensible encoding for categorical fields, handling missing values, and scaling for numeric features.

## Modeling & Evaluation
- **Candidate models:** Logistic Regression, Random Forest, XGBoost
- **Evaluation metrics:** Accuracy, F1-score (report both to balance precision/recall trade-offs)

Typical workflow:
1. Split data into train/validation/test (stratify on `is_good`).
2. Preprocess features (imputation, encoding, scaling).
3. Train baseline (Logistic Regression), then tree-based models.
4. Evaluate on validation set and report final metrics on the test set.

## Deployment
- Served via `FastAPI`.
- Containerized with `Docker` for easy deployment and reproducibility.

## Quickstart
These are example commands — adapt paths/environment as needed.

Prerequisites:
- Python 3.12+ (recommended)
- `pip`
- `docker` (for containerized deployment)

Install dependencies (if `requirements.txt` exists):
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training script (example):
```bash
python train.py --data data/imdb_metadata.csv --output models/model.pkl
```

Run the API locally (example):
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Build and run with Docker (example):
```bash
docker build -t movie-quality-api .
docker run -p 8000:8000 movie-quality-api
```