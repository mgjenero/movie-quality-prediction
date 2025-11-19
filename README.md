# 🎬 Good Movie? (IMDb Dataset)
App is currently deployed and you can acces here: https://good-movie.onrender.com/docs#/ <br>IMPORTANT: App also needs some time to wake up so be patient :) 
## Table of Contents
- **Overview**
- **Project layout**
- **Dataset**
- **Features**
- **Modeling & Evaluation**
- **Deployment**
- **Quickstart**

## Overview
This project predicts whether a movie will be well-received (good) based on metadata available on IMDb, such as budget, genre, and reviews.

**Problem type:** Binary classification

## Project layout

- `src/` — Python modules
  - `preprocessing.py` — preprocessing helpers
  - `train.py` — training script (saves `models/model.bin`)
  - `predict.py` — FastAPI prediction service
- `data/` — raw and processed CSVs
- `models/` — trained models
- `notebooks/` — EDA and preprocessing exploration
- `pyproject.toml` / `requirements.txt` — dependencies

## Dataset
- **Source:** IMDb (tabular metadata).
- **Downloaded from:** https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset

**Target:**
```
is_good = True if IMDB rating >= 7 else False
```

## Features
- **Numerical:** `budget`, `gross`, `duration`, `num_critic_for_reviews`, `num_user_for_reviews`, `num_voted_users`, `title_year`
- **Categorical:** `color`, `genres`, `language`, `country`, `content_rating`

If you are interested in feature enginnering look at `src/preprocessing.py` and in `notebooks/preprocessing.ipynb`

## Modeling & Evaluation
- **Model:** Random Forest Classifier
- **Evaluation metrics:** Accuracy (~0.84)

## Deployment
- Served via `FastAPI`.
- Containerized with `Docker` for easy deployment and reproducibility.

## Quickstart
These are example commands — adapt paths/environment as needed.

```bash
pip install uv
uv install
```

Alternatively, using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the FastAPI app locally with uv:

```bash
uv run src/predict.py
```

Alternatively, run with Python:
```bash
python src/predict.py 
```

Build and run with Docker (example):

```bash
docker build -t good_movie .
docker run -it --rm -p 8000:9696 good_movie
```
Dockerized app 

You can test app on following link: 
http://localhost:8000/predict
or manually on  http://127.0.0.1:8000/docs#/
Example of testing snippet:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '  {
    "budget": 237000000,
    "gross": 760505847,
    "duration": 178,
    "num_critic_for_reviews": 723,
    "num_user_for_reviews": 3054,
    "num_voted_users": 886204,
    "title_year": 2009,
    "color": true,
    "genres": "Action|Adventure|Fantasy|Sci-Fi",
    "language": "English",
    "country": "USA",
    "content_rating": "PG-13"
  }'
```