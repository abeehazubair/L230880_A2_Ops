# VisionGuard — AI vs Real Image Classifier (L230880_A2_Ops)

[![VisionGuard CI](https://github.com/abeehazubair/L230880_A2_Ops/actions/workflows/ci.yml/badge.svg)](https://github.com/abeehazubair/L230880_A2_Ops/actions/workflows/ci.yml)

VisionGuard is a web service that estimates whether an uploaded image is AI-generated or real.
MLOps Assignment 2 — roll number L230880.

| | |
|---|---|
| Code | https://github.com/abeehazubair/L230880_A2_Ops |
| Data (DVC) + experiments (MLflow) + model registry | https://dagshub.com/abeehazubair/L230880_A2_Ops |
| Dataset | [AI Generated Images vs Real Images (Kaggle)](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images) |

## Project layout

| Path | Purpose |
|---|---|
| `src/` | Data audit, EDA, splits, `tf.data` pipeline, models, training with MLflow, evaluation, tuning, registry |
| `app/` | Flask app: inference module, routes, validation, database trail, templates |
| `tests/` | Pytest tests run in CI (no TensorFlow needed) |
| `labs/` | Tool-lab exercises (MLflow, Keras Tuner) |
| `data/raw.dvc` | DVC pointer to the dataset stored on DagsHub |
| `Dockerfile`, `docker-compose.yml` | Web app image and the web + PostgreSQL stack |

## Run with Docker Compose

```bash
cp .env.example .env            # fill in the values
python src/fetch_model.py       # downloads the @champion model once into models/
docker compose up -d --build
docker compose ps               # db and web become "healthy"
```

Open http://localhost:8000 — upload, history and `/health`.

## Development

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements-ci.txt     # app + tests (training needs TensorFlow 2.15.1, MLflow, DVC, ...)
dvc pull                               # dataset from DagsHub
flake8 app src tests labs && pytest -v
```
