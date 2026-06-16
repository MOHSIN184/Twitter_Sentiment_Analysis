# Twitter Sentiment Analysis Pipeline & API

Python | scikit-learn | FastAPI | Docker

## 1. Overview

This repository contains an end-to-end Twitter sentiment analysis project: dataset processing, classical machine learning model training, model comparison, threshold tuning, batch inference, and a production-ready FastAPI service.

The final model classifies text into:

- `0`: negative
- `1`: positive

## 2. Features

- Reproducible data validation, cleaning, deduplication, and splitting
- TF-IDF + classical ML model experiments
- Real word + character hybrid TF-IDF features
- Validation-only decision threshold tuning
- Single text, JSON batch, and CSV batch inference
- FastAPI service with health/readiness checks
- Docker and Docker Compose deployment files
- Artifact validation and GitHub repository readiness checks

## 3. Project Structure

```text
src/sentiment_pipeline/     Core package
src/sentiment_pipeline/api/ FastAPI service
scripts/                    Runnable project scripts
tests/                      Unit and API tests
models/baseline/            Trained model artifacts
outputs/                    Logs, figures, and predictions
data/                       Raw, processed, and sample data locations
```

## 4. Dataset

The raw dataset is not included in the repository. Place it manually at:

```text
data/raw/twitter_sentiment_dataset.csv
```

Expected columns:

- `sentiment`
- `text`

For API-only usage, trained model artifacts are included:

```text
models/baseline/best_model.joblib
models/baseline/decision_threshold.json
```

## 5. Model Performance

Final model:

- Logistic Regression with real word + character hybrid TF-IDF
- Decision threshold: `0.47`
- Final test accuracy: approximately `80.13%`
- Final test macro F1: approximately `79.67%`

## 6. Installation

```bash
python -m pip install -r requirements.txt
```

## 7. Run Full Pipeline

```bash
python scripts/run_full_pipeline.py
```

This runs dataset analysis, cleaning, splitting, baseline training, model comparison, error analysis, threshold tuning, and final evaluation.

## 8. Run API

Validate artifacts first:

```bash
python scripts/13_validate_artifacts.py
```

Start the API:

```bash
python scripts/12_run_api.py
```

Open Swagger docs:

```text
http://127.0.0.1:8000/docs
```

## 9. API Endpoints

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `POST /predict-csv`

Single prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\":\"I love this product!\"}"
```

Batch prediction:

```bash
curl -X POST "http://127.0.0.1:8000/predict-batch" -H "Content-Type: application/json" -d "{\"texts\":[\"I love this product!\",\"This is terrible\"]}"
```

## 10. Batch Prediction

```bash
python scripts/11_batch_predict_csv.py --input data/sample_input.csv --output outputs/predictions/sample_predictions.csv
```

The input CSV must contain a `text` column.

## 11. Docker Usage

Build the image:

```bash
docker build -t twitter-sentiment-api .
```

Run with Docker Compose:

```bash
docker compose up --build
```

## 12. Tests

```bash
pytest -q
```

## 13. Artifact Validation

```bash
python scripts/13_validate_artifacts.py
```

Repository readiness check:

```bash
python scripts/14_repository_check.py
```

## 14. Repository Submission Notes

- Do not commit `.env`.
- Do not commit the raw dataset.
- Generated logs and prediction CSVs are ignored.
- The trained model and threshold are intended to be included for API-only usage.

Useful commands:

```bash
python -m pip install -r requirements.txt
python scripts/13_validate_artifacts.py
python scripts/14_repository_check.py
pytest -q
python scripts/12_run_api.py
```

## 15. Future Work

- Add CI/CD workflow
- Add request metrics and model monitoring
- Add neutral sentiment support
- Add transformer-based experiments after the classical ML pipeline is fully documented
