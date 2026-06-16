# Twitter Sentiment Analysis Project Report

## 1. Problem Statement

The goal is to classify Twitter text as negative or positive sentiment using a production-friendly classical machine learning pipeline and inference API.

## 2. Dataset Overview

The dataset contains two columns: `sentiment` and `text`. Labels are binary, where `0` means negative and `1` means positive. The raw dataset had 179,995 rows.

## 3. Data Cleaning

Tweet text is cleaned with HTML entity decoding, URL replacement, mention replacement, hashtag word preservation, and whitespace normalization. Emojis, punctuation, casing, and exclamation marks are preserved.

## 4. Duplicate and Conflict Handling

Exact duplicate rows are removed. Normalized duplicate text keys are used to detect conflicting labels, and conflicting duplicate groups are removed. Same-label duplicates are reduced to one row.

## 5. Train/Validation/Test Split

The cleaned dataset contains 173,465 rows. The split is stratified:

- Train rows: 138,772
- Validation rows: 17,346
- Test rows: 17,347

## 6. Baseline Model

The baseline model is TF-IDF plus Logistic Regression. It provides a strong classical ML reference point before more advanced feature comparisons.

## 7. Model Comparison

Several classical models were compared on validation data only, including Logistic Regression, Linear SVC, SGD Classifier, and Complement Naive Bayes.

## 8. Final Model

The best model is Logistic Regression with a real word + character hybrid TF-IDF representation.

## 9. Threshold Tuning

The decision threshold was tuned on validation data only. The selected threshold is 0.47.

## 10. Final Test Results

The final test results are approximately:

- Accuracy: 80.13%
- Macro F1: 79.67%

## 11. API Endpoints

The FastAPI service exposes:

- `GET /`
- `GET /health`
- `GET /ready`
- `GET /model-info`
- `POST /predict`
- `POST /predict-batch`
- `POST /predict-csv`

## 12. Repository Structure

The repository separates source code, scripts, tests, configuration, model artifacts, logs, and generated outputs.

## 13. How to Run

Install dependencies, validate artifacts, and start the API:

```bash
python -m pip install -r requirements.txt
python scripts/13_validate_artifacts.py
python scripts/12_run_api.py
```

## 14. Limitations

- Binary sentiment only
- No neutral class
- Twitter text is noisy
- Some label noise may remain
- Model is classical ML, not transformer-based

## 15. Future Improvements

- Add neutral sentiment support
- Add model monitoring and drift checks
- Add CI/CD workflow
- Add transformer-based experiments after the classical baseline is fully documented
