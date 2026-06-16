FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p outputs/logs outputs/predictions

EXPOSE 8000

CMD ["sh", "-c", "python scripts/13_validate_artifacts.py && uvicorn sentiment_pipeline.api.app:app --host 0.0.0.0 --port 8000"]
