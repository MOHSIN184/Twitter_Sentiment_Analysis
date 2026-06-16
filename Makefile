install:
	python -m pip install -r requirements.txt

test:
	pytest -q

analyze:
	python scripts/01_analyze_dataset.py

clean:
	python scripts/02_clean_dataset.py

split:
	python scripts/03_split_dataset.py

train:
	python scripts/04_train_baseline.py

evaluate:
	python scripts/08_evaluate_best_model.py

validate:
	python scripts/13_validate_artifacts.py

repo-check:
	python scripts/14_repository_check.py

api:
	python scripts/12_run_api.py

docker-build:
	docker build -t twitter-sentiment-api .

docker-run:
	docker compose up --build
