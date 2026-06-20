# ============================================================
# Customer Churn MLOps — Automation Makefile
# ============================================================

.PHONY: install format lint test train docker-build docker-up docker-down clean

PYTHON_ENV_DIR = C:/Users/HP/anaconda3/envs/churn-mlops
PYTHON = $(PYTHON_ENV_DIR)/python.exe
PIP = $(PYTHON_ENV_DIR)/Scripts/pip.exe
pytest = $(PYTHON_ENV_DIR)/Scripts/pytest.exe
black = $(PYTHON_ENV_DIR)/Scripts/black.exe
isort = $(PYTHON_ENV_DIR)/Scripts/isort.exe
flake8 = $(PYTHON_ENV_DIR)/Scripts/flake8.exe
dvc = $(PYTHON_ENV_DIR)/Scripts/dvc.exe

install:
	$(PIP) install -r requirements.txt

format:
	$(black) src/ tests/ api/
	$(isort) src/ tests/ api/

lint:
	$(black) --check src/ tests/ api/
	$(isort) --check-only src/ tests/ api/
	$(flake8) src/ tests/ api/

test:
	$(PYTHON) -m pytest tests/ -v

train:
	PYTHONIOENCODING=utf-8 PYTHONUTF8=1 MLFLOW_TRACKING_URI=http://localhost:5000 MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 AWS_ACCESS_KEY_ID=minioadmin AWS_SECRET_ACCESS_KEY=minioadmin $(dvc) repro

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

clean:
	rm -rf .pytest_cache src/__pycache__ api/__pycache__ tests/__pycache__ .coverage coverage.xml
