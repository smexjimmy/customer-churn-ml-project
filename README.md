# Customer Churn Prediction — MLOps Project

A production-grade Machine Learning project demonstrating a complete MLOps lifecycle:
data versioning, experiment tracking, model serving, CI/CD, and drift monitoring.

## Tech Stack

| Layer | Tool |
|-------|------|
| Data Versioning | DVC + MinIO |
| Experiment Tracking | MLflow + PostgreSQL |
| Model Serving | FastAPI + Docker |
| CI/CD | GitHub Actions |
| Drift Monitoring | Evidently AI |
| Metrics Dashboard | Prometheus + Grafana |
| Code Quality | black + flake8 + isort + pre-commit |

## Project Structure

```
ML project/
├── .github/workflows/   # CI/CD pipelines
├── data/raw/            # Raw data (DVC-tracked)
├── data/processed/      # Processed features (DVC-tracked)
├── src/
│   ├── data/            # Ingestion & preprocessing
│   ├── models/          # Training, evaluation, inference
│   └── monitoring/      # Drift detection
├── api/                 # FastAPI prediction service
├── tests/               # Unit & integration tests
├── notebooks/           # EDA notebooks
├── mlflow/              # MLflow server Dockerfile
├── docker-compose.yml   # All services
├── dvc.yaml             # Reproducible pipeline
└── params.yaml          # All hyperparameters
```

## Quick Start

### 1. Clone and set up environment
```bash
git clone <your-repo-url>
cd "ML project"

# Create conda environment
conda env create -f environment.yml
conda activate churn-mlops
```

### 2. Start infrastructure services
```bash
docker-compose up -d
```
- MLflow UI: http://localhost:5000
- MinIO Console: http://localhost:9001
- PostgreSQL: localhost:5432

### 3. Run the ML pipeline
```bash
dvc repro          # ingest → preprocess → train → evaluate
dvc metrics show   # Print model metrics
```

### 4. Serve the model
```bash
uvicorn api.main:app --reload
# API docs: http://localhost:8000/docs
```

### 5. Run tests
```bash
pytest tests/ -v --cov=src
```

## Branch Strategy

```
main        ← production-ready code only
develop     ← integration branch
feature/*   ← individual feature branches
```

## MLOps Phases

- [x] **Phase 1** — Foundation (Git, Docker, DVC, pre-commit)
- [ ] **Phase 2** — Data & Model (pipeline, MLflow tracking)
- [ ] **Phase 3** — Serving (FastAPI, Dockerized API)
- [ ] **Phase 4** — CI/CD (GitHub Actions)
- [ ] **Phase 5** — Monitoring (Evidently, Prometheus, Grafana)

## License

MIT
