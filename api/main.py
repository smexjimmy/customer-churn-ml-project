import logging
import os
import time
from pathlib import Path

import yaml
from fastapi import BackgroundTasks, FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from api.schemas import CustomerFeatures, PredictionResponse
from src.models.predict import predict as run_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we're in the right working directory
os.chdir(Path(__file__).parent.parent)

# Load params for feature specs
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Define Prometheus Metrics
API_REQUESTS_TOTAL = Counter(
    "api_requests_total",
    "Total number of HTTP requests processed by the API",
    ["method", "endpoint", "http_status"],
)

API_REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "HTTP request processing latency in seconds",
    ["method", "endpoint"],
)

CHURN_PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Total number of churn predictions",
    ["prediction", "risk_level"],
)

CHURN_PROBABILITY = Histogram(
    "churn_probability",
    "Distribution of churn prediction probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "ML-powered REST API for predicting customer churn. "
        "Trained on Telco Customer Churn dataset."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


def log_prediction_bg(features: dict, prediction_result: dict, params: dict):
    """
    Log prediction request features to CSV in preprocessed format.
    Runs as a FastAPI background task to avoid blocking the HTTP response.
    """
    try:
        from src.models.predict import preprocess_features

        preprocessed_df = preprocess_features(features, params)

        # Append to CSV
        log_path = "data/production_logs.csv"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        header = not os.path.exists(log_path)
        preprocessed_df.to_csv(log_path, mode="a", header=header, index=False)
        logger.debug(f"Logged prediction to {log_path}")
    except Exception as e:
        logger.error(f"Error logging prediction in background: {e}", exc_info=True)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    # Exclude system endpoints from metrics to reduce noise
    path = request.url.path
    if path in ["/metrics", "/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)

    start_time = time.time()
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        latency = time.time() - start_time
        API_REQUESTS_TOTAL.labels(
            method=request.method, endpoint=path, http_status=status_code
        ).inc()
        API_REQUEST_LATENCY.labels(method=request.method, endpoint=path).observe(
            latency
        )

    return response


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Endpoint for Prometheus to scrape API and model metrics."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint — used by Docker and load balancers."""
    return {
        "status": "healthy",
        "version": "0.2.0",
        "phase": "5 - Monitoring & Observability (with Prometheus)",
    }


@app.get("/", tags=["System"])
def root():
    """Root redirect to API docs."""
    return {"message": "Visit /docs for the API documentation."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures, background_tasks: BackgroundTasks):
    """
    Predict customer churn probability.

    Returns:
        - churn_probability: Float between 0 and 1
        - churn_prediction: Boolean (True = likely to churn)
        - risk_level: "low" | "medium" | "high"
        - model_version: Version identifier
    """
    features = customer.model_dump()
    result = run_prediction(features, params)
    result["model_version"] = "0.1.0"

    # Update metrics
    churn_pred = result["churn_prediction"]
    risk = result["risk_level"]
    prob = result["churn_probability"]

    CHURN_PREDICTIONS_TOTAL.labels(prediction=str(churn_pred), risk_level=risk).inc()
    CHURN_PROBABILITY.observe(prob)

    # Log request data asynchronously
    background_tasks.add_task(log_prediction_bg, features, result, params)

    return result
