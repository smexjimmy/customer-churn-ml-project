"""
FastAPI application for Customer Churn Prediction.
Includes health check, root, and prediction endpoints.
"""

import logging
import os
from pathlib import Path

import yaml
from fastapi import FastAPI

from api.schemas import CustomerFeatures, PredictionResponse
from src.models.predict import predict as run_prediction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure we're in the right working directory
os.chdir(Path(__file__).parent.parent)

# Load params for feature specs
with open("params.yaml") as f:
    params = yaml.safe_load(f)

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


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint — used by Docker and load balancers."""
    return {
        "status": "healthy",
        "version": "0.2.0",
        "phase": "2 - Data & Model (with inference)",
    }


@app.get("/", tags=["System"])
def root():
    """Root redirect to API docs."""
    return {"message": "Visit /docs for the API documentation."}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerFeatures):
    """
    Predict customer churn probability.

    Returns:
        - churn_probability: Float between 0 and 1
        - churn_prediction: Boolean (True = likely to churn)
        - risk_level: "low" | "medium" | "high"
        - model_version: Version identifier
    """
    features = customer.dict()
    result = run_prediction(features, params)
    result["model_version"] = "0.1.0"
    return result
