"""
FastAPI application for Customer Churn Prediction.
Full implementation in Phase 3.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "ML-powered REST API for predicting customer churn. "
        "Trained on Telco Customer Churn dataset."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health", tags=["System"])
def health_check():
    """Health check endpoint — used by Docker and load balancers."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "phase": "1 - Foundation (API stub)",
    }


@app.get("/", tags=["System"])
def root():
    """Root redirect to API docs."""
    return {"message": "Visit /docs for the API documentation."}


# Phase 3: POST /predict will be added here
