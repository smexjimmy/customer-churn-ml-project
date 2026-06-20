"""
Model Inference Module
=======================
Loads the trained model and runs predictions.
Used by the FastAPI service in Phase 3.
"""

import logging
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model — avoids reloading on every request."""
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict(features: dict) -> dict:
    """
    Run churn prediction on a single customer's features.

    Args:
        features: Dictionary of feature name → value.

    Returns:
        dict with churn_probability, churn_prediction, and risk_level.
    """
    model = load_model()
    df = pd.DataFrame([features])
    prob = float(model.predict_proba(df)[0, 1])
    prediction = prob >= 0.5

    if prob < 0.3:
        risk = "low"
    elif prob < 0.6:
        risk = "medium"
    else:
        risk = "high"

    return {
        "churn_probability": round(prob, 4),
        "churn_prediction": prediction,
        "risk_level": risk,
    }
