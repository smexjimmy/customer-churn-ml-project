"""
Model Inference Module
=======================
Loads the trained model and preprocessors (encoders, scaler) to run predictions.
Used by the FastAPI service and evaluation.
"""

import logging
import os
import pickle
from functools import lru_cache

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.pkl")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "models/label_encoders.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/standard_scaler.pkl")


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the model — avoids reloading on every request."""
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_preprocessors():
    """Load and cache encoders and scaler — fitted on training set."""
    logger.info(f"Loading preprocessors from {ENCODERS_PATH} and {SCALER_PATH}")
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return encoders, scaler


def preprocess_features(features: dict, params: dict) -> pd.DataFrame:
    """
    Preprocess raw features using fitted encoders and scaler.

    Args:
        features: Dictionary of feature name → value (raw customer data).
        params: Dictionary with feature column specifications.

    Returns:
        Preprocessed DataFrame ready for model prediction.
    """
    encoders, scaler = load_preprocessors()

    # Get expected column order directly from the trained model
    model = load_model()
    expected_cols = list(model.feature_names_in_)

    # Create DataFrame with all expected columns in order
    df = pd.DataFrame([features])[expected_cols].copy()

    # Apply label encoders to categorical columns
    cat_cols = params["features"]["categorical_columns"]
    for col in cat_cols:
        if col in df.columns and col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))

    # Apply scaler to numeric columns
    num_cols = params["features"]["numeric_columns"]
    df[num_cols] = scaler.transform(df[num_cols])

    return df


def predict(features: dict, params: dict = None) -> dict:
    """
    Run churn prediction on raw customer features.

    Args:
        features: Dictionary of feature name → value.
        params: Parameter dict with feature specs (required if preprocessing needed).

    Returns:
        dict with churn_probability, churn_prediction, and risk_level.
    """
    if params is None:
        import yaml

        with open("params.yaml") as f:
            params = yaml.safe_load(f)

    # Preprocess features
    df = preprocess_features(features, params)

    # Run prediction
    model = load_model()
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
