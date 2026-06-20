"""
Model Evaluation Module
========================
Evaluates the trained model on the held-out test set
and saves metrics to metrics/metrics.json (tracked by DVC).

Usage:
    python src/models/evaluate.py

Called by DVC stage: dvc repro evaluate
"""

import json
import logging
import os

import joblib
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def evaluate(params: dict) -> dict:
    """Load model and test set, return metrics dict."""
    target = params["data"]["target_column"]
    test = pd.read_csv(params["data"]["test_path"])
    X_test = test.drop(columns=[target])
    y_test = test[target]

    model = joblib.load(params["training"]["model_output_path"])

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "test_auc": round(roc_auc_score(y_test, y_prob), 4),
        "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
        "test_f1": round(f1_score(y_test, y_pred), 4),
        "test_precision": round(precision_score(y_test, y_pred), 4),
        "test_recall": round(recall_score(y_test, y_pred), 4),
    }
    return metrics


if __name__ == "__main__":
    params = load_params()
    os.makedirs("metrics", exist_ok=True)

    metrics = evaluate(params)

    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    metrics_path = params["training"]["metrics_output_path"]
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {metrics_path}")
