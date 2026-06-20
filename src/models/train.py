"""
Model Training Module
======================
Trains an XGBoost classifier, logs params/metrics/artifacts to MLflow,
and registers the best model in the MLflow Model Registry.

Usage:
    python src/models/train.py

Called by DVC stage: dvc repro train
"""

import logging
import os

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_data(params: dict) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    target = params["data"]["target_column"]
    train = pd.read_csv(params["data"]["train_path"])
    val = pd.read_csv(params["data"]["val_path"])
    X_train, y_train = train.drop(columns=[target]), train[target]
    X_val, y_val = val.drop(columns=[target]), val[target]
    return X_train, y_train, X_val, y_val


def build_model(params: dict) -> XGBClassifier:
    mp = params["model"]
    return XGBClassifier(
        n_estimators=mp["n_estimators"],
        max_depth=mp["max_depth"],
        learning_rate=mp["learning_rate"],
        subsample=mp["subsample"],
        colsample_bytree=mp["colsample_bytree"],
        scale_pos_weight=mp["scale_pos_weight"],
        random_state=mp["random_state"],
        eval_metric=mp["eval_metric"],
        early_stopping_rounds=params["training"]["early_stopping_rounds"],
        use_label_encoder=False,
    )


if __name__ == "__main__":
    params = load_params()
    os.makedirs("models", exist_ok=True)

    # Set MLflow tracking
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    mlflow.set_experiment(params["training"]["experiment_name"])

    X_train, y_train, X_val, y_val = load_data(params)

    with mlflow.start_run(run_name=params["training"]["run_name"]):
        # Log all params
        mlflow.log_params(params["model"])
        mlflow.log_params(
            {
                "test_size": params["data"]["test_size"],
                "val_size": params["data"]["val_size"],
            }
        )

        # Train
        model = build_model(params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Evaluate on validation set
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        mlflow.log_metric("val_auc", val_auc)
        logger.info(f"Validation AUC: {val_auc:.4f}")

        # Log model to MLflow
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name="churn-prediction",
        )

        # Save locally for DVC
        model_path = params["training"]["model_output_path"]
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
