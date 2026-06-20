"""
Data Preprocessing Module
==========================
Cleans raw churn data, encodes features, scales numerics,
and splits into train/val/test sets.

IMPORTANT: Encoders and scalers are fit ONLY on the training set
to prevent data leakage. They are persisted for use during inference.

Usage:
    python src/data/preprocess.py

Called by DVC stage: dvc repro preprocess
"""

import logging
import os
import pickle

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def clean_data(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Drop unused columns and fix data types."""
    drop_cols = params["data"].get("drop_columns", [])
    df = df.drop(columns=drop_cols, errors="ignore")

    # TotalCharges is sometimes stored as string — fix it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()

    # Encode target column: Yes → 1, No → 0
    target = params["data"]["target_column"]
    df[target] = (df[target] == "Yes").astype(int)

    return df


def encode_features(
    df: pd.DataFrame, params: dict, encoders: dict = None
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns.

    If encoders is None, fit new encoders on df (training set only).
    Otherwise, apply existing encoders (for val/test sets).

    Returns: (encoded_df, encoders_dict)
    """
    cat_cols = params["features"]["categorical_columns"]
    df = df.copy()

    if encoders is None:
        # Fit new encoders on training set
        encoders = {}
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le
    else:
        # Apply existing encoders to val/test sets
        for col in cat_cols:
            if col in df.columns:
                df[col] = encoders[col].transform(df[col].astype(str))

    return df, encoders


def scale_features(
    df: pd.DataFrame, params: dict, scaler: StandardScaler = None
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standard-scale numeric columns.

    If scaler is None, fit a new scaler on df (training set only).
    Otherwise, apply existing scaler (for val/test sets).

    Returns: (scaled_df, scaler)
    """
    num_cols = params["features"]["numeric_columns"]
    df = df.copy()

    if scaler is None:
        # Fit new scaler on training set
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        # Apply existing scaler to val/test sets
        df[num_cols] = scaler.transform(df[num_cols])

    return df, scaler


def split_data(
    df: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, StandardScaler]:
    """
    Split into train, val, and test sets.
    Fit encoders and scaler on train set, apply to val and test.

    Returns: (train, val, test, encoders_dict, scaler)
    """
    target = params["data"]["target_column"]
    test_size = params["data"]["test_size"]
    val_size = params["data"]["val_size"]
    seed = params["data"]["random_state"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_size / (1 - test_size),
        random_state=seed,
        stratify=y_train,
    )

    # Fit encoders and scaler on TRAIN set only
    X_train, encoders = encode_features(X_train, params, encoders=None)
    X_train, scaler = scale_features(X_train, params, scaler=None)

    # Apply fitted encoders and scaler to val and test sets
    X_val, _ = encode_features(X_val, params, encoders=encoders)
    X_val, _ = scale_features(X_val, params, scaler=scaler)

    X_test, _ = encode_features(X_test, params, encoders=encoders)
    X_test, _ = scale_features(X_test, params, scaler=scaler)

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, val, test, encoders, scaler


if __name__ == "__main__":
    params = load_params()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(params["data"]["raw_path"])
    df = clean_data(df, params)

    train, val, test, encoders, scaler = split_data(df, params)

    train.to_csv(params["data"]["train_path"], index=False)
    val.to_csv(params["data"]["val_path"], index=False)
    test.to_csv(params["data"]["test_path"], index=False)

    # Save encoders and scaler for inference
    encoders_path = "models/label_encoders.pkl"
    scaler_path = "models/standard_scaler.pkl"
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(
        f"Split complete — train: {len(train)}, val: {len(val)}, test: {len(test)}"
    )
    logger.info(f"Encoders saved to {encoders_path}")
    logger.info(f"Scaler saved to {scaler_path}")
