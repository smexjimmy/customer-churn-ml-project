"""
Data Preprocessing Module
==========================
Cleans raw churn data, encodes features, scales numerics,
and splits into train/val/test sets.

Usage:
    python src/data/preprocess.py

Called by DVC stage: dvc repro preprocess
"""

import logging
import os

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


def encode_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Label-encode categorical columns."""
    cat_cols = params["features"]["categorical_columns"]
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Standard-scale numeric columns."""
    num_cols = params["features"]["numeric_columns"]
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def split_data(
    df: pd.DataFrame, params: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, val, and test sets."""
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

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    return train, val, test


if __name__ == "__main__":
    params = load_params()
    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv(params["data"]["raw_path"])
    df = clean_data(df, params)
    df = encode_features(df, params)
    df = scale_features(df, params)

    train, val, test = split_data(df, params)

    train.to_csv(params["data"]["train_path"], index=False)
    val.to_csv(params["data"]["val_path"], index=False)
    test.to_csv(params["data"]["test_path"], index=False)

    logger.info(
        f"Split complete — train: {len(train)}, val: {len(val)}, test: {len(test)}"
    )
