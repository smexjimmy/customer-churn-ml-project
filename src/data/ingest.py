"""
Data Ingestion Module
=====================
Downloads the Telco Customer Churn dataset and saves it as raw CSV.

Usage:
    python src/data/ingest.py

Called by DVC stage: dvc repro ingest
"""

import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_DATA_PATH = "data/raw/churn.csv"

# Public Telco Churn dataset URL (IBM sample dataset)
DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def download_churn_data() -> pd.DataFrame:
    """
    Download the Telco Customer Churn dataset from a public URL.

    Returns:
        pd.DataFrame: Raw churn dataset with 7043 rows and 21 columns.
    """
    logger.info("Downloading Telco Customer Churn dataset...")
    df = pd.read_csv(DATASET_URL)
    logger.info(f"Downloaded {len(df)} rows, {df.shape[1]} columns.")
    return df


if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    df = download_churn_data()
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info(f"Saved raw data to {RAW_DATA_PATH}")
