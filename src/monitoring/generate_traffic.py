"""
Mock Traffic Generator
======================
Simulates production traffic by reading raw Telco Customer Churn data and sending
requests to the FastAPI prediction endpoint.

Usage:
    python src/monitoring/generate_traffic.py --num-requests 100
"""

import argparse
import logging
import random
import time

import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

API_URL = "http://localhost:8000/predict"


def load_raw_samples(data_path: str = "data/raw/churn.csv") -> list:
    """Load raw dataset and return as a list of dicts ready for prediction API."""
    logger.info(f"Loading raw data samples from {data_path}")
    df = pd.read_csv(data_path)

    # Drop identification and target columns
    cols_to_drop = ["customerID", "Churn"]
    for col in cols_to_drop:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Clean up column types (convert SeniorCitizen to int, numeric cols to float)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    # Handle any empty strings in TotalCharges (replace with MonthlyCharges or 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

    return df.to_dict(orient="records")


def send_traffic(samples: list, num_requests: int, delay: float):
    """Send requests in a loop to the prediction API."""
    logger.info(
        f"Starting traffic simulation: sending {num_requests} requests "
        f"with {delay}s delay..."
    )
    success_count = 0

    for i in range(num_requests):
        sample = random.choice(samples)

        # Let's occasionally inject a value to trigger drift if needed (mocked)
        # e.g., setting tenure to very high or MonthlyCharges to very high
        if i % 10 == 0:
            sample = sample.copy()
            sample["tenure"] = float(
                random.randint(150, 200)
            )  # Out of range (max training tenure is 72)
            sample["MonthlyCharges"] = float(
                random.randint(250, 400)
            )  # Out of range (max MonthlyCharges is ~120)

        try:
            response = requests.post(API_URL, json=sample, timeout=5)
            if response.status_code == 200:
                success_count += 1
                data = response.json()
                logger.debug(
                    f"Request {i+1} success: "
                    f"Prob={data['churn_probability']}, "
                    f"Risk={data['risk_level']}"
                )
            else:
                logger.warning(
                    f"Request {i+1} failed with status code "
                    f"{response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.error(f"Request {i+1} failed with error: {e}")

        if delay > 0:
            time.sleep(delay)

    logger.info(
        f"Traffic simulation complete. "
        f"Successfully sent {success_count}/{num_requests} requests."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate production traffic for customer churn prediction."
    )
    parser.add_argument(
        "--num-requests", type=int, default=100, help="Number of requests to send"
    )
    parser.add_argument(
        "--delay", type=float, default=0.05, help="Delay between requests in seconds"
    )
    args = parser.parse_args()

    try:
        samples = load_raw_samples()
        send_traffic(samples, args.num_requests, args.delay)
    except Exception as e:
        logger.error(f"Traffic simulation failed: {e}")
