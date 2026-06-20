"""
Data Drift Detection Module
============================
Uses Evidently AI to compare reference data (training set)
against current production data to detect feature drift.

Usage:
    python src/monitoring/drift_detector.py
"""

import json
import logging
import os

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def detect_drift(
    reference_path: str, current_path: str, params: dict, threshold_ratio: float = 0.3
) -> dict:
    """
    Compare current data distribution against the reference (training) set.

    Args:
        reference_path: Path to training/reference data CSV.
        current_path: Path to incoming/production data CSV.
        params: Project params (features config).
        threshold_ratio: Ratio of drifted features (0.0 to 1.0)
                         above which we trigger warning.

    Returns:
        dict with drift report summary.
    """
    # Check if production logs exist and have enough records
    prod_logs_path = "data/production_logs.csv"
    if os.path.exists(prod_logs_path):
        try:
            prod_df = pd.read_csv(prod_logs_path)
            # Filter out duplicate headers written during concurrent requests
            if "gender" in prod_df.columns:
                prod_df = prod_df[prod_df["gender"] != "gender"].copy()
            if len(prod_df) >= 10:
                current_path = prod_logs_path
                logger.info(
                    f"Using production logs for drift check: {current_path} "
                    f"({len(prod_df)} records)"
                )
            else:
                logger.warning(
                    f"Production logs exist but have only {len(prod_df)} records. "
                    "Falling back to test baseline."
                )
        except Exception as e:
            logger.error(
                f"Error reading production logs: {e}. Falling back to test baseline."
            )

    logger.info(f"Loading reference data from {reference_path}")
    logger.info(f"Loading current data from {current_path}")

    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    # Clean up duplicate headers if using production logs
    if current_path == prod_logs_path and "gender" in cur_df.columns:
        cur_df = cur_df[cur_df["gender"] != "gender"].copy()
        # Ensure correct numeric types without using deprecated errors='ignore'
        for col in cur_df.columns:
            try:
                cur_df[col] = pd.to_numeric(cur_df[col])
            except (ValueError, TypeError):
                pass

    # Exclude the target column to only check features drift
    target = params["data"]["target_column"]
    if target in ref_df.columns:
        ref_df = ref_df.drop(columns=[target])
    if target in cur_df.columns:
        cur_df = cur_df.drop(columns=[target])

    logger.info("Running Evidently AI Data Drift Report...")
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=ref_df, current_data=cur_df)

    # Extract report as a python dictionary
    report_dict = snapshot.dict()

    # Get summary metrics from DriftedColumnsCount metric
    drifted_columns_metric = report_dict["metrics"][0]
    metric_value = drifted_columns_metric["value"]

    number_of_drifted_features = int(metric_value["count"])
    share_of_drifted_features = float(metric_value["share"])
    number_of_features = len(report_dict["metrics"]) - 1

    dataset_drift = share_of_drifted_features >= threshold_ratio
    drift_status = "drift_detected" if dataset_drift else "no_drift"

    summary = {
        "status": drift_status,
        "number_of_features": number_of_features,
        "number_of_drifted_features": number_of_drifted_features,
        "share_of_drifted_features": round(share_of_drifted_features, 4),
        "dataset_drift_detected": bool(dataset_drift),
    }

    # Save reports
    os.makedirs("metrics", exist_ok=True)

    # Save interactive HTML visualization report
    html_report_path = "metrics/drift_report.html"
    snapshot.save_html(html_report_path)
    logger.info(f"Saved interactive HTML drift report to {html_report_path}")

    # Save JSON summary metrics
    json_report_path = "metrics/drift_metrics.json"
    with open(json_report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved drift summary metrics to {json_report_path}")

    return summary


if __name__ == "__main__":
    params = load_params()
    result = detect_drift(
        reference_path=params["data"]["train_path"],
        current_path=params["data"]["test_path"],
        params=params,
    )
    print("\n--- Drift Report Summary ---")
    print(json.dumps(result, indent=2))
