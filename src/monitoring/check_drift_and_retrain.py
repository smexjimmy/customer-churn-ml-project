"""
Drift-Triggered Retraining Script
=================================
Checks the latest drift metrics from `metrics/drift_metrics.json`.
If data drift is detected, triggers DVC pipeline reproduction to retrain the model.

Usage:
    python src/monitoring/check_drift_and_retrain.py [--force]
"""

import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_and_retrain(
    metrics_path: str = "metrics/drift_metrics.json", force: bool = False
):
    """Check drift metrics and run retraining if drift is detected."""
    if not os.path.exists(metrics_path):
        logger.error(
            f"Drift metrics file not found at {metrics_path}. "
            "Run drift_detector.py first."
        )
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)

    drift_detected = metrics.get("dataset_drift_detected", False)
    drift_share = metrics.get("share_of_drifted_features", 0.0)
    status = metrics.get("status", "unknown")

    logger.info(
        f"Loaded drift metrics. Status: {status}, Drift share: {drift_share:.4f}"
    )

    if drift_detected or force:
        if force:
            logger.info("Retraining forced by user flag.")
        else:
            logger.warning(
                f"Data drift detected (share: {drift_share:.4f} >= threshold). "
                "Triggering retraining..."
            )

        # Run model retraining
        logger.info("Executing DVC pipeline reproduction (dvc repro)...")
        try:
            # We run DVC repro in the workspace
            # Set environment variables for console encoding safety
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"

            # Run DVC repro using the active python interpreter as a module
            result = subprocess.run(
                [sys.executable, "-m", "dvc", "repro"],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            logger.info("DVC reproduction completed successfully.")
            logger.info(result.stdout)

            # Commit the changes to git or push to DVC remote
            logger.info(
                "Model retrained successfully. "
                "Remember to push updated DVC outputs to storage."
            )
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"DVC reproduction failed: {e}")
            logger.error(f"Command output: {e.stdout}")
            logger.error(f"Error output: {e.stderr}")
            return False
    else:
        logger.info("No data drift detected. Retraining is not required.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check drift and retrain model.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of drift status",
    )
    args = parser.parse_args()

    check_and_retrain(force=args.force)
