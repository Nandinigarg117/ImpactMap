"""Standalone model evaluation script: verifies saved model artifact against held-out test data."""

import json
import logging
from pathlib import Path
import sys

# Ensure src is in python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from impactmap.feature_extractor import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_saved_model(
    model_path: Path = Path("models/model.joblib"),
    test_data_path: Path = Path("data/processed/test.csv"),
) -> dict:
    """Loads saved model artifact and computes held-out test metrics."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}")
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

    artifact = joblib.load(model_path)
    model = artifact["calibrated_model"]
    threshold = artifact.get("threshold", 0.5)

    test_df = pd.read_csv(test_data_path)
    X_test = test_df[list(FEATURE_COLUMNS)]
    y_test = test_df["buggy"]

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    pr_auc = round(float(average_precision_score(y_test, probs)), 4)
    roc_auc = round(float(roc_auc_score(y_test, probs)), 4)
    brier = round(float(brier_score_loss(y_test, probs)), 4)
    cm = confusion_matrix(y_test, preds).tolist()
    base_rate = round(float(y_test.mean()), 4)

    logger.info("=== HELD-OUT TEST EVALUATION ===")
    logger.info("Model Version: %s", artifact.get("model_version", "unknown"))
    logger.info("Test Rows: %d | Positive Base Rate: %.4f", len(test_df), base_rate)
    logger.info("Decision Threshold: %.2f", threshold)
    logger.info("PR-AUC (Avg Precision): %.4f", pr_auc)
    logger.info("ROC-AUC: %.4f", roc_auc)
    logger.info("Brier Score: %.4f", brier)
    logger.info("Confusion Matrix: TN=%d, FP=%d, FN=%d, TP=%d", cm[0][0], cm[0][1], cm[1][0], cm[1][1])

    report = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "base_rate": base_rate,
        "threshold": threshold,
        "confusion_matrix": cm,
    }
    return report


if __name__ == "__main__":
    evaluate_saved_model()
