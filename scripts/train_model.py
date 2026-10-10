"""Training and evaluation pipeline: baselines, calibrated boosting, threshold tuning, and error analysis."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys

# Ensure src is in python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from impactmap.config import RANDOM_SEED
from impactmap.feature_extractor import FEATURE_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")


def load_splits():
    """Loads time-aware processed dataset splits."""
    train_path = PROCESSED_DIR / "train.csv"
    val_path = PROCESSED_DIR / "val.csv"
    test_path = PROCESSED_DIR / "test.csv"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError("Processed splits not found. Run scripts/prepare_dataset.py first.")

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    return train_df, val_df, test_df


def train_and_evaluate():
    """Executes model training, baseline comparisons, calibration, and saves artifacts."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    train_df, val_df, test_df = load_splits()

    X_train, y_train = train_df[list(FEATURE_COLUMNS)], train_df["buggy"]
    X_val, y_val = val_df[list(FEATURE_COLUMNS)], val_df["buggy"]
    X_test, y_test = test_df[list(FEATURE_COLUMNS)], test_df["buggy"]

    test_base_rate = round(float(y_test.mean()), 4)
    logger.info("Dataset loaded. Test positive base rate: %.4f", test_base_rate)

    # 1. Baseline: Dummy Classifier (Prior)
    dummy = DummyClassifier(strategy="prior")
    dummy.fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_test)[:, 1]
    dummy_preds = (dummy_probs >= 0.5).astype(int)

    # 2. Baseline: Churn Heuristic (High churn > 50 lines)
    churn_test = X_test["la"] + X_test["ld"]
    heuristic_preds = (churn_test > 50).astype(int)
    # Scaled heuristic score for ranking
    heuristic_probs = np.clip(churn_test / 200.0, 0.0, 1.0)

    # 3. Baseline: Logistic Regression
    lr_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_SEED, max_iter=500)),
    ])
    lr_pipe.fit(X_train, y_train)
    lr_val_probs = lr_pipe.predict_proba(X_val)[:, 1]
    lr_test_probs = lr_pipe.predict_proba(X_test)[:, 1]

    # 4. Primary Model: HistGradientBoostingClassifier
    hgb = HistGradientBoostingClassifier(
        class_weight="balanced",
        random_state=RANDOM_SEED,
        max_iter=100,
        min_samples_leaf=30,
        l2_regularization=1.0,
    )
    hgb.fit(X_train, y_train)

    # 5. Calibration using validation split
    frozen_hgb = FrozenEstimator(hgb)
    calibrator = CalibratedClassifierCV(frozen_hgb, method="sigmoid")
    calibrator.fit(X_val, y_val)

    val_probs = calibrator.predict_proba(X_val)[:, 1]
    test_probs = calibrator.predict_proba(X_test)[:, 1]

    # 6. Optimize decision threshold on validation set (maximizing F1)
    best_threshold = 0.5
    best_val_f1 = 0.0
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (val_probs >= t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_val_f1:
            best_val_f1 = score
            best_threshold = round(float(t), 2)

    logger.info("Optimized validation threshold: %.2f (Val F1: %.4f)", best_threshold, best_val_f1)

    # Evaluate all models on held-out test set
    def _calc_metrics(y_true, probs, preds=None, threshold=0.5):
        if preds is None:
            preds = (probs >= threshold).astype(int)
        cm = confusion_matrix(y_true, preds).tolist()
        return {
            "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y_true, probs)), 4),
            "threshold_used": threshold,
            "confusion_matrix": {
                "tn": cm[0][0],
                "fp": cm[0][1],
                "fn": cm[1][0],
                "tp": cm[1][1],
            },
        }

    test_preds = (test_probs >= best_threshold).astype(int)
    hgb_metrics = _calc_metrics(y_test, test_probs, test_preds, threshold=best_threshold)
    lr_metrics = _calc_metrics(y_test, lr_test_probs, threshold=0.5)
    heuristic_metrics = _calc_metrics(y_test, heuristic_probs, heuristic_preds, threshold=0.5)
    dummy_metrics = _calc_metrics(y_test, dummy_probs, dummy_preds, threshold=0.5)

    # 7. Error Analysis: false positives and false negatives
    test_eval_df = test_df.copy()
    test_eval_df["pred"] = test_preds
    test_eval_df["churn"] = test_eval_df["la"] + test_eval_df["ld"]
    test_eval_df["churn_bucket"] = pd.cut(
        test_eval_df["churn"],
        bins=[-1, 20, 200, 1000000],
        labels=["small (<=20)", "medium (21-200)", "large (>200)"],
    )

    error_analysis = {
        "by_size_bucket": {},
        "by_project": {},
    }

    for bucket, grp in test_eval_df.groupby("churn_bucket", observed=False):
        fp = int(((grp["buggy"] == 0) & (grp["pred"] == 1)).sum())
        fn = int(((grp["buggy"] == 1) & (grp["pred"] == 0)).sum())
        total = len(grp)
        error_analysis["by_size_bucket"][str(bucket)] = {
            "total": total,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_rate": round(fp / total, 4) if total > 0 else 0.0,
            "fn_rate": round(fn / total, 4) if total > 0 else 0.0,
        }

    for proj, grp in test_eval_df.groupby("project"):
        fp = int(((grp["buggy"] == 0) & (grp["pred"] == 1)).sum())
        fn = int(((grp["buggy"] == 1) & (grp["pred"] == 0)).sum())
        total = len(grp)
        error_analysis["by_project"][proj] = {
            "total": total,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_rate": round(fp / total, 4) if total > 0 else 0.0,
            "fn_rate": round(fn / total, 4) if total > 0 else 0.0,
        }

    # 8. Compute Global Feature Importance using SHAP
    logger.info("Computing global feature importance via SHAP...")
    sample_val = X_val.iloc[:1000]
    try:
        tree_explainer = shap.TreeExplainer(hgb)
        shap_vals = tree_explainer.shap_values(sample_val)
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        importance_dict = {col: round(float(val), 6) for col, val in zip(FEATURE_COLUMNS, mean_abs_shap)}
    except Exception as exc:
        logger.warning("SHAP global importance failed (%s); using held-out permutation importance", exc)
        perm = permutation_importance(hgb, sample_val, y_val.iloc[:len(sample_val)],
                                      scoring="average_precision", n_repeats=5,
                                      random_state=RANDOM_SEED, n_jobs=-1)
        importance_dict = {col: round(float(max(0.0, val)), 6)
                           for col, val in zip(FEATURE_COLUMNS, perm.importances_mean)}
    # Sort descending
    sorted_importance = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))

    with open(MODELS_DIR / "global_importance.json", "w", encoding="utf-8") as f:
        json.dump(sorted_importance, f, indent=2)

    # 9. Risk Tiers
    # Derive tiers from validation score quantiles and optimal threshold
    risk_tiers = {
        "High": round(best_threshold, 2),
        "Medium": round(best_threshold * 0.6, 2),
        "Low": 0.0,
        "derivation": f"High tier set at optimal validation threshold {best_threshold}; Medium tier set at 60% of High threshold.",
    }
    with open(MODELS_DIR / "risk_tiers.json", "w", encoding="utf-8") as f:
        json.dump(risk_tiers, f, indent=2)

    # 10. Save Model Pipeline Artifact
    pipeline_artifact = {
        "model_version": "1.0.0",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "calibrated_model": calibrator,
        "base_model": hgb,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": best_threshold,
        "risk_tiers": risk_tiers,
        "training_ranges": {col: {"min": float(X_train[col].min()), "max": float(X_train[col].max())}
                            for col in FEATURE_COLUMNS},
        "training_medians": {col: float(X_train[col].median()) for col in FEATURE_COLUMNS},
    }
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipeline_artifact, model_path, compress=3)
    logger.info("Model pipeline artifact saved to %s (%d bytes)", model_path, model_path.stat().st_size)

    # 11. Compile Comprehensive metrics.json
    all_metrics = {
        "model_version": "1.0.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "library_versions": {
            "scikit_learn": "1.7.2",
            "shap": "0.53.0",
            "pandas": "2.3.3",
            "numpy": "2.3.5",
        },
        "dataset": {
            "name": "ApacheJIT",
            "source": "https://github.com/hosseinkshvrz/apachejit",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "test_positive_base_rate": test_base_rate,
            "split_type": "time-aware (70/15/15 by author_date)",
        },
        "evaluation": {
            "primary_model_hgb_calibrated": hgb_metrics,
            "baseline_logistic_regression": lr_metrics,
            "baseline_churn_heuristic": heuristic_metrics,
            "baseline_dummy_prior": dummy_metrics,
        },
        "error_analysis": error_analysis,
    }

    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Metrics report saved to %s", metrics_path)
    return all_metrics


if __name__ == "__main__":
    train_and_evaluate()
