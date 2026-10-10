"""Tests validating model artifacts, metrics, risk tiers, and model loading."""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from impactmap.feature_extractor import FEATURE_COLUMNS


def test_model_artifact_integrity():
    model_path = Path("models/model.joblib")
    assert model_path.exists(), "models/model.joblib must exist"
    assert model_path.stat().st_size < 5 * 1024 * 1024, "Model artifact must be compact (< 5MB)"

    artifact = joblib.load(model_path)
    assert "calibrated_model" in artifact
    assert "base_model" in artifact
    assert "feature_columns" in artifact
    assert "threshold" in artifact
    assert "risk_tiers" in artifact

    assert artifact["feature_columns"] == list(FEATURE_COLUMNS)

    # Test prediction on sample feature row
    model = artifact["calibrated_model"]
    sample_input = np.ones((1, len(FEATURE_COLUMNS)))
    probs = model.predict_proba(sample_input)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0, 1] <= 1.0


def test_metrics_json_integrity():
    metrics_path = Path("models/metrics.json")
    assert metrics_path.exists(), "models/metrics.json must exist"

    with open(metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["model_version"] == "1.0.0"
    assert "evaluation" in data
    eval_data = data["evaluation"]
    assert "primary_model_hgb_calibrated" in eval_data
    assert "baseline_logistic_regression" in eval_data
    assert "baseline_churn_heuristic" in eval_data
    assert "baseline_dummy_prior" in eval_data

    # Check primary model numbers
    primary = eval_data["primary_model_hgb_calibrated"]
    assert primary["pr_auc"] > primary["brier_score"]
    assert primary["roc_auc"] > 0.70
    assert 0.0 < primary["threshold_used"] < 1.0
    assert "error_analysis" in data


def test_risk_tiers_integrity():
    tiers_path = Path("models/risk_tiers.json")
    assert tiers_path.exists(), "models/risk_tiers.json must exist"

    with open(tiers_path, "r", encoding="utf-8") as f:
        tiers = json.load(f)

    assert "High" in tiers
    assert "Medium" in tiers
    assert "Low" in tiers
    assert tiers["Low"] < tiers["Medium"] <= tiers["High"]


def test_global_importance_integrity():
    imp_path = Path("models/global_importance.json")
    assert imp_path.exists(), "models/global_importance.json must exist"

    with open(imp_path, "r", encoding="utf-8") as f:
        importance = json.load(f)

    for col in FEATURE_COLUMNS:
        assert col in importance
        assert importance[col] >= 0.0
