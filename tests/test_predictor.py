"""Predictor contract, failure handling, and schema/range checks."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from impactmap.feature_extractor import FEATURE_COLUMNS, FEATURE_DTYPES
from impactmap.predictor import _load_cached, predict_risk


def _fixture_artifacts(tmp_path: Path):
    x = np.array([[i + j for j in range(len(FEATURE_COLUMNS))] for i in range(20)], dtype=float)
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, [0] * 10 + [1] * 10)
    model_path = tmp_path / "model.joblib"
    joblib.dump({"model_version": "test", "calibrated_model": model,
                 "feature_columns": list(FEATURE_COLUMNS), "threshold": .5,
                 "risk_tiers": {"Low": 0, "Medium": .3, "High": .6},
                 "training_ranges": {c: {"min": 0, "max": 30} for c in FEATURE_COLUMNS},
                 "training_medians": {c: 10 for c in FEATURE_COLUMNS}}, model_path)
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"feature_columns": list(FEATURE_COLUMNS),
                                      "feature_dtypes": FEATURE_DTYPES,
                                      "feature_descriptions": {c: c for c in FEATURE_COLUMNS}}))
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"library_versions": {}}))
    tiers_path = tmp_path / "tiers.json"
    tiers_path.write_text(json.dumps({"Low": 0, "Medium": .3, "High": .6}))
    return model_path, schema_path, metrics_path, tiers_path


def _features(start=1):
    return {c: float(start + i) for i, c in enumerate(FEATURE_COLUMNS)}


def test_predictor_loads_test_pipeline_and_flags_out_of_range(tmp_path):
    paths = _fixture_artifacts(tmp_path)
    _load_cached.cache_clear()
    values = _features()
    values["la"] = 100
    result = predict_risk(values, model_path=paths[0], schema_path=paths[1],
                          metrics_path=paths[2], tiers_path=paths[3])
    assert result.available
    assert 0 <= result.score <= 1
    assert result.threshold_used == .5
    assert any("Out-of-training-range" in n for n in result.reliability_notes)


def test_predictor_missing_model_and_schema_mismatch(tmp_path):
    paths = _fixture_artifacts(tmp_path)
    _load_cached.cache_clear()
    missing = predict_risk(_features(), model_path=tmp_path / "absent.joblib",
                           schema_path=paths[1], metrics_path=paths[2])
    assert not missing.available and "not found" in missing.error
    bad_schema = json.loads(paths[1].read_text())
    bad_schema["feature_columns"] = bad_schema["feature_columns"][::-1]
    paths[1].write_text(json.dumps(bad_schema))
    _load_cached.cache_clear()
    mismatch = predict_risk(_features(), model_path=paths[0], schema_path=paths[1], metrics_path=paths[2])
    assert not mismatch.available and "do not match" in mismatch.error


def test_predictor_missing_features_and_zero_source_files(tmp_path):
    paths = _fixture_artifacts(tmp_path)
    _load_cached.cache_clear()
    missing = _features()
    missing.pop("nf")
    result = predict_risk(missing, model_path=paths[0], schema_path=paths[1], metrics_path=paths[2])
    assert not result.available and "nf" in result.error
    zero = _features()
    zero["nf"] = 0
    result = predict_risk(zero, model_path=paths[0], schema_path=paths[1], metrics_path=paths[2])
    assert not result.available and "zero changed" in result.error
