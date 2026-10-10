"""SHAP and explicit approximation paths for local model explanations."""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from impactmap.feature_extractor import FEATURE_COLUMNS, FEATURE_DTYPES
from impactmap.explainer import explain_prediction
from impactmap.predictor import LoadedModel, _load_cached
import impactmap.explainer as explainer_module


def _loaded(tmp_path: Path):
    x = np.array([[i + j for j in range(len(FEATURE_COLUMNS))] for i in range(20)], dtype=float)
    model = DecisionTreeClassifier(max_depth=2, random_state=0).fit(x, [0] * 10 + [1] * 10)
    artifact = {"model_version": "test", "calibrated_model": model, "base_model": model,
                "feature_columns": list(FEATURE_COLUMNS),
                "training_medians": {c: 10 for c in FEATURE_COLUMNS}}
    schema = {"feature_columns": list(FEATURE_COLUMNS), "feature_dtypes": FEATURE_DTYPES,
              "feature_descriptions": {c: "the selected commit feature" for c in FEATURE_COLUMNS}}
    return LoadedModel(artifact, schema, [])


def test_shap_local_explanation_for_small_tree(tmp_path):
    loaded = _loaded(tmp_path)
    values = {c: 11.0 + i for i, c in enumerate(FEATURE_COLUMNS)}
    result = explain_prediction(values, loaded=loaded, importance_path=tmp_path / "missing.json")
    assert result.available
    assert result.local_method == "SHAP"
    assert len(result.features) == len(FEATURE_COLUMNS)
    assert result.features[0].feature == "la"
    assert result.features[0].value == values["la"]
    assert result.features[0].direction in {"raises", "lowers", "neutral"}


def test_shap_failure_uses_labeled_local_approximation(monkeypatch, tmp_path):
    loaded = _loaded(tmp_path)
    importance = tmp_path / "global.json"
    importance.write_text(json.dumps({"la": 1.2}))

    def fail(*args, **kwargs):
        raise RuntimeError("forced SHAP failure")

    monkeypatch.setattr(explainer_module, "_shap_contributions", fail)
    values = {c: 11.0 + i for i, c in enumerate(FEATURE_COLUMNS)}
    result = explain_prediction(values, loaded=loaded, importance_path=importance)
    assert result.available
    assert "approximation" in result.local_method
    assert result.global_importance == {"la": 1.2}
    assert any("approximate" in n for n in result.notes)
    assert len(result.features) == len(FEATURE_COLUMNS)
