"""End-to-end orchestrator checks on a real fixture Git repository."""

import json
from pathlib import Path

import git
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from impactmap.config import AppConfig
from impactmap.explainer import explain_prediction
from impactmap.feature_extractor import FEATURE_COLUMNS
from impactmap.predictor import LoadedModel, predict_risk
from tests.fixtures.repo_fixtures import GitRepoBuilder
import impactmap.analysis as analysis_module


def _test_pipeline(tmp_path: Path):
    x = np.array([[i + j for j in range(len(FEATURE_COLUMNS))] for i in range(20)], dtype=float)
    tree = DecisionTreeClassifier(max_depth=2, random_state=42).fit(x, [0] * 10 + [1] * 10)
    artifact = {"model_version": "integration-test", "calibrated_model": tree,
                "base_model": tree, "feature_columns": list(FEATURE_COLUMNS),
                "threshold": .5, "risk_tiers": {"Low": 0, "Medium": .3, "High": .6},
                "training_ranges": {c: {"min": 0, "max": 50} for c in FEATURE_COLUMNS},
                "training_medians": {c: 10 for c in FEATURE_COLUMNS}}
    model_path = tmp_path / "test-model.joblib"
    joblib.dump(artifact, model_path)
    schema_path = Path("models/feature_schema.json").resolve()
    metrics_path = tmp_path / "test-metrics.json"
    metrics_path.write_text(json.dumps({"library_versions": {}}), encoding="utf-8")
    loaded = LoadedModel(artifact, json.loads(schema_path.read_text()), [])
    return model_path, schema_path, metrics_path, loaded


def test_run_analysis_ml_available_and_result_matches_ui_contract(tmp_path, monkeypatch):
    builder = GitRepoBuilder(tmp_path / "fixture-repo")
    commits = builder.build_standard_repo()
    repo = git.Repo(tmp_path / "fixture-repo")
    model_path, schema_path, metrics_path, loaded = _test_pipeline(tmp_path)
    monkeypatch.setattr(analysis_module, "_model_identity", lambda: "integration-test:model")
    monkeypatch.setattr(analysis_module, "predict_risk", lambda features: predict_risk(
        features, model_path=model_path, schema_path=schema_path, metrics_path=metrics_path,
        tiers_path=tmp_path / "missing-tiers.json"))
    monkeypatch.setattr(analysis_module, "explain_prediction", lambda features: explain_prediction(
        features, loaded=loaded, importance_path=tmp_path / "missing-importance.json"))
    analysis_module.clear_analysis_cache()

    result = analysis_module.run_analysis(repo, commits["modify"], AppConfig())
    assert result.diff.commit_sha == commits["modify"]
    assert result.graph.total_nodes > 0
    assert result.impact.evidence_records["src/pkg/calc.py"].category == "changed"
    assert result.features is not None
    assert result.prediction.available
    assert 0 <= result.prediction.score <= 1
    assert result.explanation.available
    assert result.ranking.files
    assert result.cache_key[0] == result.repository_identity
    assert result.cache_key[1] == commits["modify"]
    assert result.cache_key[2] == AppConfig().config_hash()
    assert result.cache_key[3].startswith("integration-test")


def test_run_analysis_keeps_graph_when_ml_unavailable(tmp_path, monkeypatch):
    builder = GitRepoBuilder(tmp_path / "fixture-repo")
    commits = builder.build_standard_repo()
    repo = git.Repo(tmp_path / "fixture-repo")
    monkeypatch.setattr(analysis_module, "_model_identity", lambda: "unavailable:test")
    monkeypatch.setattr(analysis_module, "predict_risk", lambda features:
                        analysis_module.PredictionResult(False, error="test model unavailable"))
    analysis_module.clear_analysis_cache()

    result = analysis_module.run_analysis(repo, commits["modify"], AppConfig())
    assert result.graph.total_nodes > 0
    assert result.impact.evidence_records["src/pkg/calc.py"].category == "changed"
    assert not result.prediction.available
    assert result.prediction.error == "test model unavailable"
    assert not result.explanation.available
    assert "commit_risk" in result.ranking.omitted_components
    assert result.ranking.files

    deleted_result = analysis_module.run_analysis(repo, commits["delete"], AppConfig())
    assert deleted_result.parent_graph is not None
    assert deleted_result.parent_graph.total_nodes > 0
    assert any(item.path == "README.md" for item in deleted_result.ranking.removed)


def test_run_analysis_converts_predictor_exception_to_unavailable(tmp_path, monkeypatch):
    builder = GitRepoBuilder(tmp_path / "fixture-repo")
    commits = builder.build_standard_repo()
    repo = git.Repo(tmp_path / "fixture-repo")
    monkeypatch.setattr(analysis_module, "_model_identity", lambda: "exception-test")

    def fail(_features):
        raise RuntimeError("simulated prediction failure")

    monkeypatch.setattr(analysis_module, "predict_risk", fail)
    analysis_module.clear_analysis_cache()
    result = analysis_module.run_analysis(repo, commits["modify"], AppConfig())
    assert result.graph.total_nodes > 0
    assert not result.prediction.available
    assert "simulated prediction failure" in result.prediction.error
