"""Sanitized structured report and ranked-file CSV export tests."""

import csv
import io
import json
from datetime import datetime

import networkx as nx
import pytest

from impactmap.analysis import AnalysisResult
from impactmap.config import AppConfig
from impactmap.dependency_graph import GraphBuildResult
from impactmap.diff_analyzer import CommitDiffSummary, FileChange
from impactmap.explainer import ExplanationResult, FeatureExplanation
from impactmap.impact_analyzer import ImpactAnalysisResult, ImpactEvidence
from impactmap.impact_ranker import RankedFile, RankingResult
from impactmap.predictor import PredictionResult
from impactmap.report_generator import (
    CSV_COLUMNS,
    ReportExportError,
    build_report_payload,
    generate_csv_report,
    generate_json_report,
)


def _analysis(available=True):
    graph = nx.DiGraph()
    graph.add_node("src/mod.py", category="python_source")
    graph.add_node("tests/test_mod.py", category="test")
    graph.add_edge("tests/test_mod.py", "src/mod.py")
    evidence = {
        "src/mod.py": ImpactEvidence("src/mod.py", "changed", 0, ["src/mod.py"],
                                     "src/mod.py (Directly changed)", ["src/mod.py"], 1),
        "tests/test_mod.py": ImpactEvidence("tests/test_mod.py", "direct", 1,
                                             ["tests/test_mod.py", "src/mod.py"],
                                             "tests/test_mod.py -> src/mod.py", ["src/mod.py"], 1),
    }
    diff = CommitDiffSummary(
        "a" * 40, "b" * 40, False, False, False, False, 1, 3, 1, 4,
        [FileChange("src/mod.py", None, "modified", 3, 1, False, "python_source",
                    patch_text="PRIVATE FILE CONTENT SHOULD NOT EXPORT")], [],
    )
    impact = ImpactAnalysisResult("a" * 40, 2, 1, 1, 1, 0, 0, 1, .5,
                                  evidence_records=evidence)
    rank_file = RankedFile(1, "src/mod.py", "python_source", "High", .71,
                           "Direct change", "Review diff", ["src/mod.py"],
                           {"status": 1.0, "commit_risk": .6},
                           "src/mod.py directly changed; heuristic priority score .710.", 0)
    ranking = RankingResult([rank_file], [], {"status": .5, "commit_risk": .5}, [],
                            False, .6 if available else None,
                            {"dependency_distance": {"src/mod.py": 1},
                             "changed_status": {"src/mod.py": 1},
                             "full_combined_score": {"src/mod.py": 1}},
                            notes=["History unavailable or unreliable."])
    prediction = (PredictionResult(True, .6, "High", .5, model_version="test-model") if available
                  else PredictionResult(False, model_version=None,
                                        error="Model unavailable at /Users/private/model.joblib; contact dev@example.com."))
    explanation = ExplanationResult(
        True, "SHAP", "training-time SHAP", [FeatureExplanation("nf", 1, .12, "raises", .12,
                                                                   "This commit touches one file and raised risk.")],
        {"nf": .4},
    ) if available else ExplanationResult(False, error="Unavailable")
    return AnalysisResult(
        repository_identity="/Users/alice/MyProject/.git",
        commit_sha="a" * 40,
        commit_message="fix: path /Users/alice/private and dev@example.com ghp_SUPERSECRET123456",
        author="Author <dev@example.com>", authored_datetime="2026-01-01T00:00:00+00:00",
        is_merge_commit=False, parent_shas=["b" * 40], diff=diff,
        graph=GraphBuildResult(graph, "a" * 40, 2, 1,
                               parse_failures={"src/bad.py": "parse failed at /Users/alice/private"},
                               unresolved_imports={"src/mod.py": ["internal.nope"]}),
        parent_graph=None, impact=impact, features=None, prediction=prediction,
        explanation=explanation, ranking=ranking,
        warnings=["Credential URL https://alice:pass123@github.com/org/repo?token=ghp_QUERYSECRET1234",
                  "Token=ghp_WARNINGSECRET123456 and /Users/alice/private"],
    )


def test_json_export_has_required_sections_and_sanitizes_sensitive_values():
    result = _analysis()
    payload = json.loads(generate_json_report(
        result, AppConfig(),
        repository_url="https://alice:password123@github.com/org/repo.git?token=ghp_URLSECRET123456#fragment",
    ))
    assert {"repository", "commit", "analysis_timestamp_utc", "versions", "configuration",
            "changed_files", "change_metrics", "dependency_impact", "ml_risk", "explanation",
            "ranking", "limitations", "warnings"} <= set(payload)
    assert payload["repository"]["identifier"] == "https://github.com/org/repo.git"
    assert payload["ranking"]["score_label"] == "heuristic priority score"
    assert payload["ml_risk"]["status"] == "available"
    assert payload["explanation"]["method"] == "SHAP"
    datetime.fromisoformat(payload["analysis_timestamp_utc"])
    serialized = json.dumps(payload)
    for secret in ("ghp_SUPERSECRET123456", "ghp_URLSECRET123456", "ghp_QUERYSECRET1234",
                   "ghp_WARNINGSECRET123456", "dev@example.com", "alice:password123",
                   "/Users/alice/private", "/Users/alice/MyProject", "PRIVATE FILE CONTENT"):
        assert secret not in serialized
    assert "alice" not in payload["repository"]["identifier"]


def test_csv_has_expected_columns_and_one_row_per_ranked_file():
    result = _analysis()
    csv_text = generate_csv_report(result)
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    assert list(rows[0]) == CSV_COLUMNS
    assert len(rows) == len(result.ranking.files) == 1
    assert rows[0]["path"] == "src/mod.py"
    assert rows[0]["heuristic_priority_score"] == "0.71"
    assert "PRIVATE FILE CONTENT" not in csv_text


def test_ml_unavailable_case_is_explicit_and_still_exports_graph_ranking():
    payload = build_report_payload(_analysis(available=False), AppConfig())
    assert payload["ml_risk"]["status"] == "unavailable"
    assert payload["ml_risk"]["score"] is None
    assert "unavailable" in payload["ml_risk"]["interpretation"].lower()
    assert payload["ranking"]["files"]
    assert payload["explanation"]["method"] == "unavailable"
    output = json.dumps(payload)
    assert "/Users/private" not in output
    assert "dev@example.com" not in output


def test_export_failures_are_wrapped_in_clear_messages():
    with pytest.raises(ReportExportError, match="Could not generate JSON report"):
        generate_json_report(None)  # type: ignore[arg-type]
    with pytest.raises(ReportExportError, match="Could not generate CSV report"):
        generate_csv_report(None)  # type: ignore[arg-type]
