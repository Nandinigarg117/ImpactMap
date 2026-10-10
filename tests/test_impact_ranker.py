"""Deterministic checks for impact ranking rules and comparison baselines."""

from impactmap.diff_analyzer import CommitDiffSummary, FileChange
from impactmap.impact_analyzer import ImpactAnalysisResult, ImpactEvidence
from impactmap.impact_ranker import rank_impacts


def _fc(path, status="modified", added=20, deleted=0, category="python_source"):
    return FileChange(path, None, status, added, deleted, False, category)


def _ev(path, category, distance, *, seeds=None, paths=1, broken=False, seed_distances=None):
    shortest = [path, "src/mid.py", "src/seed.py"] if distance == 2 else ([path, "src/seed.py"] if distance == 1 else [])
    return ImpactEvidence(path, category, distance, shortest, " -> ".join(shortest),
                          seeds or [], paths, broken, "Imports removed module 'src/gone.py'" if broken else None,
                          False, seed_distances or {})


def _analysis(records):
    return ImpactAnalysisResult("fake-sha", len(records), 0, 1, 1, 1, 0, 8, 0.0,
                                evidence_records=records)


def _diff(*changes):
    return CommitDiffSummary("fake-sha", "parent", False, False, False, False,
                             len(changes), sum(c.lines_added for c in changes),
                             sum(c.lines_deleted for c in changes),
                             sum(c.lines_added + c.lines_deleted for c in changes), list(changes), [])


def test_order_changed_direct_indirect_and_distant():
    records = {
        "src/changed.py": _ev("src/changed.py", "changed", 0, seeds=["src/changed.py"]),
        "src/direct.py": _ev("src/direct.py", "direct", 1, seeds=["src/changed.py"]),
        "src/indirect.py": _ev("src/indirect.py", "indirect", 2, seeds=["src/changed.py"]),
        "src/distant.py": _ev("src/distant.py", "indirect", 4, seeds=["src/changed.py"]),
    }
    result = rank_impacts(_analysis(records), _diff(_fc("src/changed.py", added=80)), history_reliable=False)
    assert [f.path for f in result.files] == ["src/changed.py", "src/direct.py", "src/indirect.py", "src/distant.py"]
    assert result.files[1].evidence_level == "Direct import of changed file"
    assert result.files[2].evidence_level == "Indirect (2 hops)"
    assert result.files[3].evidence_level == "Weak evidence"


def test_missing_ml_and_history_renormalize_remaining_weights():
    records = {"src/changed.py": _ev("src/changed.py", "changed", 0, seeds=["src/changed.py"])}
    result = rank_impacts(_analysis(records), _diff(_fc("src/changed.py")), history_reliable=False)
    assert set(result.omitted_components) == {"history", "commit_risk"}
    assert abs(sum(result.effective_weights.values()) - 1.0) < 1e-12
    assert "history" not in result.effective_weights and "commit_risk" not in result.effective_weights
    # Present commit risk participates identically for every file, and the weights
    # still normalize when history remains unavailable.
    with_risk = rank_impacts(_analysis(records), _diff(_fc("src/changed.py")),
                             commit_risk=.9, history_reliable=False)
    assert "commit_risk" in with_risk.effective_weights
    assert abs(sum(with_risk.effective_weights.values()) - 1.0) < 1e-12


def test_ties_break_by_distance_then_path():
    records = {
        "src/z_direct.py": _ev("src/z_direct.py", "direct", 1, seeds=["src/changed.py"]),
        "src/a_direct.py": _ev("src/a_direct.py", "direct", 1, seeds=["src/changed.py"]),
        "src/changed.py": _ev("src/changed.py", "changed", 0, seeds=["src/changed.py"]),
    }
    zero_churn = _diff(_fc("src/changed.py", added=0))
    # Equalize all direct evidence and assert deterministic path tie order.
    result = rank_impacts(_analysis(records), zero_churn, history_reliable=False,
                          weights={"status": 0, "proximity": 1, "path_support": 0,
                                   "change_magnitude": 0, "history": 0, "commit_risk": 0})
    direct_paths = [f.path for f in result.files if f.distance == 1]
    assert direct_paths == ["src/a_direct.py", "src/z_direct.py"]


def test_weak_evidence_never_gets_high_tier():
    records = {"src/far.py": _ev("src/far.py", "indirect", 8, seeds=["src/changed.py"], paths=10)}
    result = rank_impacts(_analysis(records), _diff(_fc("src/changed.py")),
                          commit_risk=1.0, history_reliable=False)
    far = next(f for f in result.files if f.path == "src/far.py")
    assert far.priority_tier not in {"High", "Critical"}


def test_removed_file_is_separate_and_broken_import_gets_boost():
    records = {
        "src/changed.py": _ev("src/changed.py", "changed", 0, seeds=["src/changed.py"]),
        "src/broken_caller.py": _ev("src/broken_caller.py", "direct", 1, broken=True),
        "src/ordinary_caller.py": _ev("src/ordinary_caller.py", "direct", 1, seeds=["src/changed.py"]),
    }
    result = rank_impacts(_analysis(records), _diff(_fc("src/changed.py"),
                                                    _fc("src/gone.py", "deleted", added=0)),
                          history_reliable=False)
    broken = next(f for f in result.files if f.path == "src/broken_caller.py")
    ordinary = next(f for f in result.files if f.path == "src/ordinary_caller.py")
    assert broken.component_breakdown["broken_import_boost"] > 0
    assert broken.heuristic_priority_score > ordinary.heuristic_priority_score
    assert broken.suggested_action == "Inspect import of removed module"
    assert [f.path for f in result.removed] == ["src/gone.py"]


def test_three_baselines_are_deterministic_and_follow_their_rules():
    records = {
        "src/changed.py": _ev("src/changed.py", "changed", 0, seeds=["src/changed.py"]),
        "src/direct.py": _ev("src/direct.py", "direct", 1, seeds=["src/changed.py"]),
        "src/indirect.py": _ev("src/indirect.py", "indirect", 2, seeds=["src/changed.py"]),
    }
    result = rank_impacts(_analysis(records), _diff(_fc("src/changed.py", added=100)),
                          history_reliable=False)
    dist = result.baselines["dependency_distance"]
    status = result.baselines["changed_status"]
    full = result.baselines["full_combined_score"]
    assert dist["src/changed.py"] < dist["src/direct.py"] < dist["src/indirect.py"]
    assert status["src/changed.py"] == 1
    assert status["src/direct.py"] < status["src/indirect.py"]  # deterministic path order
    assert full == {f.path: f.rank for f in result.files}
    assert "not accuracy measurements" in result.validation_note
