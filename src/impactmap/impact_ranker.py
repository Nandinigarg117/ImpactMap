"""Transparent, deterministic heuristic priority ranking for changed-file impact.

Scores summarize evidence and configurable heuristics. They are not probabilities
that a file will fail. The six configured component weights are renormalized over
available evidence. Commit risk is identical for every file, so it cannot reorder
files; it shifts priority scores and lowers the High tier boundary by at most 0.15.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import git

from impactmap.config import (
    AppConfig,
    BROKEN_IMPORT_BOOST,
    COMMIT_RISK_HIGH_TIER_SHIFT,
    DEFAULT_RANKING_WEIGHTS,
    MAX_HISTORY_DEPTH,
    MAGNITUDE_CAP,
    PATH_CAP,
    PRIORITY_TIER_THRESHOLDS,
)
from impactmap.diff_analyzer import CommitDiffSummary, FileChange, classify_file_category
from impactmap.impact_analyzer import ImpactAnalysisResult, ImpactEvidence


@dataclass
class RankedFile:
    rank: int
    path: str
    category: str
    priority_tier: str
    heuristic_priority_score: float
    evidence_level: str
    suggested_action: str
    shortest_dependency_path: List[str]
    component_breakdown: Dict[str, Optional[float]]
    reason: str
    distance: int


@dataclass
class RemovedFile:
    path: str
    category: str
    status: str = "removed"
    note: str = "Removed files cannot be run; inspect dependents for broken imports."


@dataclass
class RankingResult:
    files: List[RankedFile]
    removed: List[RemovedFile]
    effective_weights: Dict[str, float]
    omitted_components: List[str]
    history_reliable: bool
    commit_risk: Optional[float]
    baselines: Dict[str, Dict[str, int]]
    notes: List[str] = field(default_factory=list)
    score_label: str = "heuristic priority score"
    validation_note: str = (
        "No file-level ground truth is available; baseline comparisons are consistency checks against documented rules, not accuracy measurements."
    )


def _bounded_history_counts(repo: git.Repo, paths: Sequence[str], commit_sha: str,
                            max_depth: int) -> Tuple[Optional[Dict[str, int]], bool]:
    """Count commits touching each path within bounded ancestry, excluding selected commit."""
    try:
        if repo.git.rev_parse("--is-shallow-repository").strip().lower() == "true":
            return None, False
        commit = repo.commit(commit_sha)
        if not commit.parents:
            return None, False
        parent = commit.parents[0]
        counts = {path: 0 for path in paths}
        # GitPython's path walk per file is deterministic and bounded.
        for path in paths:
            counts[path] = sum(1 for _ in repo.iter_commits(parent, paths=path, max_count=max_depth))
        return counts, True
    except Exception:
        return None, False


def _normalize_history(counts: Mapping[str, int]) -> Dict[str, float]:
    if not counts:
        return {}
    low, high = min(counts.values()), max(counts.values())
    if high == low:
        return {path: 0.0 for path in counts}
    return {path: (value - low) / (high - low) for path, value in counts.items()}


def _tier(score: float, evidence_level: str, commit_risk: Optional[float]) -> str:
    critical = PRIORITY_TIER_THRESHOLDS["Critical"]
    high = PRIORITY_TIER_THRESHOLDS["High"]
    medium = PRIORITY_TIER_THRESHOLDS["Medium"]
    # Keep Critical stable; higher commit risk lowers the High boundary, rather
    # than claiming that the shared commit score distinguishes individual files.
    if commit_risk is not None:
        high -= max(0.0, min(1.0, commit_risk)) * COMMIT_RISK_HIGH_TIER_SHIFT
    high = max(medium, high)
    if evidence_level == "Weak evidence":
        return "Low" if score < medium else "Medium"
    if score >= critical:
        return "Critical"
    if score >= high:
        return "High"
    if score >= medium:
        return "Medium"
    return "Low"


def _evidence_level(ev: ImpactEvidence, changed: bool) -> str:
    if changed:
        return "Direct change"
    if ev.is_broken_import or ev.category == "direct":
        return "Direct import of changed file"
    if ev.distance >= 2 and ev.distance <= 3:
        return f"Indirect ({ev.distance} hops)"
    return "Weak evidence"


def _action(level: str, broken: bool, changed: bool) -> str:
    if broken:
        return "Inspect import of removed module"
    if changed:
        return "Review diff"
    if level == "Direct import of changed file":
        return "Re-run tests covering this module"
    if level.startswith("Indirect"):
        return "Run focused regression tests"
    return "Consider a smoke test if this area is critical"


def _reason(path: str, ev: ImpactEvidence, level: str, score: float,
            components: Mapping[str, Optional[float]], omitted: Sequence[str]) -> str:
    if ev.is_broken_import:
        basis = ev.broken_import_details or "imports a removed or renamed module"
    elif level == "Direct change":
        basis = "directly changed in the selected commit"
    elif ev.distance > 0:
        basis = f"depends on a changed file at reverse distance {ev.distance}"
    else:
        basis = "no dependency path to a changed file was found"
    active = ", ".join(k for k, v in components.items() if v is not None)
    omission = f" Omitted unavailable components: {', '.join(omitted)}." if omitted else ""
    return f"{path} is prioritized because it {basis}; heuristic priority score {score:.3f} uses {active}.{omission}"


def rank_impacts(
    impact: ImpactAnalysisResult,
    diff: CommitDiffSummary,
    *,
    repo: Optional[git.Repo] = None,
    commit_risk: Optional[float] = None,
    config: Optional[AppConfig] = None,
    weights: Optional[Mapping[str, float]] = None,
    history_reliable: Optional[bool] = None,
    max_history_depth: int = MAX_HISTORY_DEPTH,
) -> RankingResult:
    """Rank impacted and changed files with auditable components and baselines."""
    configured = dict(weights if weights is not None else (config or AppConfig()).ranking_weights)
    names = tuple(DEFAULT_RANKING_WEIGHTS)
    if set(configured) != set(names):
        raise ValueError(f"Weights must define exactly: {', '.join(names)}")
    if any(not math.isfinite(float(v)) or float(v) < 0 for v in configured.values()):
        raise ValueError("Ranking weights must be finite and non-negative")
    if sum(configured.values()) <= 0:
        raise ValueError("At least one ranking weight must be positive")
    if commit_risk is not None and (not math.isfinite(float(commit_risk)) or not 0 <= commit_risk <= 1):
        raise ValueError("commit_risk must be a finite score in [0, 1]")

    changes = {fc.path: fc for fc in diff.file_changes}
    changed_status = {fc.path for fc in diff.file_changes if fc.status in {"added", "modified", "renamed"}}
    deleted = sorted((RemovedFile(fc.path, fc.category) for fc in diff.file_changes if fc.status == "deleted"), key=lambda x: x.path)
    paths = sorted(set(impact.evidence_records) | changed_status)
    evidence = dict(impact.evidence_records)
    for path in paths:
        if path not in evidence:
            evidence[path] = ImpactEvidence(path, "changed" if path in changed_status else "unrelated",
                                            0 if path in changed_status else -1,
                                            [path] if path in changed_status else [], path, [path] if path in changed_status else [],
                                            1 if path in changed_status else 0)

    counts: Optional[Dict[str, int]] = None
    hist_ok = False
    if diff.is_initial_commit or diff.has_truncated_history or history_reliable is False:
        hist_ok = False
    elif repo is not None and history_reliable is not False:
        counts, hist_ok = _bounded_history_counts(repo, paths, diff.commit_sha, max_history_depth)
    elif history_reliable:
        # Explicitly claiming reliable history without counts would fabricate a component.
        hist_ok = False
    history_values = _normalize_history(counts or {}) if hist_ok and counts is not None else {}
    if history_reliable is True and not hist_ok:
        # Caller requested reliability but no evidence source was supplied.
        hist_ok = False

    usable = dict(configured)
    omitted: List[str] = []
    if not hist_ok:
        usable.pop("history", None)
        omitted.append("history")
    if commit_risk is None:
        usable.pop("commit_risk", None)
        omitted.append("commit_risk")
    weight_total = sum(usable.values())
    effective = {k: v / weight_total for k, v in usable.items()} if weight_total else {}

    # Retain each changed seed's line churn for path-specific magnitude.
    churn = {fc.path: max(0, fc.lines_added) + max(0, fc.lines_deleted)
             for fc in diff.file_changes if fc.status in {"added", "modified", "renamed"}}
    provisional = []
    for path in paths:
        ev = evidence[path]
        changed = path in changed_status
        distance = 0 if changed else ev.distance
        reachable = list(ev.reachable_seeds)
        if changed:
            magnitude = min(1.0, math.log1p(churn.get(path, 0)) / math.log1p(MAGNITUDE_CAP))
        elif ev.is_broken_import:
            magnitude = 0.0
        else:
            seed_distances = ev.seed_distances or {s: distance for s in reachable}
            reached = [s for s in reachable if s in churn and seed_distances.get(s, distance) > 0]
            if reached:
                seed = max(reached, key=lambda s: (churn[s], s))
                seed_distance = seed_distances.get(seed, distance)
                proximity_scale = 1.0 / seed_distance if seed_distance > 0 else 1.0
                magnitude = min(1.0, math.log1p(churn[seed]) / math.log1p(MAGNITUDE_CAP)) * proximity_scale
            else:
                magnitude = 0.0
        num_paths = ev.path_count + max(0, len(reachable) - 1)
        path_support = min(1.0, math.log1p(num_paths) / math.log1p(PATH_CAP)) if num_paths else 0.0
        if ev.is_broken_import:
            path_support = max(path_support, min(1.0, math.log1p(2) / math.log1p(PATH_CAP)))
        proximity = 1.0 if changed else (1.0 / distance if distance > 0 else 0.0)
        components: Dict[str, Optional[float]] = {
            "status": 1.0 if changed else 0.0,
            "proximity": proximity,
            "path_support": path_support,
            "change_magnitude": magnitude,
            "history": history_values.get(path) if hist_ok else None,
            "commit_risk": float(commit_risk) if commit_risk is not None else None,
        }
        score = sum(effective[k] * float(components[k]) for k in effective)
        boost = BROKEN_IMPORT_BOOST if ev.is_broken_import else 0.0
        score = min(1.0, max(0.0, score + boost))
        level = _evidence_level(ev, changed)
        tier = _tier(score, level, commit_risk)
        category = changes[path].category if path in changes else classify_file_category(path)
        components["broken_import_boost"] = boost
        reason = _reason(path, ev, level, score, components, omitted)
        shortest_path = list(ev.shortest_path)
        if ev.is_broken_import:
            shortest_path = [path, "[removed/renamed module]"]
        provisional.append({"path": path, "category": category, "score": score, "tier": tier,
                            "level": level, "action": _action(level, ev.is_broken_import, changed),
                            "shortest": shortest_path, "components": components,
                            "reason": reason, "distance": distance, "changed": changed})

    # Baselines share deterministic tie rules. Distance baseline prioritizes changed,
    # then shorter reverse distance; changed-status baseline prioritizes changed only.
    full_sorted = sorted(provisional, key=lambda x: (-x["score"], x["distance"] if x["distance"] >= 0 else math.inf, x["path"]))
    distance_sorted = sorted(provisional, key=lambda x: (0 if x["changed"] else 1,
                                                          x["distance"] if x["distance"] >= 0 else math.inf,
                                                          x["path"]))
    status_sorted = sorted(provisional, key=lambda x: (0 if x["changed"] else 1, x["path"]))
    def ranks(rows):
        return {row["path"]: i + 1 for i, row in enumerate(rows)}
    ranked = [RankedFile(i + 1, row["path"], row["category"], row["tier"], row["score"],
                         row["level"], row["action"], row["shortest"], row["components"],
                         row["reason"], row["distance"]) for i, row in enumerate(full_sorted)]
    notes = []
    if not hist_ok:
        notes.append("History unavailable or unreliable; its weight was omitted and remaining weights renormalized.")
    if commit_risk is None:
        notes.append("Commit risk unavailable; its weight was omitted and remaining weights renormalized.")
    notes.append("Commit risk is shared across files and cannot reorder files within this commit.")
    return RankingResult(ranked, deleted, effective, omitted, hist_ok, commit_risk,
                         {"dependency_distance": ranks(distance_sorted),
                          "changed_status": ranks(status_sorted),
                          "full_combined_score": ranks(full_sorted)}, notes)
