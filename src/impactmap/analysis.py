"""Shared end-to-end analysis orchestration used by the UI and integration tests."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import git

from impactmap.config import AppConfig
from impactmap.dependency_graph import GraphBuildResult, build_dependency_graph
from impactmap.diff_analyzer import CommitDiffSummary, analyze_commit_diff
from impactmap.explainer import ExplanationResult, explain_prediction
from impactmap.feature_extractor import ExtractedFeatures, extract_commit_features
from impactmap.impact_analyzer import ImpactAnalysisResult, analyze_impact
from impactmap.impact_ranker import RankingResult, rank_impacts
from impactmap.predictor import PredictionResult, load_model, predict_risk

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    repository_identity: str
    commit_sha: str
    commit_message: str
    author: str
    authored_datetime: str
    is_merge_commit: bool
    parent_shas: List[str]
    diff: CommitDiffSummary
    graph: GraphBuildResult
    parent_graph: Optional[GraphBuildResult]
    impact: ImpactAnalysisResult
    features: Optional[ExtractedFeatures]
    prediction: PredictionResult
    explanation: ExplanationResult
    ranking: RankingResult
    analysis_timestamp_utc: str = ""
    warnings: List[str] = field(default_factory=list)
    cache_key: Tuple[str, str, str, str] = ()


_ANALYSIS_CACHE: Dict[Tuple[str, str, str, str], AnalysisResult] = {}
_ANALYSIS_CACHE_LIMIT = 16


def _repo_identity(repo: git.Repo) -> str:
    try:
        return str(Path(repo.git_dir).resolve())
    except Exception:
        return str(repo.working_dir or repo.git_dir)


def _model_identity() -> str:
    """Stable model version plus artifact stat so replacing a model invalidates cache."""
    model_path = Path(__file__).resolve().parents[2] / "models/model.joblib"
    try:
        loaded = load_model()
        stat = model_path.stat()
        return f"{loaded.artifact.get('model_version', 'unknown')}:{stat.st_size}:{stat.st_mtime_ns}"
    except Exception as exc:
        try:
            stat = model_path.stat()
            stamp = f"{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            stamp = "missing"
        logger.warning("Model unavailable while deriving analysis cache version: %s", exc)
        return f"unavailable:{stamp}"


def clear_analysis_cache() -> None:
    """Clear cached orchestrator results, primarily useful after model/config changes."""
    _ANALYSIS_CACHE.clear()


def run_analysis(repo: git.Repo, commit_sha: str, config: Optional[AppConfig]) -> AnalysisResult:
    """Run diff, graph/impact, commit features, optional ML/explanation, and ranking.

    A model or feature-extraction failure is represented in ``prediction`` and does
    not discard the diff, graph, impact, or graph-only ranking.
    """
    cfg = config or AppConfig()
    identity = _repo_identity(repo)
    model_version = _model_identity()
    key = (identity, commit_sha, cfg.config_hash(), model_version)
    cached = _ANALYSIS_CACHE.get(key)
    if cached is not None:
        return cached

    diff = analyze_commit_diff(repo, commit_sha)
    analysis_started_at = datetime.now(timezone.utc).isoformat()
    warnings: List[str] = []
    graph = build_dependency_graph(repo, commit_sha, config=cfg)
    if graph.parse_failures:
        warnings.append(f"{len(graph.parse_failures)} Python file(s) could not be parsed.")
    if graph.unresolved_imports:
        unresolved = sum(map(len, graph.unresolved_imports.values()))
        warnings.append(f"{unresolved} internal import(s) could not be resolved.")

    deleted_or_renamed = any(fc.status in {"deleted", "renamed"} for fc in diff.file_changes)
    parent_graph = None
    if deleted_or_renamed and diff.parent_sha and not diff.is_initial_commit:
        try:
            parent_graph = build_dependency_graph(repo, diff.parent_sha, config=cfg)
        except Exception as exc:
            warnings.append(f"Parent dependency graph unavailable for removed/renamed files: {exc}")
    impact = analyze_impact(repo, diff, config=cfg)
    warnings.extend(w for w in impact.warnings if w not in warnings)

    features: Optional[ExtractedFeatures] = None
    try:
        features = extract_commit_features(repo, commit_sha, diff_summary=diff,
                                           max_history_depth=cfg.max_history_depth)
        warnings.extend(features.notes)
    except Exception as exc:
        warnings.append(f"Commit features unavailable: {exc}")

    if features is not None:
        try:
            prediction = predict_risk(features)
        except Exception as exc:
            logger.exception("Commit risk prediction failed")
            prediction = PredictionResult(False, error=f"Risk score unavailable: {exc}")
    else:
        prediction = PredictionResult(False, error="Feature extraction failed; risk score unavailable.")
    if not prediction.available and prediction.error:
        warnings.append(prediction.error)

    if features is not None and prediction.available:
        try:
            explanation = explain_prediction(features)
        except Exception as exc:
            logger.exception("Commit explanation failed")
            explanation = ExplanationResult(False, error=f"Explanation unavailable: {exc}")
    else:
        explanation = ExplanationResult(False, error="Explanation unavailable because the ML prediction is unavailable.")

    ranking = rank_impacts(impact, diff, repo=repo,
                           commit_risk=prediction.score if prediction.available else None,
                           config=cfg, max_history_depth=cfg.max_history_depth)
    commit = repo.commit(commit_sha)
    result = AnalysisResult(
        repository_identity=identity,
        commit_sha=commit.hexsha,
        commit_message=(commit.message or "").strip().splitlines()[0] if commit.message else "(no commit message)",
        author=commit.author.name if commit.author else "Unknown",
        authored_datetime=commit.authored_datetime.isoformat(),
        is_merge_commit=len(commit.parents) > 1,
        parent_shas=[p.hexsha for p in commit.parents],
        diff=diff,
        graph=graph,
        parent_graph=parent_graph,
        impact=impact,
        features=features,
        prediction=prediction,
        explanation=explanation,
        ranking=ranking,
        analysis_timestamp_utc=analysis_started_at,
        warnings=warnings,
        cache_key=key,
    )
    if len(_ANALYSIS_CACHE) >= _ANALYSIS_CACHE_LIMIT:
        _ANALYSIS_CACHE.pop(next(iter(_ANALYSIS_CACHE)))
    _ANALYSIS_CACHE[key] = result
    return result
