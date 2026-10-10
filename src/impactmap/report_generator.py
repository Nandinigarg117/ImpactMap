"""Sanitized JSON and CSV exports for completed ImpactMap analyses."""

import csv
from datetime import datetime, timezone
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from impactmap import __version__ as APP_VERSION
from impactmap.analysis import AnalysisResult
from impactmap.config import AppConfig


class ReportExportError(ValueError):
    """Raised when a report cannot be safely serialized."""


def sanitize_url(url: str) -> str:
    """Keep only a URL's public scheme, host, and path; discard userinfo/query/fragment."""
    try:
        parsed = urlsplit(str(url).strip())
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            return ""
        host = parsed.hostname.lower()
        if parsed.port:
            host = f"{host}:{parsed.port}"
        path = re.sub(r"/{2,}", "/", parsed.path)
        return urlunsplit((parsed.scheme.lower(), host, path, "", ""))
    except (ValueError, TypeError):
        return ""


def sanitize_text(value: Any) -> str:
    """Remove common credential, email, and absolute-path forms from export text."""
    text = str(value)
    # Normalize URLs before applying the generic absolute-path redactor.
    url_pattern = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
    def clean_url(match):
        candidate = match.group(0).rstrip(".,;)]}")
        trailing = match.group(0)[len(candidate):]
        return sanitize_url(candidate) + trailing
    text = url_pattern.sub(clean_url, text)
    text = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*", "Bearer [redacted]", text)
    text = re.sub(r"(?i)\b(?:gh[pousr]_[A-Za-z0-9_]{8,}|github_pat_[A-Za-z0-9_]{8,}|glpat-[A-Za-z0-9_-]{8,}|xox[baprs]-[A-Za-z0-9-]{8,})\b", "[redacted token]", text)
    text = re.sub(r"(?i)\b(AKIA[A-Z0-9]{12,})\b", "[redacted key]", text)
    text = re.sub(r"(?i)\b(password|passwd|token|secret|access[_-]?key)\s*([=:])\s*([^\s,;]+)", r"\1\2[redacted]", text)
    text = re.sub(r"\b[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[email redacted]", text)
    # File paths in reports are repository-relative. Any path beginning at the
    # filesystem root is private machine context and is omitted.
    text = re.sub(r"(?<![\w:/])/(?:[^\s,;()\[\]{}]+/?)+", "[local path]", text)
    text = re.sub(r"(?i)\b[A-Z]:\\(?:[^\s,;()\[\]{}]+\\?)+", "[local path]", text)
    return text


def _repository_identifier(result: AnalysisResult, repository_url: Optional[str]) -> str:
    if repository_url:
        public = sanitize_url(repository_url)
        if public:
            return public
    identity = str(result.repository_identity).rstrip("/\\")
    name = Path(identity).name
    if name == ".git":
        name = Path(identity).parent.name
    return sanitize_text(name or "local-repository")


def _clean(value: Any) -> Any:
    """Recursively sanitize strings in structured user/repository-derived data."""
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, Mapping):
        return {sanitize_text(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_clean(v) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize_text(value)


def build_report_payload(result: AnalysisResult, config: Optional[AppConfig] = None,
                         repository_url: Optional[str] = None) -> Dict[str, Any]:
    """Build the full report schema without commit contents, author, or local paths."""
    cfg = config or AppConfig()
    prediction = result.prediction
    if prediction.available:
        risk = {
            "status": "available",
            "label": "commit-level risk score",
            "score": prediction.score,
            "tier": prediction.tier,
            "threshold_used": prediction.threshold_used,
            "interpretation": (
                "This score describes commit-level model output. It is not a file-level failure probability, "
                "does not identify a defect, and does not replace graph evidence."
            ),
        }
    else:
        risk = {
            "status": "unavailable",
            "label": "commit-level risk score",
            "score": None,
            "tier": "unavailable",
            "threshold_used": None,
            "interpretation": "The ML risk score is unavailable; graph evidence and heuristic priority ranking remain separate and available.",
            "reason": prediction.error or "The model did not provide a score.",
        }

    changed_files = [{
        "path": fc.path,
        "status": fc.status,
        "category": fc.category,
        "lines_added": fc.lines_added,
        "lines_deleted": fc.lines_deleted,
        "is_binary": fc.is_binary,
    } for fc in result.diff.file_changes]
    dependency_records = []
    for path, ev in sorted(result.impact.evidence_records.items()):
        dependency_records.append({
            "path": path,
            "impact_category": ev.category,
            "reverse_distance": ev.distance,
            "shortest_dependency_path": ev.shortest_path,
            "reachable_changed_seeds": ev.reachable_seeds,
            "path_count": ev.path_count,
            "is_broken_import": ev.is_broken_import,
            "broken_import_details": ev.broken_import_details,
        })

    explanations = result.explanation
    explanation_summary = {
        "available": explanations.available,
        "method": explanations.local_method or "unavailable",
        "global_method": explanations.global_method or "unavailable",
        "top_features": [{
            "feature": feature.feature,
            "value": feature.value,
            "contribution": feature.contribution,
            "direction": feature.direction,
            "magnitude": feature.magnitude,
            "summary": feature.text,
        } for feature in sorted(explanations.features, key=lambda x: abs(x.contribution), reverse=True)[:5]],
        "global_importance": explanations.global_importance,
        "unavailable_reason": explanations.error,
    }
    ranked_files = [{
        "rank": item.rank,
        "path": item.path,
        "category": item.category,
        "priority_tier": item.priority_tier,
        "heuristic_priority_score": item.heuristic_priority_score,
        "evidence_level": item.evidence_level,
        "distance": item.distance,
        "suggested_action": item.suggested_action,
        "shortest_dependency_path": item.shortest_dependency_path,
        "component_breakdown": item.component_breakdown,
        "reason": item.reason,
    } for item in result.ranking.files]
    removed_files = [{"path": item.path, "category": item.category,
                      "status": item.status, "note": item.note}
                     for item in result.ranking.removed]
    report = {
        "report_version": "1.0",
        "repository": {"identifier": _repository_identifier(result, repository_url),
                       "identifier_kind": "public_url" if repository_url and sanitize_url(repository_url) else "local_folder_name"},
        "commit": {
            "hash": result.commit_sha,
            "is_merge_commit": result.is_merge_commit,
            "parent_hashes": result.parent_shas,
        },
        "analysis_timestamp_utc": result.analysis_timestamp_utc or datetime.now(timezone.utc).isoformat(),
        "versions": {"app": APP_VERSION, "model": prediction.model_version or "unavailable"},
        "configuration": {
            "max_file_bytes": cfg.max_file_bytes,
            "max_files": cfg.max_files,
            "max_history_depth": cfg.max_history_depth,
            "max_depth": cfg.max_depth,
            "max_paths_per_node": cfg.max_paths_per_node,
            "excluded_directories": sorted(cfg.excluded_dirs),
            "ranking_weights_configured": cfg.ranking_weights,
            "ranking_weights_effective": result.ranking.effective_weights,
            "omitted_ranking_components": result.ranking.omitted_components,
            "config_hash": cfg.config_hash(),
        },
        "changed_files": changed_files,
        "change_metrics": {
            "files_changed": result.diff.files_changed_count,
            "lines_added": result.diff.lines_added_total,
            "lines_deleted": result.diff.lines_deleted_total,
            "total_churn": result.diff.total_churn,
            "is_initial_commit": result.diff.is_initial_commit,
            "is_merge_commit": result.diff.is_merge_commit,
            "is_empty_commit": result.diff.is_empty_commit,
            "history_truncated": result.diff.has_truncated_history,
        },
        "dependency_impact": {
            "edge_convention": "A -> B means A imports B",
            "graph": {"nodes": result.graph.total_nodes, "edges": result.graph.total_edges,
                      "parse_failures": _clean(result.graph.parse_failures),
                      "unresolved_imports": _clean(result.graph.unresolved_imports),
                      "truncated": result.graph.truncated},
            "parent_graph": ({"nodes": result.parent_graph.total_nodes,
                              "edges": result.parent_graph.total_edges,
                              "parse_failures": _clean(result.parent_graph.parse_failures),
                              "unresolved_imports": _clean(result.parent_graph.unresolved_imports)}
                             if result.parent_graph else None),
            "summary": {"changed_files": result.impact.changed_files_count,
                        "direct_dependents": result.impact.direct_dependents_count,
                        "indirect_dependents": result.impact.indirect_dependents_count,
                        "unrelated_files": result.impact.unrelated_count,
                        "max_distance": result.impact.max_distance},
            "evidence": dependency_records,
            "removed_files": removed_files,
        },
        "ml_risk": risk,
        "explanation": explanation_summary,
        "ranking": {
            "score_label": "heuristic priority score",
            "interpretation": "Graph evidence and commit-level ML risk are distinct; the ML risk factor is shared by all files and cannot reorder them.",
            "files": ranked_files,
            "baselines": result.ranking.baselines,
        },
        "limitations": [
            "Static Python import analysis cannot resolve dynamic imports, reflection, framework wiring, runtime plugin loading, or non-Python code.",
            "A commit-level risk score is not a file-level failure probability; a high score does not prove a defect and a low score does not guarantee safety.",
            "The model was trained on ApacheJIT Apache-project data; domain shift may affect results on other repositories.",
            result.ranking.validation_note,
        ],
        "warnings": list(result.warnings) + list(result.ranking.notes) + list(explanations.notes),
    }
    return _clean(report)


def generate_json_report(result: AnalysisResult, config: Optional[AppConfig] = None,
                         repository_url: Optional[str] = None) -> str:
    """Serialize a sanitized full JSON report or raise a user-safe export error."""
    try:
        return json.dumps(build_report_payload(result, config, repository_url), indent=2,
                          ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        raise ReportExportError(f"Could not generate JSON report: {sanitize_text(exc)}") from exc


CSV_COLUMNS = [
    "rank", "path", "category", "priority_tier", "heuristic_priority_score",
    "evidence_level", "distance", "suggested_action", "shortest_dependency_path",
    "component_breakdown", "reason",
]


def generate_csv_report(result: AnalysisResult) -> str:
    """Serialize ranked files only, one ranked file per CSV row."""
    try:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for item in result.ranking.files:
            row = {
                "rank": item.rank,
                "path": item.path,
                "category": item.category,
                "priority_tier": item.priority_tier,
                "heuristic_priority_score": item.heuristic_priority_score,
                "evidence_level": item.evidence_level,
                "distance": item.distance,
                "suggested_action": item.suggested_action,
                "shortest_dependency_path": " -> ".join(item.shortest_dependency_path),
                "component_breakdown": json.dumps(item.component_breakdown, ensure_ascii=False, allow_nan=False),
                "reason": item.reason,
            }
            writer.writerow(_clean(row))
        return stream.getvalue()
    except Exception as exc:
        raise ReportExportError(f"Could not generate CSV report: {sanitize_text(exc)}") from exc
