"""Streamlit presentation layer. Analysis work lives in impactmap.analysis."""

from __future__ import annotations

import html
import json
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import git
import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from impactmap.analysis import AnalysisResult, run_analysis
from impactmap.config import (
    AppConfig,
    DEFAULT_RANKING_WEIGHTS,
    EXCLUDED_DIRS,
    MAX_DIFF_LINES,
    MAX_FILES,
)
from impactmap.repository import list_recent_commits, safe_clone_repo, validate_and_open_local_repo


st.set_page_config(page_title="ImpactMap", page_icon="IM", layout="wide")

SEMANTIC_COLORS = {
    "changed": "#f5a623",
    "direct": "#ff6b6b",
    "indirect": "#41b6c4",
    "unrelated": "#8b949e",
}
CATEGORY_COLORS = {
    "python_source": SEMANTIC_COLORS["indirect"],
    "test": "#a78bfa",
    "config": "#f0c674",
    "docs": "#8b949e",
    "generated_vendored": "#d2a8ff",
    "binary": "#6e7681",
    "other_source": "#79c0ff",
    "other": "#9da7b3",
}
LIMITATIONS = (
    "ImpactMap uses static Python import analysis. Dynamic imports, reflection, framework wiring, "
    "runtime plugin loading, and non-Python code can create relationships this graph cannot see. "
    "A risk score is a commit-level model output, not proof of a defect; a low score does not guarantee safety."
)


def safe_text(value: object) -> str:
    """Escape repository-controlled text before it is passed to rich UI surfaces."""
    return html.escape(str(value), quote=True)


@st.cache_resource(show_spinner=False)
def _open_local_repo(path: str):
    return validate_and_open_local_repo(path)


@st.cache_resource(show_spinner=False)
def _clone_remote_repo(url: str):
    target = Path(tempfile.gettempdir()) / "impactmap-clones"
    target.mkdir(parents=True, exist_ok=True)
    return safe_clone_repo(url, target)


@st.cache_data(show_spinner=False, ttl=60)
def _commit_rows(repo_path: str, _head_sha: str, search: str):
    repo = git.Repo(repo_path)
    return [c.__dict__ for c in list_recent_commits(repo, max_count=300, search_query=search or None)]


@st.cache_data(show_spinner=False, ttl=300)
def _metrics_data():
    path = Path(__file__).resolve().parent / "models/metrics.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _config_from_controls() -> AppConfig:
    weights = {name: float(st.session_state.get(f"weight_{name}", default))
               for name, default in DEFAULT_RANKING_WEIGHTS.items()}
    raw_excluded = st.session_state.get("excluded_dirs_text", "\n".join(sorted(EXCLUDED_DIRS)))
    excluded = frozenset(line.strip() for line in raw_excluded.splitlines() if line.strip())
    return AppConfig(max_files=int(st.session_state.get("max_files", MAX_FILES)),
                     max_depth=int(st.session_state.get("max_depth", 10)),
                     excluded_dirs=excluded,
                     ranking_weights=weights)


def _clear_analysis():
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_error", None)


def _load_repo():
    local = st.session_state.get("local_repo_path", "").strip()
    remote = st.session_state.get("remote_repo_url", "").strip()
    try:
        if local:
            repo = _open_local_repo(local)
            path = str(Path(repo.working_dir).resolve())
        elif remote:
            repo = _clone_remote_repo(remote)
            path = str(Path(repo.working_dir).resolve())
        else:
            raise ValueError("Enter a local Git repository path or a public HTTPS repository URL.")
        st.session_state.repo_path = path
        st.session_state.repo_loaded = True
        st.session_state.repo_label = Path(path).name
        st.session_state.repo_head = repo.head.commit.hexsha
        st.session_state.repo_load_error = None
        _clear_analysis()
    except Exception as exc:
        st.session_state.repo_loaded = False
        st.session_state.repo_load_error = str(exc)


def _plot_dependency_graph(result: AnalysisResult, max_distance: int, impact_filter: list[str],
                           category_filter: list[str], full_graph: bool, cap: int = 180):
    graph = result.graph.graph
    changed = {fc.path for fc in result.diff.file_changes
               if fc.status in {"added", "modified", "renamed"} and fc.path in graph}
    reverse = graph.reverse(copy=False)
    distances = {path: ev.distance for path, ev in result.impact.evidence_records.items()}
    included = set(graph.nodes)
    if not full_graph:
        included = set(changed)
        queue = [(p, 0) for p in sorted(changed) if p in graph]
        while queue:
            node, depth = queue.pop(0)
            if depth >= max_distance:
                continue
            for dependent in sorted(reverse.successors(node)):
                if dependent not in included:
                    included.add(dependent)
                    queue.append((dependent, depth + 1))
        if not included:
            included = set(sorted(graph.nodes)[:cap])
    if len(included) > cap:
        included = set(sorted(included, key=lambda p: (distances.get(p, 10**6), p))[:cap])
    filtered = []
    for node in included:
        ev = result.impact.evidence_records.get(node)
        impact_cat = "changed" if node in changed else (ev.category if ev else "unrelated")
        file_cat = graph.nodes[node].get("category", "other")
        if impact_cat not in impact_filter or file_cat not in category_filter:
            continue
        distance = 0 if node in changed else distances.get(node, -1)
        if distance >= 0 and distance > max_distance:
            continue
        filtered.append(node)
    visible = set(filtered)
    subgraph = graph.subgraph(visible).copy()
    if not subgraph.nodes:
        st.info("No graph nodes match these filters. Broaden the filters or select the full graph.")
        return
    pos = nx.spring_layout(subgraph, seed=42, k=0.8, iterations=35)
    edge_x, edge_y = [], []
    for a, b in sorted(subgraph.edges()):
        edge_x.extend([pos[a][0], pos[b][0], None])
        edge_y.extend([pos[a][1], pos[b][1], None])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#58616d"),
                             hoverinfo="skip", showlegend=False))
    groups = {key: [] for key in SEMANTIC_COLORS}
    for node in sorted(subgraph.nodes()):
        ev = result.impact.evidence_records.get(node)
        key = "changed" if node in changed else (ev.category if ev and ev.category in groups else "unrelated")
        groups[key].append(node)
    for key, nodes in groups.items():
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
            text=[safe_text(Path(n).name) for n in nodes], textposition="top center",
            marker=dict(size=12, color=SEMANTIC_COLORS[key], line=dict(width=1, color="#10151c")),
            name=key.title(),
            customdata=[[safe_text(n), safe_text(graph.nodes[n].get("category", "other")),
                         distances.get(n, 0 if n in changed else -1),
                         graph.in_degree(n), graph.out_degree(n)] for n in nodes],
            hovertemplate="%{customdata[0]}<br>Impact: " + key + "<br>File category: %{customdata[1]}<br>Distance: %{customdata[2]}<br>Imports into: %{customdata[3]}<br>Imports: %{customdata[4]}<extra></extra>"
        ))
    fig.update_layout(template="plotly_dark", height=540, margin=dict(l=10, r=10, t=20, b=10),
                      paper_bgcolor="#111820", plot_bgcolor="#111820", legend=dict(orientation="h"),
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)


def _overview_tab(result: AnalysisResult):
    st.subheader(safe_text(result.repository_identity.rsplit("/", 1)[-1]))
    st.write(safe_text(result.commit_message))
    st.caption(f"Repository: {safe_text(result.repository_identity)}")
    st.caption(f"Commit: `{safe_text(result.commit_sha)}`")
    direct = result.impact.direct_dependents_count
    indirect = result.impact.indirect_dependents_count
    risk = f"{result.prediction.tier} · {result.prediction.score:.3f}" if result.prediction.available else "Unavailable"
    cols = st.columns(5)
    cols[0].metric("Files changed", result.diff.files_changed_count)
    cols[1].metric("Lines added", result.diff.lines_added_total)
    cols[2].metric("Lines deleted", result.diff.lines_deleted_total)
    cols[3].metric("Graph · nodes / edges", f"{result.graph.total_nodes} / {result.graph.total_edges}")
    cols[4].metric("ML risk tier", risk)
    st.caption(f"Graph candidates: {direct} direct · {indirect} indirect")
    st.markdown("#### Key findings")
    findings = []
    if result.diff.is_merge_commit:
        findings.append("This is a merge commit; the diff is compared with its first parent.")
    if result.impact.evidence_records:
        top = next((f for f in result.ranking.files if f.evidence_level == "Direct change"), None)
        if top:
            findings.append(f"{safe_text(top.path)} is directly changed and ranks first evidence-wise.")
        if result.impact.direct_dependents_count:
            findings.append(f"Static imports identify {direct} direct and {indirect} indirect dependent files.")
    if result.prediction.available:
        findings.append(f"The commit risk score is {result.prediction.score:.3f} ({result.prediction.tier}).")
    else:
        findings.append(f"ML risk is unavailable: {safe_text(result.prediction.error or 'No prediction was produced.')}")
    if not findings:
        findings.append("No changed Python files or dependent modules were found in this commit.")
    for item in findings:
        st.write(f"• {item}")
    st.info(LIMITATIONS, icon="ℹ️")
    for warning in result.warnings:
        st.warning(safe_text(warning))


def _commit_tab(result: AnalysisResult, repo: git.Repo):
    commit = repo.commit(result.commit_sha)
    st.markdown("#### Commit metadata")
    st.write(f"Author: {safe_text(result.author)}")
    st.write(f"Authored: {safe_text(result.authored_datetime)}")
    st.write(f"Parents: {', '.join(safe_text(p) for p in result.parent_shas) or 'None (initial commit)'}")
    st.write(f"Commit message: {safe_text(result.commit_message)}")
    rows = [{"path": safe_text(fc.path), "status": fc.status, "category": fc.category,
             "added": fc.lines_added, "deleted": fc.lines_deleted, "binary": fc.is_binary}
            for fc in result.diff.file_changes]
    if not rows:
        st.info("This commit has no file changes.")
        return
    status_filter = st.multiselect("Statuses", sorted({r["status"] for r in rows}), default=sorted({r["status"] for r in rows}), key="diff_status_filter")
    category_filter = st.multiselect("File categories", sorted({r["category"] for r in rows}), default=sorted({r["category"] for r in rows}), key="diff_category_filter")
    shown = [r for r in rows if r["status"] in status_filter and r["category"] in category_filter]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    paths = [r["path"] for r in shown]
    if paths:
        selected = st.selectbox("Inspect text diff", paths, key="diff_path")
        fc = next((x for x in result.diff.file_changes if safe_text(x.path) == selected), None)
        if fc:
            if fc.is_binary:
                st.info("Binary content is not displayed.")
            elif fc.patch_text:
                patch_lines = fc.patch_text.splitlines()[:MAX_DIFF_LINES]
                st.code("\n".join(patch_lines), language="diff", line_numbers=False)
                if len(fc.patch_text.splitlines()) > MAX_DIFF_LINES:
                    st.caption(f"Diff view capped at {MAX_DIFF_LINES} lines.")
            else:
                st.info("No textual patch is available for this file.")


def _dependency_tab(result: AnalysisResult):
    st.caption("Edge convention: A → B means A imports B.")
    st.caption("Static analysis cannot see dynamic imports, reflection, framework wiring, runtime plugin loading, or non-Python code.")
    cols = st.columns(4)
    impact_filter = cols[0].multiselect("Impact category", ["changed", "direct", "indirect", "unrelated"],
                                        default=["changed", "direct", "indirect", "unrelated"], key="graph_impact_filter")
    file_cats = sorted({result.graph.graph.nodes[n].get("category", "other") for n in result.graph.graph.nodes})
    category_filter = cols[1].multiselect("File category", file_cats, default=file_cats, key="graph_file_filter")
    max_distance = cols[2].slider("Maximum dependent distance", 1, 10, min(4, 10), key="graph_max_distance")
    full_allowed = result.graph.total_nodes <= 250
    full = cols[3].checkbox("Show full graph", value=False, disabled=not full_allowed, key="graph_full")
    if not full_allowed:
        st.caption("Full graph is disabled above 250 nodes; showing the changed-file neighborhood with a 180-node cap.")
    _plot_dependency_graph(result, max_distance, impact_filter, category_filter, full)
    options = sorted(result.graph.graph.nodes)
    if options:
        selected = st.selectbox("Inspect module", options, format_func=lambda p: safe_text(p), key="module_picker")
        graph = result.graph.graph
        ev = result.impact.evidence_records.get(selected)
        left, right = st.columns(2)
        left.markdown("**Imports (dependencies)**")
        left.write([safe_text(p) for p in sorted(graph.successors(selected))] or ["None detected"])
        right.markdown("**Imported by (dependents)**")
        right.write([safe_text(p) for p in sorted(graph.predecessors(selected))] or ["None detected"])
        if ev:
            st.write("Shortest path: " + safe_text(ev.shortest_path_str))
    unresolved = result.graph.unresolved_imports
    parse_failures = result.graph.parse_failures
    st.markdown("#### Unresolved imports and parse failures")
    if unresolved:
        st.dataframe([{"file": safe_text(path), "imports": safe_text(", ".join(items))}
                      for path, items in sorted(unresolved.items())], use_container_width=True, hide_index=True)
    else:
        st.caption("No unresolved internal imports reported.")
    if parse_failures:
        st.dataframe([{"file": safe_text(path), "parse error": safe_text(error)}
                      for path, error in sorted(parse_failures.items())], use_container_width=True, hide_index=True)
    else:
        st.caption("No Python parse failures reported.")


def _risk_tab(result: AnalysisResult):
    prediction = result.prediction
    st.markdown("#### Commit risk")
    if prediction.available:
        st.metric("Risk tier", prediction.tier)
        st.metric("Risk score", f"{prediction.score:.4f}")
        st.caption(f"Decision threshold: {prediction.threshold_used}; model version: {safe_text(prediction.model_version)}")
    else:
        st.warning(f"Risk score unavailable: {safe_text(prediction.error or 'The model could not produce a score.')}")
    metrics = _metrics_data()
    if metrics:
        st.markdown("#### Held-out model evaluation")
        dataset = metrics.get("dataset", {})
        st.caption(f"Dataset: {safe_text(dataset.get('name', 'Unknown'))} · split: {safe_text(dataset.get('split_type', 'not recorded'))} · test rows: {dataset.get('test_rows', 'not recorded')}")
        primary = metrics.get("evaluation", {}).get("primary_model_hgb_calibrated", {})
        metric_names = [("PR-AUC", "pr_auc"), ("ROC-AUC", "roc_auc"), ("Precision", "precision"),
                        ("Recall", "recall"), ("F1", "f1"), ("Brier score", "brier_score")]
        columns = st.columns(len(metric_names))
        for col, (label, key) in zip(columns, metric_names):
            if key in primary:
                col.metric(label, primary[key])
    else:
        st.info("Evaluation metrics are unavailable; no evaluation numbers are shown.")
    if result.features and result.features.missing_or_degraded:
        st.warning("Degraded features: " + ", ".join(safe_text(x) for x in result.features.missing_or_degraded))
        for note in result.features.notes:
            st.caption(safe_text(note))
    if result.explanation.available:
        local = sorted(result.explanation.features, key=lambda x: abs(x.contribution), reverse=True)
        st.markdown("#### Local explanation")
        st.caption(f"Method: {safe_text(result.explanation.local_method)}. Contributions describe model behavior, not bug causality.")
        fig = go.Figure(go.Bar(x=[x.contribution for x in local], y=[safe_text(x.feature) for x in local],
                               orientation="h", marker_color=["#f97583" if x.contribution > 0 else "#79c0ff" for x in local],
                               customdata=[[x.value, safe_text(x.text)] for x in local],
                               hovertemplate="%{y}<br>Contribution: %{x:.4f}<br>Value: %{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        fig.update_layout(template="plotly_dark", height=380, margin=dict(l=10, r=10, t=15, b=10),
                          paper_bgcolor="#111820", plot_bgcolor="#111820", xaxis_title="Signed contribution (raises / lowers model output)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(safe_text(result.explanation.error or "Local explanation unavailable."))
    if result.explanation.global_importance:
        st.markdown("#### Global feature importance")
        imp = sorted(result.explanation.global_importance.items(), key=lambda x: x[1], reverse=True)
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[safe_text(k) for k, _ in imp], orientation="h", marker_color="#58a6ff"))
        fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=15, b=10),
                          paper_bgcolor="#111820", plot_bgcolor="#111820", xaxis_title=safe_text(result.explanation.global_method))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Global feature importance is unavailable.")
    st.warning("Domain shift: the model was trained on ApacheJIT Apache-project data; results may not transfer to other repositories.")
    st.warning("Uncertainty: graph coverage is static and feature history may be degraded; interpret scores as advisory.")
    for disclaimer in result.explanation.disclaimers:
        st.caption(safe_text(disclaimer))


def _priority_tab(result: AnalysisResult):
    st.markdown("The table combines static graph evidence and an optional shared commit-level risk score. That shared score shifts scores and tier boundaries; it does not distinguish files. These are heuristic priorities, not failure probabilities.")
    st.caption("Effective weights: " + ", ".join(f"{k} {v:.2f}" for k, v in result.ranking.effective_weights.items()))
    rows = [{"rank": f.rank, "path": safe_text(f.path), "tier": f.priority_tier,
             "heuristic priority score": round(f.heuristic_priority_score, 4),
             "evidence": f.evidence_level, "distance": f.distance,
             "suggested action": f.suggested_action}
            for f in result.ranking.files]
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        with st.expander("Per-file evidence"):
            for file in result.ranking.files:
                with st.expander(f"{file.rank}. {safe_text(file.path)} · {file.priority_tier}"):
                    st.write("Shortest dependency path: " + safe_text(" → ".join(file.shortest_dependency_path) or "No path found"))
                    st.write("Component breakdown")
                    st.json(file.component_breakdown)
                    st.write(safe_text(file.reason))
    else:
        st.info("No files are available to rank for this commit.")
    if result.ranking.removed:
        st.markdown("#### Removed files")
        st.dataframe([{"path": safe_text(f.path), "category": f.category, "status": f.status, "note": f.note}
                      for f in result.ranking.removed], use_container_width=True, hide_index=True)
    with st.expander("Compare strategies"):
        st.caption(result.ranking.validation_note)
        for name, ranks in result.ranking.baselines.items():
            st.markdown(f"**{safe_text(name.replace('_', ' ').title())}**")
            st.dataframe([{"rank": rank, "path": safe_text(path)} for path, rank in sorted(ranks.items(), key=lambda kv: kv[1])],
                         use_container_width=True, hide_index=True)


def _report_tab(result: AnalysisResult):
    st.markdown("Preview only — report downloads will be added in a later release.")
    preview = {
        "repository": safe_text(result.repository_identity),
        "commit": result.commit_sha,
        "message": safe_text(result.commit_message),
        "files_changed": result.diff.files_changed_count,
        "risk_available": result.prediction.available,
        "risk_score": result.prediction.score,
        "risk_tier": result.prediction.tier,
        "ranked_files": [{"rank": x.rank, "path": safe_text(x.path), "tier": x.priority_tier,
                          "heuristic_priority_score": x.heuristic_priority_score,
                          "evidence": x.evidence_level, "reason": safe_text(x.reason)}
                         for x in result.ranking.files[:50]],
        "removed_files": [safe_text(x.path) for x in result.ranking.removed],
        "limitations": LIMITATIONS,
    }
    st.json(preview)


st.title("ImpactMap")
st.caption("Commit impact analysis for repository review and test planning")
st.info(LIMITATIONS, icon="ℹ️")

with st.sidebar:
    st.header("Repository")
    st.text_input("Local repository path", key="local_repo_path", placeholder="/path/to/git/repository")
    st.text_input("Public HTTPS URL", key="remote_repo_url", placeholder="https://github.com/owner/repository")
    st.button("Load repository", key="load_repo", on_click=_load_repo, use_container_width=True)
    if st.session_state.get("repo_load_error"):
        st.error(safe_text(st.session_state.repo_load_error))
    if st.session_state.get("repo_loaded"):
        st.success(f"Loaded {safe_text(st.session_state.repo_label)}")
    st.divider()
    st.header("Analysis options")
    if st.button("Reset ranking weights", key="reset_weights", use_container_width=True):
        for name, value in DEFAULT_RANKING_WEIGHTS.items():
            st.session_state[f"weight_{name}"] = value
        st.rerun()
    for name, value in DEFAULT_RANKING_WEIGHTS.items():
        st.slider(name.replace("_", " ").title(), 0.0, 1.0, float(value), 0.05,
                  key=f"weight_{name}")
    st.slider("Graph traversal depth", 1, 25, 10, key="max_depth")
    st.slider("Maximum indexed files", 10, 1000, MAX_FILES, 10, key="max_files")
    st.text_area("Excluded directories (one per line)", value="\n".join(sorted(EXCLUDED_DIRS)),
                 key="excluded_dirs_text", height=110)

repo = None
commit_rows = []
selected_sha = None
if st.session_state.get("repo_loaded") and st.session_state.get("repo_path"):
    try:
        repo = _open_local_repo(st.session_state.repo_path)
        st.sidebar.markdown("#### Select commit")
        search = st.sidebar.text_input("Search hash, message, or author", key="commit_search")
        commit_rows = _commit_rows(st.session_state.repo_path, st.session_state.repo_head, search)
        if commit_rows:
            option_by_label = {f"{c['short_sha']} · {safe_text(c['message'])} · {safe_text(c['author_name'])}": c["hexsha"] for c in commit_rows}
            labels = list(option_by_label)
            selected_label = st.sidebar.selectbox("Commit", labels, key="selected_commit_label")
            selected_sha = option_by_label[selected_label]
            st.sidebar.button("Analyze selected commit", key="analyze_commit", type="primary",
                              on_click=lambda: _run_selected(repo, selected_sha, _config_from_controls()),
                              use_container_width=True)
        else:
            st.sidebar.info("No commits match this search.")
    except Exception as exc:
        st.error(f"Repository/commit loading failed: {safe_text(exc)}")


def _run_selected(repo_arg, sha_arg, config_arg):
    try:
        with st.spinner("Analyzing commit, dependencies, features, and risk..."):
            st.session_state.analysis_result = run_analysis(repo_arg, sha_arg, config_arg)
            st.session_state.analysis_error = None
    except Exception as exc:
        st.session_state.analysis_result = None
        st.session_state.analysis_error = str(exc)


result = st.session_state.get("analysis_result")
if result is not None and selected_sha and repo is not None:
    current_config_hash = _config_from_controls().config_hash()
    current_repo_identity = str(Path(repo.git_dir).resolve())
    if (result.commit_sha != selected_sha or result.cache_key[0] != current_repo_identity or
            result.cache_key[2] != current_config_hash):
        result = None
        st.session_state.pop("analysis_result", None)
        st.info("The selected commit or analysis options changed. Run the analysis again to refresh these results.")
if st.session_state.get("analysis_error"):
    st.error(f"Analysis failed: {safe_text(st.session_state.analysis_error)}. Check the repository path, commit, and available history, then retry.")
if result is None:
    st.subheader("Ready to analyze")
    st.write("Load a local Git repository or a public HTTPS repository, choose a commit, and run the analysis.")
    st.info("The repository is read from Git objects. ImpactMap does not execute analyzed repository code.")
else:
    if result.prediction.available:
        st.caption("Commit risk score is model output. File ranking below is a separate heuristic priority score.")
    else:
        st.caption("Graph results are available. The ML risk section is marked unavailable and its ranking weight was omitted.")
    tabs = st.tabs(["Overview", "Commit Explorer", "Dependency Explorer", "Risk Analysis", "Testing Priority", "Report Export"])
    with tabs[0]:
        _overview_tab(result)
    with tabs[1]:
        _commit_tab(result, repo)
    with tabs[2]:
        _dependency_tab(result)
    with tabs[3]:
        _risk_tab(result)
    with tabs[4]:
        _priority_tab(result)
    with tabs[5]:
        _report_tab(result)
