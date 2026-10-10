"""Impact analysis: seed mapping, reverse BFS dependent traversal, shortest dependency chains, and summary metrics."""

from collections import deque
from dataclasses import dataclass, field
import itertools
import logging
from typing import Dict, List, Optional, Set, Tuple

import git
import networkx as nx

from impactmap.config import (
    AppConfig,
    MAX_DEPTH,
    MAX_PATHS_PER_NODE,
)
from impactmap.dependency_graph import (
    GraphBuildResult,
    build_dependency_graph,
)
from impactmap.diff_analyzer import CommitDiffSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImpactEvidence:
    """Graph evidence for a single repository file relative to the changed commit."""
    path: str
    category: str  # "changed", "direct", "indirect", "unrelated"
    distance: int   # 0 for changed, 1 for direct, 2+ for indirect, -1 for unrelated
    shortest_path: List[str]  # e.g. ['src/a.py', 'src/b.py', 'src/seed.py']
    shortest_path_str: str     # e.g. "src/a.py → src/b.py → src/seed.py"
    reachable_seeds: List[str]
    path_count: int
    is_broken_import: bool = False
    broken_import_details: Optional[str] = None
    is_paths_capped: bool = False


@dataclass
class ImpactAnalysisResult:
    """Complete results of static impact analysis across the repository graph."""
    commit_sha: str
    total_nodes: int
    total_edges: int
    changed_files_count: int
    direct_dependents_count: int
    indirect_dependents_count: int
    unrelated_count: int
    max_distance: int
    graph_density: float
    evidence_records: Dict[str, ImpactEvidence] = field(default_factory=dict)
    dependencies_of_changed: Dict[str, List[str]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


def _find_capped_paths(
    g: nx.DiGraph,
    source: str,
    target: str,
    cutoff: int,
    max_paths: int,
) -> Tuple[int, bool]:
    """Counts simple paths from source to target up to cutoff depth, capped at max_paths."""
    count = 0
    is_capped = False
    try:
        for _ in nx.all_simple_paths(g, source=source, target=target, cutoff=cutoff):
            count += 1
            if count >= max_paths:
                is_capped = True
                break
    except Exception:
        pass
    return count, is_capped


def analyze_impact(
    repo: git.Repo,
    diff_summary: CommitDiffSummary,
    config: Optional[AppConfig] = None,
) -> ImpactAnalysisResult:
    """Deterministic, graph-based impact analysis.
    
    Identifies:
    - Seeds: changed Python files (added, modified, renamed).
    - Direct dependents: files that directly import any seed (distance 1).
    - Indirect dependents: files that transitively import any seed (distance 2..MAX_DEPTH).
    - Dependents of deleted or renamed modules via parent commit's dependency graph.
    - Dependencies of changed files (context only, not counted as impacted).
    """
    cfg = config or AppConfig()
    warnings: List[str] = []

    # 1. Build current commit graph
    current_build = build_dependency_graph(repo, diff_summary.commit_sha, config=cfg)
    g_curr = current_build.graph

    if current_build.parse_failures:
        warnings.append(
            f"{len(current_build.parse_failures)} Python file(s) failed AST parsing and were indexed as isolated nodes."
        )
    if current_build.unresolved_imports:
        unresolved_count = sum(len(v) for v in current_build.unresolved_imports.values())
        warnings.append(
            f"{unresolved_count} import statement(s) looked internal but could not be resolved to files."
        )
    if current_build.truncated:
        warnings.append(f"Analysis was truncated at maximum file limit ({cfg.max_files} files).")

    # 2. Identify seeds and rename mappings from diff
    changed_python_files: Set[str] = set()
    deleted_python_files: Set[str] = set()
    renamed_python_map: Dict[str, str] = {}  # old_path -> new_path

    for fc in diff_summary.file_changes:
        if fc.category in ("python_source", "test"):
            if fc.status == "deleted":
                deleted_python_files.add(fc.path)
            elif fc.status == "renamed" and fc.old_path:
                renamed_python_map[fc.old_path] = fc.path
                changed_python_files.add(fc.path)
            else:
                changed_python_files.add(fc.path)

    # 3. Handle deleted and renamed files via parent commit graph
    broken_import_dependents: Dict[str, str] = {}  # dependent_path -> reason
    if (deleted_python_files or renamed_python_map) and diff_summary.parent_sha and not diff_summary.is_initial_commit:
        try:
            parent_build = build_dependency_graph(repo, diff_summary.parent_sha, config=cfg)
            g_parent = parent_build.graph

            # Inspect dependents of deleted files in parent graph
            for del_file in deleted_python_files:
                if del_file in g_parent:
                    # In parent graph, edge X -> del_file means X imported del_file
                    for pred in g_parent.predecessors(del_file):
                        # Translate if pred itself was renamed
                        curr_pred = renamed_python_map.get(pred, pred)
                        if curr_pred in g_curr:
                            broken_import_dependents[curr_pred] = (
                                f"Imports removed module '{del_file}'"
                            )

            # Inspect dependents of renamed files in parent graph
            for old_path, new_path in renamed_python_map.items():
                if old_path in g_parent:
                    for pred in g_parent.predecessors(old_path):
                        curr_pred = renamed_python_map.get(pred, pred)
                        if curr_pred in g_curr and not g_curr.has_edge(curr_pred, new_path):
                            # The file hasn't updated import to new_path
                            broken_import_dependents[curr_pred] = (
                                f"Imports un-updated module path '{old_path}' (renamed to '{new_path}')"
                            )
        except Exception as e:
            logger.warning("Could not build parent graph for deleted file analysis: %s", e)

    # 4. Context: dependencies of changed files (what changed files import)
    dependencies_of_changed: Dict[str, List[str]] = {}
    for seed in changed_python_files:
        if seed in g_curr:
            # S -> Target means S imports Target
            dependencies_of_changed[seed] = sorted(list(g_curr.successors(seed)))

    # 5. BFS reverse-traversal to find dependents of seeds in current graph
    # G_rev: edge Target -> S means S imports Target (i.e. S is dependent of Target)
    g_rev = g_curr.reverse(copy=False)

    # For each node in g_curr, compute distance, shortest path, and reachable seeds
    # node -> (min_dist, shortest_path_from_node_to_seed, list_of_seeds, total_paths, is_capped)
    traversal_info: Dict[str, Tuple[int, List[str], List[str], int, bool]] = {}

    for node in g_curr.nodes:
        if node in changed_python_files:
            continue

        reachable_seeds: List[str] = []
        best_dist = 999999
        best_path: List[str] = []
        total_path_count = 0
        any_capped = False

        for seed in changed_python_files:
            if seed not in g_curr:
                continue

            # In g_curr, path goes from node -> ... -> seed
            try:
                if nx.has_path(g_curr, source=node, target=seed):
                    shortest_p = nx.shortest_path(g_curr, source=node, target=seed)
                    dist = len(shortest_p) - 1
                    if dist <= cfg.max_depth:
                        reachable_seeds.append(seed)
                        if dist < best_dist:
                            best_dist = dist
                            best_path = shortest_p

                        paths_cnt, is_cap = _find_capped_paths(
                            g_curr, source=node, target=seed, cutoff=cfg.max_depth, max_paths=cfg.max_paths_per_node
                        )
                        total_path_count += paths_cnt
                        if is_cap:
                            any_capped = True
            except Exception:
                pass

        if reachable_seeds:
            traversal_info[node] = (best_dist, best_path, reachable_seeds, total_path_count, any_capped)

    # 6. Build evidence records for every node in g_curr
    evidence_records: Dict[str, ImpactEvidence] = {}
    direct_count = 0
    indirect_count = 0
    unrelated_count = 0
    max_distance_seen = 0

    for node in g_curr.nodes:
        if node in changed_python_files:
            category = "changed"
            dist = 0
            shortest_p = [node]
            path_str = f"{node} (Directly changed)"
            seeds = [node]
            path_cnt = 1
            is_cap = False
            is_broken = False
            broken_details = None

        elif node in broken_import_dependents:
            # High-evidence direct dependent via deleted/renamed import
            category = "direct"
            dist = 1
            shortest_p = [node]
            broken_details = broken_import_dependents[node]
            path_str = f"{node} → [Deleted/Renamed Module]"
            seeds = []
            path_cnt = 1
            is_cap = False
            is_broken = True
            direct_count += 1
            max_distance_seen = max(max_distance_seen, 1)

        elif node in traversal_info:
            dist, shortest_p, seeds, path_cnt, is_cap = traversal_info[node]
            category = "direct" if dist == 1 else "indirect"
            path_str = " → ".join(shortest_p)
            is_broken = False
            broken_details = None

            if dist == 1:
                direct_count += 1
            else:
                indirect_count += 1
            max_distance_seen = max(max_distance_seen, dist)

        else:
            category = "unrelated"
            dist = -1
            shortest_p = []
            path_str = "None (No dependency path to changed files)"
            seeds = []
            path_cnt = 0
            is_cap = False
            is_broken = False
            broken_details = None
            unrelated_count += 1

        evidence_records[node] = ImpactEvidence(
            path=node,
            category=category,
            distance=dist,
            shortest_path=shortest_p,
            shortest_path_str=path_str,
            reachable_seeds=seeds,
            path_count=path_cnt,
            is_broken_import=is_broken,
            broken_import_details=broken_details,
            is_paths_capped=is_cap,
        )

    total_nodes = g_curr.number_of_nodes()
    total_edges = g_curr.number_of_edges()
    density = nx.density(g_curr) if total_nodes > 1 else 0.0

    return ImpactAnalysisResult(
        commit_sha=diff_summary.commit_sha,
        total_nodes=total_nodes,
        total_edges=total_edges,
        changed_files_count=len(changed_python_files),
        direct_dependents_count=direct_count,
        indirect_dependents_count=indirect_count,
        unrelated_count=unrelated_count,
        max_distance=max_distance_seen,
        graph_density=density,
        evidence_records=evidence_records,
        dependencies_of_changed=dependencies_of_changed,
        warnings=warnings,
    )
