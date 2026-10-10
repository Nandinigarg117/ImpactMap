"""Static dependency mapping: AST import extraction, module resolution, and NetworkX dependency graph."""

import ast
from dataclasses import dataclass, field
import hashlib
import logging
from pathlib import Path
import sys
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import git
import networkx as nx

from impactmap.config import (
    AppConfig,
    EXCLUDED_DIRS,
    MAX_FILE_BYTES,
    MAX_FILES,
)
from impactmap.repository import list_python_files_in_commit, read_file_blob

logger = logging.getLogger(__name__)

# Standard library module names in Python 3.10+
if hasattr(sys, "stdlib_module_names"):
    STDLIB_MODULES: FrozenSet[str] = frozenset(sys.stdlib_module_names)
else:
    STDLIB_MODULES: FrozenSet[str] = frozenset({
        "abc", "argparse", "ast", "asyncio", "base64", "collections", "contextlib",
        "copy", "csv", "dataclasses", "datetime", "decimal", "difflib", "dis",
        "enum", "functools", "glob", "hashlib", "heapq", "http", "importlib",
        "inspect", "io", "itertools", "json", "logging", "math", "multiprocessing",
        "operator", "os", "pathlib", "pickle", "platform", "queue", "random",
        "re", "shutil", "signal", "socket", "sqlite3", "ssl", "stat", "string",
        "subprocess", "sys", "tempfile", "threading", "time", "traceback",
        "types", "typing", "unittest", "urllib", "uuid", "warnings", "weakref", "zipfile",
    })


@dataclass(frozen=True)
class ImportRecord:
    """Detailed record of a single import extracted via AST."""
    source_file: str
    target_module: str
    imported_name: Optional[str]
    line_number: int
    is_conditional: bool
    is_relative: bool
    level: int
    raw_statement: str


@dataclass
class GraphBuildResult:
    """Outcome of building the dependency graph at a given commit."""
    graph: nx.DiGraph
    commit_sha: str
    total_nodes: int
    total_edges: int
    parse_failures: Dict[str, str] = field(default_factory=dict)
    external_imports: Dict[str, int] = field(default_factory=dict)
    unresolved_imports: Dict[str, List[str]] = field(default_factory=dict)
    truncated: bool = False
    cache_hit: bool = False


class _ImportExtractor(ast.NodeVisitor):
    """AST visitor that detects Import and ImportFrom statements and conditional scopes."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.imports: List[ImportRecord] = []
        self._conditional_depth = 0

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_Try(self, node: ast.Try):
        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_If(self, node: ast.If):
        # Detect if TYPE_CHECKING or conditional
        is_type_checking = False
        if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            is_type_checking = True
        elif isinstance(node.test, ast.Attribute) and node.test.attr == "TYPE_CHECKING":
            is_type_checking = True

        self._conditional_depth += 1
        self.generic_visit(node)
        self._conditional_depth -= 1

    def visit_Import(self, node: ast.Import):
        is_cond = self._conditional_depth > 0
        for alias in node.names:
            stmt = f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else "")
            self.imports.append(
                ImportRecord(
                    source_file=self.source_file,
                    target_module=alias.name,
                    imported_name=None,
                    line_number=node.lineno,
                    is_conditional=is_cond,
                    is_relative=False,
                    level=0,
                    raw_statement=stmt,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        is_cond = self._conditional_depth > 0
        mod_name = node.module or ""
        level = node.level or 0
        is_rel = level > 0

        for alias in node.names:
            prefix = "." * level
            stmt = f"from {prefix}{mod_name} import {alias.name}"
            self.imports.append(
                ImportRecord(
                    source_file=self.source_file,
                    target_module=mod_name,
                    imported_name=alias.name,
                    line_number=node.lineno,
                    is_conditional=is_cond,
                    is_relative=is_rel,
                    level=level,
                    raw_statement=stmt,
                )
            )


def detect_source_roots(known_files: Set[str]) -> List[str]:
    """Detects likely Python source roots (e.g. '', 'src', top-level packages)."""
    roots = [""]
    # Check for src root
    if any(p.startswith("src/") for p in known_files):
        roots.append("src")

    # Check top-level directories that contain __init__.py or multiple python files
    top_dirs: Set[str] = set()
    for p in known_files:
        parts = p.split("/")
        if len(parts) > 1 and parts[0] not in {"tests", "test", "docs", "scripts"}:
            top_dirs.add(parts[0])

    for td in sorted(top_dirs):
        if td not in roots:
            roots.append(td)

    return roots


def _file_to_module_name(file_path: str, source_roots: Sequence[str]) -> str:
    """Converts a repo-relative path to its dotted module name using the deepest matching source root."""
    posix_path = file_path.replace("\\", "/").strip("/")
    path_without_ext = posix_path.rsplit(".", 1)[0]

    # Try matching source roots, longest first
    sorted_roots = sorted(source_roots, key=lambda r: len(r), reverse=True)
    for root in sorted_roots:
        if root and path_without_ext.startswith(f"{root}/"):
            rel = path_without_ext[len(root) + 1:]
            if rel.endswith("/__init__"):
                rel = rel[:-9]
            return rel.replace("/", ".")

    if path_without_ext.endswith("/__init__"):
        path_without_ext = path_without_ext[:-9]

    return path_without_ext.replace("/", ".")


def resolve_import(
    imp: ImportRecord,
    source_file: str,
    known_files: Set[str],
    source_roots: Sequence[str],
) -> Tuple[Optional[List[str]], str]:
    """Resolves an ImportRecord to internal target files, 'external', or 'unresolved'.
    
    Returns:
    - (list_of_resolved_files, 'internal') if resolved to one or more internal files.
    - (None, 'external') if standard library or recognized third-party.
    - (None, 'unresolved') if looked internal but could not be resolved.
    """
    # 1. Relative import resolution
    if imp.is_relative:
        parts = source_file.split("/")
        pkg_parts = parts[:-1]  # directory containing source_file

        # Each level beyond 1 goes up one directory
        up_steps = imp.level - 1
        if up_steps > len(pkg_parts):
            return None, "unresolved"

        base_parts = pkg_parts[: len(pkg_parts) - up_steps] if up_steps > 0 else pkg_parts
        base_dir = "/".join(base_parts)

        candidates: List[str] = []
        if imp.target_module:
            rel_path_mod = imp.target_module.replace(".", "/")
            prefix = f"{base_dir}/{rel_path_mod}".strip("/")
        else:
            prefix = base_dir.strip("/")

        # Check submodule or imported symbol as submodule
        if imp.imported_name and imp.imported_name != "*":
            cand_sub = f"{prefix}/{imp.imported_name}.py".strip("/")
            if cand_sub in known_files:
                candidates.append(cand_sub)
            cand_sub_package = f"{prefix}/{imp.imported_name}/__init__.py".strip("/")
            if cand_sub_package in known_files and cand_sub_package not in candidates:
                candidates.append(cand_sub_package)

        # Check module as file
        cand_file = f"{prefix}.py".strip("/")
        if cand_file in known_files and cand_file not in candidates:
            candidates.append(cand_file)

        # Check module as package __init__.py
        cand_pkg = f"{prefix}/__init__.py".strip("/")
        if cand_pkg in known_files and cand_pkg not in candidates:
            candidates.append(cand_pkg)

        if candidates:
            return candidates, "internal"
        return None, "unresolved"

    # 2. Check stdlib for absolute import
    top_level_name = imp.target_module.split(".")[0]
    if top_level_name in STDLIB_MODULES:
        return None, "external"

    # 3. Absolute import resolution against detected source roots
    mod_path = imp.target_module.replace(".", "/")
    resolved_candidates: List[str] = []

    for root in source_roots:
        root_prefix = f"{root}/" if root else ""

        # If from pkg import submodule, check submodule first
        if imp.imported_name and imp.imported_name != "*":
            sub_path = f"{root_prefix}{mod_path}/{imp.imported_name}.py"
            if sub_path in known_files and sub_path not in resolved_candidates:
                resolved_candidates.append(sub_path)
            sub_package_path = f"{root_prefix}{mod_path}/{imp.imported_name}/__init__.py"
            if sub_package_path in known_files and sub_package_path not in resolved_candidates:
                resolved_candidates.append(sub_package_path)

        # Check direct module .py
        direct_file = f"{root_prefix}{mod_path}.py"
        if direct_file in known_files and direct_file not in resolved_candidates:
            resolved_candidates.append(direct_file)

        # Check package __init__.py
        pkg_init = f"{root_prefix}{mod_path}/__init__.py"
        if pkg_init in known_files and pkg_init not in resolved_candidates:
            resolved_candidates.append(pkg_init)

    if resolved_candidates:
        return resolved_candidates, "internal"

    # 4. If target_module starts with a known internal package/module name -> unresolved
    internal_top_level_names: Set[str] = set()
    for root in source_roots:
        prefix = f"{root}/" if root else ""
        for f in known_files:
            rel = f[len(prefix):] if (prefix and f.startswith(prefix)) else f
            top_part = rel.split("/")[0].replace(".py", "")
            internal_top_level_names.add(top_part)

    if top_level_name in internal_top_level_names:
        return None, "unresolved"

    return None, "external"


# In-memory dependency graph cache: (repo_path, commit_sha, config_hash) -> GraphBuildResult
_GRAPH_CACHE: Dict[Tuple[str, str, str], GraphBuildResult] = {}


def build_dependency_graph(
    repo: git.Repo,
    commit_sha: str,
    config: Optional[AppConfig] = None,
    use_cache: bool = True,
) -> GraphBuildResult:
    """Builds the static Python dependency graph at commit_sha directly from git blobs.
    
    Graph definition:
    - Nodes: repo-relative POSIX file paths.
    - Directed Edge A -> B means A imports (depends on) B.
    - Impact propagates backwards along reversed edges (predecessors of changed files).
    """
    cfg = config or AppConfig()
    repo_key = str(repo.working_dir or repo.git_dir)
    cache_key = (repo_key, commit_sha, cfg.config_hash())

    if use_cache and cache_key in _GRAPH_CACHE:
        cached = _GRAPH_CACHE[cache_key]
        return GraphBuildResult(
            graph=cached.graph,
            commit_sha=cached.commit_sha,
            total_nodes=cached.total_nodes,
            total_edges=cached.total_edges,
            parse_failures=dict(cached.parse_failures),
            external_imports=dict(cached.external_imports),
            unresolved_imports=dict(cached.unresolved_imports),
            truncated=cached.truncated,
            cache_hit=True,
        )

    try:
        commit = repo.commit(commit_sha)
    except Exception as e:
        raise ValueError(f"Commit '{commit_sha}' not found: {e}") from e

    py_files, is_truncated = list_python_files_in_commit(
        commit,
        excluded_dirs=cfg.excluded_dirs,
        max_files=cfg.max_files,
    )

    known_files = set(py_files)
    source_roots = detect_source_roots(known_files)

    g = nx.DiGraph()
    parse_failures: Dict[str, str] = {}
    external_imports: Dict[str, int] = {}
    unresolved_imports: Dict[str, List[str]] = {}

    # Initialize all nodes
    for f in py_files:
        category = "test" if any(ind in f for ind in ("/tests/", "/test/", "test_", "_test.py")) else "python_source"
        mod_name = _file_to_module_name(f, source_roots)
        g.add_node(
            f,
            module_name=mod_name,
            category=category,
            parse_failed=False,
            parse_error="",
            line_count=0,
        )

    # Parse and extract imports from blobs
    for f in py_files:
        blob = read_file_blob(commit, f, max_bytes=cfg.max_file_bytes)
        if blob is None:
            continue

        content = blob.content_text
        g.nodes[f]["line_count"] = len(content.splitlines())

        try:
            tree = ast.parse(content, filename=f)
        except (SyntaxError, UnicodeDecodeError, ValueError) as err:
            err_msg = str(err)
            parse_failures[f] = err_msg
            g.nodes[f]["parse_failed"] = True
            g.nodes[f]["parse_error"] = err_msg
            logger.debug("AST parse failed for %s: %s", f, err_msg)
            continue

        visitor = _ImportExtractor(source_file=f)
        visitor.visit(tree)

        for imp in visitor.imports:
            resolved_targets, classification = resolve_import(
                imp=imp,
                source_file=f,
                known_files=known_files,
                source_roots=source_roots,
            )

            if classification == "internal" and resolved_targets:
                for target_file in resolved_targets:
                    # Ignore self-import
                    if target_file == f:
                        continue

                    # Ensure target exists in graph
                    if target_file not in g:
                        category = "test" if "test" in target_file else "python_source"
                        g.add_node(
                            target_file,
                            module_name=_file_to_module_name(target_file, source_roots),
                            category=category,
                            parse_failed=False,
                            parse_error="",
                            line_count=0,
                        )

                    if g.has_edge(f, target_file):
                        edge_data = g[f][target_file]
                        if imp.line_number not in edge_data["import_lines"]:
                            edge_data["import_lines"].append(imp.line_number)
                            edge_data["import_statements"].append(imp.raw_statement)
                    else:
                        kind = "relative" if imp.is_relative else ("from_import" if imp.imported_name else "direct")
                        g.add_edge(
                            f,
                            target_file,
                            import_lines=[imp.line_number],
                            import_statements=[imp.raw_statement],
                            kind=kind,
                            is_conditional=imp.is_conditional,
                        )

            elif classification == "external":
                top_pkg = (imp.target_module or "").split(".")[0] or "unknown"
                external_imports[top_pkg] = external_imports.get(top_pkg, 0) + 1

            elif classification == "unresolved":
                unresolved_name = imp.target_module + (f".{imp.imported_name}" if imp.imported_name else "")
                unresolved_imports.setdefault(f, []).append(unresolved_name)

    result = GraphBuildResult(
        graph=g,
        commit_sha=commit.hexsha,
        total_nodes=g.number_of_nodes(),
        total_edges=g.number_of_edges(),
        parse_failures=parse_failures,
        external_imports=external_imports,
        unresolved_imports=unresolved_imports,
        truncated=is_truncated,
        cache_hit=False,
    )

    if use_cache:
        _GRAPH_CACHE[cache_key] = result

    return result
