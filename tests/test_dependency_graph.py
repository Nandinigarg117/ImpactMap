"""Unit tests for AST dependency graph builder, resolution rules, and edge cases."""

from pathlib import Path
import pytest
import git
import networkx as nx

from impactmap.config import AppConfig
from impactmap.dependency_graph import (
    build_dependency_graph,
    detect_source_roots,
    resolve_import,
)
from impactmap.repository import validate_and_open_local_repo


@pytest.fixture
def complex_repo(tmp_path: Path) -> git.Repo:
    """Creates a git repository with rich Python import scenarios:
    - Multi-level chain: a -> b -> c
    - Circular import: circ1 <-> circ2
    - Relative imports: . and ..
    - Syntax error: broken.py
    - Excluded dir: venv/ignored.py
    - External & Unresolved imports
    """
    repo_dir = tmp_path / "dep_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(repo_dir, b="main")

    src_pkg = repo_dir / "src" / "pkg"
    src_pkg.mkdir(parents=True, exist_ok=True)
    sub_pkg = src_pkg / "sub"
    sub_pkg.mkdir(parents=True, exist_ok=True)
    venv_dir = repo_dir / "venv"
    venv_dir.mkdir(parents=True, exist_ok=True)

    # 1. Package inits
    (src_pkg / "__init__.py").write_text("# pkg init\nfrom . import c\n", encoding="utf-8")
    (sub_pkg / "__init__.py").write_text("# sub init\n", encoding="utf-8")

    # 2. Leaf module c
    (src_pkg / "c.py").write_text("C_VALUE = 42\n", encoding="utf-8")

    # 3. Middle module b (imports c via relative import, plus external stdlib os)
    (src_pkg / "b.py").write_text(
        "import os\nfrom .c import C_VALUE\n\ndef get_b():\n    return C_VALUE + 1\n",
        encoding="utf-8",
    )

    # 4. Top module a (imports b via absolute import and external json)
    (src_pkg / "a.py").write_text(
        "import json\nimport pkg.b\nfrom pkg.b import get_b\n\ndef run():\n    return get_b()\n",
        encoding="utf-8",
    )

    # 5. Deep relative import in sub/nested.py (imports ..c)
    (sub_pkg / "nested.py").write_text(
        "from ..c import C_VALUE\nfrom .. import b\n",
        encoding="utf-8",
    )

    # 6. Circular import modules
    (src_pkg / "circ1.py").write_text("from . import circ2\nVAL1 = 1\n", encoding="utf-8")
    (src_pkg / "circ2.py").write_text("from . import circ1\nVAL2 = 2\n", encoding="utf-8")

    # 7. Unrelated module
    (src_pkg / "unrelated.py").write_text("SECRET = 100\n", encoding="utf-8")

    # 8. Broken syntax file
    (src_pkg / "broken.py").write_text("def invalid_syntax(:\n    pass\n", encoding="utf-8")

    # 9. Unresolved internal import
    (src_pkg / "caller_unresolved.py").write_text("import pkg.non_existent_module\n", encoding="utf-8")

    # 10. File in venv (must be excluded)
    (venv_dir / "ignored.py").write_text("VENV_CODE = True\n", encoding="utf-8")

    repo.git.add(all=True)
    repo.index.commit(
        "feat: setup dependency graph test scenarios",
        author=git.Actor("Test", "test@test.com"),
        committer=git.Actor("Test", "test@test.com"),
        author_date="2026-01-01 12:00:00",
        commit_date="2026-01-01 12:00:00",
    )
    return repo


def test_build_dependency_graph(complex_repo: git.Repo):
    head_sha = complex_repo.head.commit.hexsha
    result = build_dependency_graph(complex_repo, head_sha, use_cache=False)

    g = result.graph
    assert isinstance(g, nx.DiGraph)

    # Verify exclusions: venv file must NOT be in graph
    assert "venv/ignored.py" not in g

    # Verify exact expected edges:
    # A -> B (a imports b)
    assert g.has_edge("src/pkg/a.py", "src/pkg/b.py")
    assert "import pkg.b" in g["src/pkg/a.py"]["src/pkg/b.py"]["import_statements"]

    # B -> C (b imports c)
    assert g.has_edge("src/pkg/b.py", "src/pkg/c.py")

    # Nested relative imports: nested.py -> c.py and nested.py -> b.py
    assert g.has_edge("src/pkg/sub/nested.py", "src/pkg/c.py")
    assert g.has_edge("src/pkg/sub/nested.py", "src/pkg/b.py")

    # Circular imports: mutual edges terminate cleanly
    assert g.has_edge("src/pkg/circ1.py", "src/pkg/circ2.py")
    assert g.has_edge("src/pkg/circ2.py", "src/pkg/circ1.py")

    # Unrelated module has node but no edges to other modules
    assert "src/pkg/unrelated.py" in g
    assert g.in_degree("src/pkg/unrelated.py") == 0
    assert g.out_degree("src/pkg/unrelated.py") == 0

    # Broken syntax file still added as node with parse_failed flag
    assert "src/pkg/broken.py" in g
    assert g.nodes["src/pkg/broken.py"]["parse_failed"] is True
    assert "src/pkg/broken.py" in result.parse_failures

    # External stdlib imports recorded
    assert "os" in result.external_imports or "json" in result.external_imports
    assert "os" not in g  # No nodes created for external modules

    # Unresolved import recorded
    assert "src/pkg/caller_unresolved.py" in result.unresolved_imports
    assert "pkg.non_existent_module" in result.unresolved_imports["src/pkg/caller_unresolved.py"]


def test_dependency_graph_caching(complex_repo: git.Repo):
    head_sha = complex_repo.head.commit.hexsha
    res1 = build_dependency_graph(complex_repo, head_sha, use_cache=True)
    res2 = build_dependency_graph(complex_repo, head_sha, use_cache=True)

    assert res2.cache_hit is True
    assert res2.total_nodes == res1.total_nodes
    assert res2.total_edges == res1.total_edges
