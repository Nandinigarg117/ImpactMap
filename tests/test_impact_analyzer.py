"""Unit tests for impact analyzer: reverse BFS traversal, distances, shortest chains, and deleted module handling."""

from pathlib import Path
import pytest
import git

from impactmap.config import AppConfig
from impactmap.diff_analyzer import analyze_commit_diff
from impactmap.impact_analyzer import analyze_impact


@pytest.fixture
def impact_test_repo(tmp_path: Path) -> git.Repo:
    """Builds a test repo designed to verify impact analysis:
    Commit 1:
      - c.py (leaf)
      - b.py (imports c)
      - a.py (imports b) -> chain a -> b -> c
      - side.py (imports c)
      - unrelated.py (no imports)
      - to_delete.py
      - caller_of_deleted.py (imports to_delete)
    Commit 2:
      - modifies c.py (seed)
    Commit 3:
      - deletes to_delete.py
    """
    repo_dir = tmp_path / "impact_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(repo_dir, b="main")

    src = repo_dir / "src"
    src.mkdir(parents=True, exist_ok=True)

    # Base files
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "c.py").write_text("C_VAR = 1\n", encoding="utf-8")
    (src / "b.py").write_text("import src.c\nB_VAR = src.c.C_VAR + 1\n", encoding="utf-8")
    (src / "a.py").write_text("import src.b\nA_VAR = src.b.B_VAR + 1\n", encoding="utf-8")
    (src / "side.py").write_text("import src.c\nSIDE_VAR = src.c.C_VAR\n", encoding="utf-8")
    (src / "unrelated.py").write_text("UNRELATED_VAR = 99\n", encoding="utf-8")
    (src / "to_delete.py").write_text("OLD = True\n", encoding="utf-8")
    (src / "caller_of_deleted.py").write_text("import src.to_delete\nVAL = src.to_delete.OLD\n", encoding="utf-8")

    repo.git.add(all=True)
    repo.index.commit(
        "feat: base commit",
        author=git.Actor("Test", "test@test.com"),
        committer=git.Actor("Test", "test@test.com"),
        author_date="2026-01-01 12:00:00",
        commit_date="2026-01-01 12:00:00",
    )

    # Commit 2: modify c.py
    (src / "c.py").write_text("C_VAR = 2  # modified\n", encoding="utf-8")
    repo.git.add("src/c.py")
    repo.index.commit(
        "fix: modify c.py",
        author=git.Actor("Test", "test@test.com"),
        committer=git.Actor("Test", "test@test.com"),
        author_date="2026-01-01 13:00:00",
        commit_date="2026-01-01 13:00:00",
    )

    # Commit 3: delete to_delete.py
    repo.git.rm("src/to_delete.py")
    repo.index.commit(
        "refactor: delete to_delete.py",
        author=git.Actor("Test", "test@test.com"),
        committer=git.Actor("Test", "test@test.com"),
        author_date="2026-01-01 14:00:00",
        commit_date="2026-01-01 14:00:00",
    )

    return repo


def test_impact_analysis_direct_and_indirect(impact_test_repo: git.Repo):
    # Commit 2 modifies src/c.py
    commits = list(impact_test_repo.iter_commits("HEAD"))
    c2_sha = commits[1].hexsha  # Commit 2: modify c.py

    diff_summary = analyze_commit_diff(impact_test_repo, c2_sha)
    assert diff_summary.files_changed_count == 1
    assert diff_summary.file_changes[0].path == "src/c.py"

    res = analyze_impact(impact_test_repo, diff_summary)

    # Verify seed
    assert "src/c.py" in res.evidence_records
    c_ev = res.evidence_records["src/c.py"]
    assert c_ev.category == "changed"
    assert c_ev.distance == 0

    # Verify direct dependents (distance = 1): b.py and side.py directly import c.py
    assert "src/b.py" in res.evidence_records
    b_ev = res.evidence_records["src/b.py"]
    assert b_ev.category == "direct"
    assert b_ev.distance == 1
    assert b_ev.shortest_path == ["src/b.py", "src/c.py"]
    assert "src/b.py → src/c.py" in b_ev.shortest_path_str

    assert "src/side.py" in res.evidence_records
    side_ev = res.evidence_records["src/side.py"]
    assert side_ev.category == "direct"
    assert side_ev.distance == 1

    # Verify indirect dependent (distance = 2): a.py imports b.py which imports c.py
    assert "src/a.py" in res.evidence_records
    a_ev = res.evidence_records["src/a.py"]
    assert a_ev.category == "indirect"
    assert a_ev.distance == 2
    assert a_ev.shortest_path == ["src/a.py", "src/b.py", "src/c.py"]
    assert "src/a.py → src/b.py → src/c.py" in a_ev.shortest_path_str

    # Verify unrelated
    assert "src/unrelated.py" in res.evidence_records
    unrel_ev = res.evidence_records["src/unrelated.py"]
    assert unrel_ev.category == "unrelated"
    assert unrel_ev.distance == -1

    # Verify counts
    assert res.changed_files_count == 1
    assert res.direct_dependents_count >= 2
    assert res.indirect_dependents_count >= 1
    assert res.max_distance == 2


def test_impact_analysis_deleted_module(impact_test_repo: git.Repo):
    # Commit 3 deletes src/to_delete.py
    head_sha = impact_test_repo.head.commit.hexsha
    diff_summary = analyze_commit_diff(impact_test_repo, head_sha)
    assert any(fc.status == "deleted" for fc in diff_summary.file_changes)

    res = analyze_impact(impact_test_repo, diff_summary)

    # In current commit, caller_of_deleted still exists and imported to_delete
    assert "src/caller_of_deleted.py" in res.evidence_records
    caller_ev = res.evidence_records["src/caller_of_deleted.py"]
    assert caller_ev.category == "direct"
    assert caller_ev.is_broken_import is True
    assert "Imports removed module" in (caller_ev.broken_import_details or "")


def test_dependencies_of_changed_files_context(impact_test_repo: git.Repo):
    # If a commit modifies b.py (which imports c.py), c.py is in dependencies_of_changed, NOT an impacted dependent!
    c2_sha = list(impact_test_repo.iter_commits("HEAD"))[1].hexsha
    diff_summary = analyze_commit_diff(impact_test_repo, c2_sha)
    res = analyze_impact(impact_test_repo, diff_summary)

    # c.py does not import anything in this repo
    assert res.dependencies_of_changed.get("src/c.py") == []


def test_rename_mapping_finds_dependents_that_keep_old_import(tmp_path: Path):
    repo_dir = tmp_path / "rename_repo"
    repo_dir.mkdir()
    repo = git.Repo.init(repo_dir, b="main")
    (repo_dir / "src/pkg").mkdir(parents=True)
    (repo_dir / "src/pkg/__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "src/pkg/old_name.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_dir / "src/pkg/caller.py").write_text("import pkg.old_name\n", encoding="utf-8")
    repo.git.add(all=True)
    repo.index.commit("base")

    repo.git.mv("src/pkg/old_name.py", "src/pkg/new_name.py")
    repo.index.commit("rename module without updating caller")
    summary = analyze_commit_diff(repo, repo.head.commit.hexsha)
    renamed = next(fc for fc in summary.file_changes if fc.status == "renamed")
    assert (renamed.old_path, renamed.path) == ("src/pkg/old_name.py", "src/pkg/new_name.py")

    result = analyze_impact(repo, summary)
    caller = result.evidence_records["src/pkg/caller.py"]
    assert caller.category == "direct"
    assert caller.is_broken_import is True
    assert "renamed to 'src/pkg/new_name.py'" in caller.broken_import_details
