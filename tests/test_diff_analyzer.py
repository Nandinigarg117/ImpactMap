"""Tests for commit diff analysis, rename detection, numstat metrics, and file categorization."""

from pathlib import Path
import pytest
import git

from impactmap.diff_analyzer import (
    analyze_commit_diff,
    classify_file_category,
    get_file_diff_patch,
)
from tests.fixtures.repo_fixtures import GitRepoBuilder


def test_classify_file_category():
    assert classify_file_category("src/app.py") == "python_source"
    assert classify_file_category("tests/test_model.py") == "test"
    assert classify_file_category("src/deep_test.py") == "test"
    assert classify_file_category("requirements.txt") == "docs"  # .txt extension
    assert classify_file_category("config.yaml") == "config"
    assert classify_file_category(".gitignore") == "config"
    assert classify_file_category("README.md") == "docs"
    assert classify_file_category("LICENSE") == "docs"
    assert classify_file_category("image.png") == "binary"
    assert classify_file_category("vendor/lib.py") == "generated_vendored"
    assert classify_file_category("script.js") == "other_source"
    assert classify_file_category("data.unknown") == "other"


def test_initial_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["initial"])

    assert summary.is_initial_commit is True
    assert summary.is_merge_commit is False
    assert summary.is_empty_commit is False
    assert summary.files_changed_count == 4
    paths = [fc.path for fc in summary.file_changes]
    assert "src/pkg/__init__.py" in paths
    assert "src/pkg/calc.py" in paths
    assert "tests/test_calc.py" in paths
    assert "README.md" in paths
    assert all(fc.status == "added" for fc in summary.file_changes)
    assert summary.lines_added_total > 0
    assert summary.lines_deleted_total == 0


def test_modified_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["modify"])

    assert summary.is_initial_commit is False
    assert summary.files_changed_count == 1
    fc = summary.file_changes[0]
    assert fc.path == "src/pkg/calc.py"
    assert fc.status == "modified"
    assert fc.category == "python_source"
    assert fc.lines_added > 0


def test_add_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["add"])

    assert summary.files_changed_count == 1
    fc = summary.file_changes[0]
    assert fc.path == "src/pkg/utils.py"
    assert fc.status == "added"


def test_rename_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["rename"], find_renames=True)

    assert summary.files_changed_count == 1
    fc = summary.file_changes[0]
    assert fc.status == "renamed"
    assert fc.path == "src/pkg/helpers.py"
    assert fc.old_path == "src/pkg/utils.py"


def test_delete_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["delete"])

    assert summary.files_changed_count == 1
    fc = summary.file_changes[0]
    assert fc.status == "deleted"
    assert fc.path == "README.md"
    assert fc.lines_deleted > 0


def test_binary_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["binary"])

    assert summary.files_changed_count == 1
    fc = summary.file_changes[0]
    assert fc.path == "assets/logo.png"
    assert fc.is_binary is True
    assert fc.category == "binary"


def test_empty_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["empty"])

    assert summary.is_empty_commit is True
    assert summary.files_changed_count == 0
    assert summary.total_churn == 0


def test_merge_commit_diff(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    summary = analyze_commit_diff(repo, commits["merge"])

    assert summary.is_merge_commit is True
    assert summary.parent_sha is not None


def test_get_file_diff_patch(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")
    patch = get_file_diff_patch(repo, commits["modify"], "src/pkg/calc.py")

    assert "def sub(a, b):" in patch
    assert "---" in patch
    assert "+++" in patch


def test_missing_parent_is_reported_as_truncated_history(tmp_path: Path, monkeypatch):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()
    repo = git.Repo(tmp_path / "repo")
    commit = repo.commit(commits["modify"])
    parent_sha = commit.parents[0].hexsha
    real_commit = repo.commit

    def missing_parent(rev=None, *args, **kwargs):
        if rev == parent_sha:
            raise ValueError("parent is outside shallow history")
        return real_commit(rev, *args, **kwargs)

    monkeypatch.setattr(repo, "commit", missing_parent)
    summary = analyze_commit_diff(repo, commits["modify"])
    assert summary.has_truncated_history is True
    assert summary.is_initial_commit is True
    assert summary.files_changed_count > 0
