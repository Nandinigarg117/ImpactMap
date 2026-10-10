"""Tests for repository validation, safe URL handling, commit listing, and blob reading."""

from pathlib import Path
import pytest
import git

from impactmap.repository import (
    CorruptRepositoryError,
    EmptyRepositoryError,
    InvalidRepositoryPathError,
    InvalidRepositoryURLError,
    list_python_files_in_commit,
    list_recent_commits,
    read_file_blob,
    validate_and_open_local_repo,
    validate_local_path,
    validate_remote_url,
)
from tests.fixtures.repo_fixtures import GitRepoBuilder


def test_validate_local_path_success(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    builder.build_standard_repo()

    valid = validate_local_path(tmp_path / "repo")
    assert valid == (tmp_path / "repo").resolve()


def test_validate_local_path_errors(tmp_path: Path):
    # Non-existent
    with pytest.raises(InvalidRepositoryPathError):
        validate_local_path(tmp_path / "does_not_exist")

    # Leading dash
    with pytest.raises(InvalidRepositoryPathError):
        validate_local_path("-rf /")

    # Not a git repo
    regular_dir = tmp_path / "plain_dir"
    regular_dir.mkdir()
    with pytest.raises(InvalidRepositoryPathError, match="not a Git repository"):
        validate_local_path(regular_dir)

    # Outside allowed roots
    builder = GitRepoBuilder(tmp_path / "repo")
    builder.build_standard_repo()
    allowed_root = tmp_path / "other_allowed"
    allowed_root.mkdir()
    with pytest.raises(InvalidRepositoryPathError, match="outside configured allowed roots"):
        validate_local_path(tmp_path / "repo", allowed_roots=[allowed_root])


def test_empty_repo_raises_empty_error(tmp_path: Path):
    empty_dir = tmp_path / "empty_repo"
    empty_dir.mkdir()
    git.Repo.init(empty_dir)

    with pytest.raises(EmptyRepositoryError):
        validate_and_open_local_repo(empty_dir)


def test_validate_remote_url_success():
    url1 = "https://github.com/psf/requests.git"
    assert validate_remote_url(url1) == "https://github.com/psf/requests.git"

    url2 = "https://gitlab.com/group/project"
    assert validate_remote_url(url2) == "https://gitlab.com/group/project.git"

    url3 = "https://bitbucket.org/owner/repo"
    assert validate_remote_url(url3) == "https://bitbucket.org/owner/repo.git"


@pytest.mark.parametrize("invalid_url", [
    "http://github.com/psf/requests.git",         # non-https
    "ssh://git@github.com/psf/requests.git",       # ssh
    "file:///etc/passwd",                         # file protocol
    "https://user:token@github.com/psf/repo.git",  # embedded credentials
    "https://evil.com/malicious/repo.git",        # disallowed host
    "--upload-pack=touch /tmp/pwn",               # leading dash
    "https://github.com/single_part",             # incomplete repo path
    "https://github.com/\x00/repo.git",           # null byte / control char
    "",                                           # empty
])
def test_validate_remote_url_failures(invalid_url: str):
    with pytest.raises(InvalidRepositoryURLError):
        validate_remote_url(invalid_url)


def test_list_recent_commits(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = validate_and_open_local_repo(tmp_path / "repo")
    recent = list_recent_commits(repo, max_count=50)

    assert len(recent) >= 8
    shas = [c.hexsha for c in recent]
    assert commits["initial"] in shas
    assert commits["merge"] in shas

    # Verify merge commit flag
    merge_info = next(c for c in recent if c.hexsha == commits["merge"])
    assert merge_info.is_merge is True
    assert len(merge_info.parent_shas) == 2

    # Verify search query filter
    filtered = list_recent_commits(repo, max_count=50, search_query="sub function")
    assert len(filtered) == 1
    assert filtered[0].hexsha == commits["modify"]


def test_read_file_blob(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = validate_and_open_local_repo(tmp_path / "repo")
    initial_commit = repo.commit(commits["initial"])

    blob = read_file_blob(initial_commit, "src/pkg/calc.py")
    assert blob is not None
    assert "def add(a, b):" in blob.content_text
    assert blob.is_truncated is False

    # Non-existent file in this commit
    missing = read_file_blob(initial_commit, "non_existent.py")
    assert missing is None

    # Max bytes truncation
    small_blob = read_file_blob(initial_commit, "src/pkg/calc.py", max_bytes=10)
    assert small_blob is not None
    assert small_blob.is_truncated is True
    assert len(small_blob.content_bytes) == 10


def test_list_python_files_in_commit(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = validate_and_open_local_repo(tmp_path / "repo")
    initial_commit = repo.commit(commits["initial"])

    py_files, is_truncated = list_python_files_in_commit(initial_commit)
    assert is_truncated is False
    assert "src/pkg/__init__.py" in py_files
    assert "src/pkg/calc.py" in py_files
    assert "tests/test_calc.py" in py_files
    assert "README.md" not in py_files
