"""Diff analysis: commit-vs-parent changes, numstat metrics, rename tracking, and file classification."""

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import git

from impactmap.config import (
    BINARY_EXTENSIONS,
    CONFIG_EXTENSIONS,
    DOC_EXTENSIONS,
    EMPTY_TREE_SHA,
    GENERATED_INDICATORS,
    MAX_DIFF_LINES,
    OTHER_SOURCE_EXTENSIONS,
    PYTHON_SOURCE_EXTENSIONS,
    TEST_PATH_INDICATORS,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileChange:
    """Detailed change information for a single file in a commit."""
    path: str
    old_path: Optional[str]
    status: str  # 'added', 'modified', 'deleted', 'renamed', 'copied'
    lines_added: int
    lines_deleted: int
    is_binary: bool
    category: str  # 'python_source', 'test', 'config', 'docs', 'generated_vendored', 'binary', 'other_source', 'other'
    patch_text: Optional[str] = None


@dataclass(frozen=True)
class CommitDiffSummary:
    """Aggregated diff metrics and file changes for a selected commit."""
    commit_sha: str
    parent_sha: Optional[str]
    is_initial_commit: bool
    is_merge_commit: bool
    is_empty_commit: bool
    has_truncated_history: bool
    files_changed_count: int
    lines_added_total: int
    lines_deleted_total: int
    total_churn: int
    file_changes: List[FileChange]
    notes: List[str]


def classify_file_category(path: str, is_binary: bool = False) -> str:
    """Transparently classifies a file into one of eight standardized categories.
    
    Categories:
    - binary: explicit binary files or binary extensions
    - test: test modules matching test_* or in test/ directory
    - python_source: non-test Python files (.py, .pyw, .pyi)
    - config: configuration files (.json, .yaml, .toml, .ini, etc.)
    - docs: documentation (.md, .rst, .txt, etc.)
    - generated_vendored: third-party, vendor, or build output
    - other_source: source code in other programming languages (.js, .java, etc.)
    - other: any other file type
    """
    if is_binary:
        return "binary"

    norm_path = "/" + path.replace("\\", "/").strip("/").lower()
    suffix = Path(path).suffix.lower()

    if suffix in BINARY_EXTENSIONS:
        return "binary"

    for gen in GENERATED_INDICATORS:
        if gen in norm_path:
            return "generated_vendored"

    # Test classification
    filename = Path(path).name.lower()
    is_test_file = any(
        indicator in norm_path for indicator in TEST_PATH_INDICATORS
    ) or filename.startswith("test_") or filename.endswith("_test.py")

    if suffix in PYTHON_SOURCE_EXTENSIONS:
        return "test" if is_test_file else "python_source"

    if suffix in CONFIG_EXTENSIONS or filename in {".gitignore", ".env.example", "dockerfile"}:
        return "config"

    if suffix in DOC_EXTENSIONS or filename in {"license", "copying", "notice", "authors", "changes"}:
        return "docs"

    if suffix in OTHER_SOURCE_EXTENSIONS:
        return "other_source"

    return "other"


def _parse_name_status_line(line: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """Parses a line from `git diff --name-status -M` into (status, new_path, old_path)."""
    parts = line.strip().split("\t")
    if not parts or not parts[0]:
        return None

    raw_code = parts[0].strip()
    status_char = raw_code[0].upper()

    if status_char == "R":  # Renamed: R100\told_path\tnew_path
        old_path = parts[1] if len(parts) > 1 else ""
        new_path = parts[2] if len(parts) > 2 else old_path
        return "renamed", new_path, old_path
    elif status_char == "C":  # Copied: C100\told_path\tnew_path
        old_path = parts[1] if len(parts) > 1 else ""
        new_path = parts[2] if len(parts) > 2 else old_path
        return "copied", new_path, old_path
    elif status_char == "A":
        return "added", parts[1], None
    elif status_char == "D":
        return "deleted", parts[1], None
    elif status_char == "M":
        return "modified", parts[1], None
    else:
        return "modified", parts[1], None


def _parse_numstat(numstat_output: str) -> Dict[str, Tuple[int, int, bool]]:
    """Parses `git diff --numstat` output into {file_path: (lines_added, lines_deleted, is_binary)}."""
    stats: Dict[str, Tuple[int, int, bool]] = {}
    if not numstat_output:
        return stats

    for line in numstat_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        added_str, deleted_str, path_str = parts[0], parts[1], parts[2]

        is_binary = (added_str == "-" and deleted_str == "-")
        la = 0 if is_binary else int(added_str) if added_str.isdigit() else 0
        ld = 0 if is_binary else int(deleted_str) if deleted_str.isdigit() else 0

        # Handle numstat rename syntax: {old => new}/file or old => new
        resolved_path = path_str
        if " => " in path_str:
            # Handle {old => new} or old => new
            match = re.search(r"\{.*? => (.*?)\}", path_str)
            if match:
                prefix = path_str[:match.start()]
                suffix = path_str[match.end():]
                resolved_path = prefix + match.group(1) + suffix
            else:
                parts_arrow = path_str.split(" => ")
                resolved_path = parts_arrow[1].strip()

        stats[resolved_path] = (la, ld, is_binary)
        # Also store raw path in case of direct lookup
        stats[path_str] = (la, ld, is_binary)

    return stats


def analyze_commit_diff(
    repo: git.Repo,
    commit_sha: str,
    find_renames: bool = True,
    max_diff_lines: int = MAX_DIFF_LINES,
) -> CommitDiffSummary:
    """Analyzes all file changes, line metrics, and statuses for a commit against its parent."""
    try:
        commit = repo.commit(commit_sha)
    except Exception as e:
        raise ValueError(f"Commit '{commit_sha}' not found in repository: {e}") from e

    notes: List[str] = []
    is_initial_commit = False
    is_merge_commit = False
    has_truncated_history = False
    parent_sha: Optional[str] = None

    if len(commit.parents) == 0:
        is_initial_commit = True
        notes.append("Initial commit: diffed against empty git tree.")
        parent_target = EMPTY_TREE_SHA
    elif len(commit.parents) > 1:
        is_merge_commit = True
        parent_sha = commit.parents[0].hexsha
        parent_target = parent_sha
        notes.append(f"Merge commit: diffed against first parent ({parent_sha[:7]}).")
    else:
        parent_sha = commit.parents[0].hexsha
        parent_target = parent_sha

    # Verify parent object is readable in case of shallow clone boundary
    try:
        _ = repo.commit(parent_target)
    except Exception:
        if not is_initial_commit:
            has_truncated_history = True
            notes.append("Parent commit not present in shallow clone; diffed against empty tree.")
            parent_target = EMPTY_TREE_SHA
            is_initial_commit = True

    # Run git diff --name-status -M
    diff_args = ["-M"] if find_renames else []
    try:
        name_status_raw = repo.git.diff("--name-status", *diff_args, parent_target, commit.hexsha)
        numstat_raw = repo.git.diff("--numstat", *diff_args, parent_target, commit.hexsha)
    except git.GitCommandError as e:
        logger.error("Git diff command failed: %s", e)
        raise RuntimeError(f"Failed to execute git diff: {e}") from e

    numstat_dict = _parse_numstat(numstat_raw)
    file_changes: List[FileChange] = []
    lines_added_total = 0
    lines_deleted_total = 0

    if name_status_raw.strip():
        for line in name_status_raw.splitlines():
            parsed = _parse_name_status_line(line)
            if not parsed:
                continue
            status, path, old_path = parsed

            # Lookup lines added and deleted from numstat
            numstat_info = numstat_dict.get(path) or numstat_dict.get(old_path or "")
            if numstat_info:
                la, ld, is_bin = numstat_info
            else:
                la, ld, is_bin = 0, 0, False

            category = classify_file_category(path, is_binary=is_bin)
            lines_added_total += la
            lines_deleted_total += ld

            file_changes.append(
                FileChange(
                    path=path,
                    old_path=old_path,
                    status=status,
                    lines_added=la,
                    lines_deleted=ld,
                    is_binary=is_bin,
                    category=category,
                )
            )

    is_empty_commit = (len(file_changes) == 0)
    if is_empty_commit:
        notes.append("Empty commit: zero files changed.")

    return CommitDiffSummary(
        commit_sha=commit.hexsha,
        parent_sha=parent_sha,
        is_initial_commit=is_initial_commit,
        is_merge_commit=is_merge_commit,
        is_empty_commit=is_empty_commit,
        has_truncated_history=has_truncated_history,
        files_changed_count=len(file_changes),
        lines_added_total=lines_added_total,
        lines_deleted_total=lines_deleted_total,
        total_churn=lines_added_total + lines_deleted_total,
        file_changes=file_changes,
        notes=notes,
    )


def get_file_diff_patch(
    repo: git.Repo,
    commit_sha: str,
    file_path: str,
    max_lines: int = MAX_DIFF_LINES,
) -> str:
    """Fetches formatted textual diff patch for a single file, safely truncated."""
    try:
        commit = repo.commit(commit_sha)
        parent_target = commit.parents[0].hexsha if commit.parents else EMPTY_TREE_SHA
        patch = repo.git.diff(parent_target, commit.hexsha, "--", file_path)
    except Exception as e:
        return f"Unable to retrieve diff patch: {e}"

    lines = patch.splitlines()
    if len(lines) > max_lines:
        truncated_patch = "\n".join(lines[:max_lines])
        truncated_patch += f"\n\n[Diff truncated: displayed {max_lines} of {len(lines)} lines]"
        return truncated_patch

    return patch
