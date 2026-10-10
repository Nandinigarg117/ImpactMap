"""Repository operations: safe loading, strict URL/path validation, secure cloning, and direct blob reading."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Dict, FrozenSet, List, Optional, Sequence, Set, Tuple
import urllib.parse

import git

from impactmap.config import (
    ALLOWED_LOCAL_ROOTS,
    ALLOWED_REMOTE_HOSTS,
    CLONE_TIMEOUT_SECONDS,
    EXCLUDED_DIRS,
    MAX_FILE_BYTES,
    MAX_FILES,
    MAX_HISTORY_DEPTH,
)

logger = logging.getLogger(__name__)


class RepositoryError(Exception):
    """Base exception for repository operations."""
    pass


class InvalidRepositoryPathError(RepositoryError):
    """Raised when local path is invalid, inaccessible, or outside allowed roots."""
    pass


class InvalidRepositoryURLError(RepositoryError):
    """Raised when remote repository URL is malformed, uses disallowed protocol or host."""
    pass


class EmptyRepositoryError(RepositoryError):
    """Raised when repository has no commits."""
    pass


class CorruptRepositoryError(RepositoryError):
    """Raised when git repository is bare, corrupted, or cannot be parsed."""
    pass


class CloneError(RepositoryError):
    """Raised when cloning fails or times out."""
    pass


@dataclass(frozen=True)
class CommitInfo:
    """Metadata for a commit presented in the UI or exports."""
    hexsha: str
    short_sha: str
    author_name: str
    message: str
    authored_datetime_iso: str
    authored_timestamp: int
    is_merge: bool
    parent_shas: Tuple[str, ...]


@dataclass(frozen=True)
class FileBlob:
    """Read-only file content extracted directly from a git commit blob."""
    path: str
    content_bytes: bytes
    is_truncated: bool
    byte_size: int

    @property
    def content_text(self) -> str:
        """Decode content with utf-8 fallback to replacement characters."""
        return self.content_bytes.decode("utf-8", errors="replace")


def validate_local_path(
    path_str: str | Path,
    allowed_roots: Sequence[Path] = ALLOWED_LOCAL_ROOTS,
) -> Path:
    """Strictly validates that the local path exists, is a directory, and contains a valid .git."""
    if not path_str or str(path_str).strip().startswith("-"):
        raise InvalidRepositoryPathError("Path cannot be empty or start with a dash.")

    raw_path = Path(path_str).expanduser()
    try:
        resolved = raw_path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as e:
        raise InvalidRepositoryPathError(f"Directory not found: {raw_path}") from e

    if not resolved.is_dir():
        raise InvalidRepositoryPathError(f"Path is not a directory: {resolved}")

    # Check allowed roots if specified
    if allowed_roots:
        resolved_roots = [r.resolve() for r in allowed_roots]
        is_allowed = any(
            resolved == r or r in resolved.parents
            for r in resolved_roots
        )
        if not is_allowed:
            raise InvalidRepositoryPathError(
                f"Path {resolved} is outside configured allowed roots: {allowed_roots}"
            )

    # Check .git existence
    dot_git = resolved / ".git"
    if not dot_git.exists():
        raise InvalidRepositoryPathError(f"Directory is not a Git repository (no .git found): {resolved}")

    return resolved


def validate_and_open_local_repo(
    path_str: str | Path,
    allowed_roots: Sequence[Path] = ALLOWED_LOCAL_ROOTS,
) -> git.Repo:
    """Validates local path and opens git.Repo, asserting at least one commit exists."""
    valid_path = validate_local_path(path_str, allowed_roots)
    try:
        repo = git.Repo(valid_path)
    except Exception as e:
        raise CorruptRepositoryError(f"Failed to open Git repository at {valid_path}: {e}") from e

    if repo.bare:
        raise CorruptRepositoryError(f"Bare repositories are not supported: {valid_path}")

    # Verify at least one commit exists
    try:
        _ = repo.head.commit
    except (ValueError, IndexError, git.GitCommandError) as e:
        raise EmptyRepositoryError(f"Repository has no commits: {valid_path}") from e

    return repo


def validate_remote_url(
    url: str,
    allowed_hosts: Sequence[str] = ALLOWED_REMOTE_HOSTS,
) -> str:
    """Validates remote URL: strictly https://, no user/password, allowlisted host, no dashes/control chars."""
    if not url or not isinstance(url, str):
        raise InvalidRepositoryURLError("URL must be a non-empty string.")

    cleaned_url = url.strip()
    if cleaned_url.startswith("-"):
        raise InvalidRepositoryURLError("URL cannot start with an option character '-'.")

    # Reject control characters or newlines
    if any(ord(c) < 32 or ord(c) == 127 for c in cleaned_url):
        raise InvalidRepositoryURLError("URL contains invalid control characters.")

    try:
        parsed = urllib.parse.urlsplit(cleaned_url)
    except Exception as e:
        raise InvalidRepositoryURLError(f"Invalid URL structure: {e}") from e

    if parsed.scheme.lower() != "https":
        raise InvalidRepositoryURLError(f"Only HTTPS URLs are allowed. Disallowed scheme: '{parsed.scheme}'")

    if parsed.username or parsed.password:
        raise InvalidRepositoryURLError("Embedded credentials (user:pass@) in URLs are strictly rejected.")

    netloc_host = (parsed.hostname or "").lower()
    if not netloc_host:
        raise InvalidRepositoryURLError("URL hostname is missing.")

    if not any(netloc_host == h.lower() or netloc_host.endswith("." + h.lower()) for h in allowed_hosts):
        raise InvalidRepositoryURLError(
            f"Host '{netloc_host}' is not in allowed hosts: {list(allowed_hosts)}"
        )

    # Path must look like /owner/repo or /owner/repo.git
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    if len(path_parts) < 2:
        raise InvalidRepositoryURLError(f"URL path '{parsed.path}' does not specify a repository.")

    # Reconstruct clean normalized URL without fragments or query params
    clean_path = "/" + "/".join(path_parts)
    if not clean_path.endswith(".git"):
        clean_path += ".git"

    return urllib.parse.urlunsplit(("https", netloc_host, clean_path, "", ""))


def safe_clone_repo(
    url: str,
    target_cache_dir: Path,
    max_history_depth: int = MAX_HISTORY_DEPTH,
    timeout_seconds: int = CLONE_TIMEOUT_SECONDS,
    allowed_hosts: Sequence[str] = ALLOWED_REMOTE_HOSTS,
) -> git.Repo:
    """Clones a remote repo into a managed cache directory using safe flags and subprocess.run."""
    valid_url = validate_remote_url(url, allowed_hosts)

    url_hash = hashlib.sha256(valid_url.encode("utf-8")).hexdigest()[:16]
    repo_dir = target_cache_dir / f"repo_{url_hash}"

    if (repo_dir / ".git").exists():
        logger.info("Found cached repository clone at %s", repo_dir)
        try:
            return git.Repo(repo_dir)
        except Exception:
            logger.warning("Existing cached clone was corrupt, removing and re-cloning.")
            shutil.rmtree(repo_dir, ignore_errors=True)

    repo_dir.mkdir(parents=True, exist_ok=True)
    empty_hooks_dir = target_cache_dir / "empty_hooks"
    empty_hooks_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "git",
        "-c", f"core.hooksPath={empty_hooks_dir}",
        "-c", "protocol.file.allow=never",
        "clone",
        "--no-checkout",
        "--no-recurse-submodules",
        "--depth", str(max(1, max_history_depth)),
        valid_url,
        str(repo_dir),
    ]

    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/local/bin"),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "echo",
        "HOME": os.environ.get("HOME", "/tmp"),
    }

    logger.info("Safe cloning %s into %s", valid_url, repo_dir)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            env=clean_env,
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise CloneError(f"Cloning timed out after {timeout_seconds} seconds.") from e
    except Exception as e:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise CloneError(f"Clone failed to start: {e}") from e

    if proc.returncode != 0:
        shutil.rmtree(repo_dir, ignore_errors=True)
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        # Ensure no credential tokens leaked
        sanitized_err = re.sub(r"://[^@]+@", "://", err_msg)
        raise CloneError(f"Git clone exited with code {proc.returncode}: {sanitized_err}")

    try:
        repo = git.Repo(repo_dir)
        _ = repo.head.commit
        return repo
    except Exception as e:
        shutil.rmtree(repo_dir, ignore_errors=True)
        raise CorruptRepositoryError(f"Cloned repository has no valid commits: {e}") from e


def list_recent_commits(
    repo: git.Repo,
    max_count: int = 100,
    search_query: Optional[str] = None,
) -> List[CommitInfo]:
    """Lists recent commits up to max_count with commit metadata, optional message/hash search."""
    commits: List[CommitInfo] = []
    try:
        iter_commits = repo.iter_commits("HEAD", max_count=max_count)
    except git.GitCommandError as e:
        raise RepositoryError(f"Error reading commits from HEAD: {e}") from e

    query = search_query.strip().lower() if search_query else None

    for c in iter_commits:
        msg = (c.message or "").strip()
        first_line = msg.splitlines()[0] if msg else "(no commit message)"
        short_sha = c.hexsha[:7]
        author = c.author.name if c.author else "Unknown"

        if query:
            if (query not in short_sha.lower() and
                query not in c.hexsha.lower() and
                query not in first_line.lower() and
                query not in author.lower()):
                continue

        dt = datetime.fromtimestamp(c.authored_date, tz=timezone.utc)
        parent_shas = tuple(p.hexsha for p in c.parents)

        commits.append(
            CommitInfo(
                hexsha=c.hexsha,
                short_sha=short_sha,
                author_name=author,
                message=first_line,
                authored_datetime_iso=dt.isoformat(),
                authored_timestamp=c.authored_date,
                is_merge=len(c.parents) > 1,
                parent_shas=parent_shas,
            )
        )

    return commits


def read_file_blob(
    commit: git.Commit,
    file_path: str,
    max_bytes: int = MAX_FILE_BYTES,
) -> Optional[FileBlob]:
    """Reads a file directly from the commit tree blob without checking out working tree."""
    posix_path = file_path.replace("\\", "/").strip("/")
    try:
        blob = commit.tree / posix_path
    except (KeyError, TypeError):
        return None

    # Skip non-blobs (submodules, trees)
    if not isinstance(blob, git.Blob):
        return None

    # Skip symlinks (mode 0o120000)
    if stat.S_ISLNK(blob.mode):
        logger.debug("Skipping symlink blob: %s", posix_path)
        return None

    size = blob.size
    is_truncated = False
    data_stream = blob.data_stream
    if size > max_bytes:
        raw_bytes = data_stream.read(max_bytes)
        is_truncated = True
    else:
        raw_bytes = data_stream.read()

    return FileBlob(
        path=posix_path,
        content_bytes=raw_bytes,
        is_truncated=is_truncated,
        byte_size=size,
    )


def list_python_files_in_commit(
    commit: git.Commit,
    excluded_dirs: FrozenSet[str] = EXCLUDED_DIRS,
    max_files: int = MAX_FILES,
) -> Tuple[List[str], bool]:
    """Traverses commit tree directly to find all .py files, respecting exclusions and file count cap."""
    py_files: List[str] = []
    is_truncated = False

    def _traverse(tree: git.Tree, prefix: str = ""):
        nonlocal is_truncated
        for item in tree:
            if is_truncated:
                return

            if isinstance(item, git.Tree):
                dirname = item.name
                if dirname in excluded_dirs or any(
                    dirname.startswith(exc.rstrip("*")) for exc in excluded_dirs if exc.endswith("*")
                ):
                    continue
                _traverse(item, prefix=f"{prefix}{dirname}/")

            elif isinstance(item, git.Blob):
                if item.name.endswith((".py", ".pyw", ".pyi")):
                    # Skip symlinks
                    if stat.S_ISLNK(item.mode):
                        continue
                    full_path = f"{prefix}{item.name}"
                    if len(py_files) >= max_files:
                        is_truncated = True
                        return
                    py_files.append(full_path)

    _traverse(commit.tree)
    py_files.sort()
    return py_files, is_truncated
