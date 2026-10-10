"""Hermetic Git repository builders for test suites."""

from pathlib import Path
import subprocess
from typing import Dict
import git


class GitRepoBuilder:
    """Creates a real hermetic git repository in tmp_path with predictable commit history."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        self.repo = git.Repo.init(self.repo_dir, b="main")
        self.commits: Dict[str, str] = {}

        # Configure deterministic identity
        with self.repo.config_writer() as config:
            config.set_value("user", "name", "Test Author")
            config.set_value("user", "email", "test@example.com")
            config.set_value("commit", "gpgsign", "false")

    def _commit(self, message: str) -> str:
        self.repo.git.add(all=True)
        c = self.repo.index.commit(
            message,
            author=git.Actor("Test Author", "test@example.com"),
            committer=git.Actor("Test Author", "test@example.com"),
            author_date="2026-01-01 12:00:00",
            commit_date="2026-01-01 12:00:00",
        )
        return c.hexsha

    def build_standard_repo(self) -> Dict[str, str]:
        """Builds a rich sequence of commits:
        - initial: calc.py, __init__.py, test_calc.py, doc.md
        - modify: edits calc.py
        - add: adds utils.py
        - rename: renames utils.py -> helpers.py
        - delete: removes doc.md
        - binary: adds logo.png
        - empty: empty commit
        - merge: merge commit
        """
        # Commit 1: Initial commit
        pkg_dir = self.repo_dir / "src" / "pkg"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        test_dir = self.repo_dir / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)

        (pkg_dir / "__init__.py").write_text("# pkg init\n", encoding="utf-8")
        (pkg_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (test_dir / "test_calc.py").write_text("from pkg.calc import add\n\ndef test_add():\n    assert add(1, 2) == 3\n", encoding="utf-8")
        (self.repo_dir / "README.md").write_text("# Test Repo\nInitial documentation.\n", encoding="utf-8")
        self.commits["initial"] = self._commit("feat: initial commit")

        # Commit 2: Modify calc.py
        (pkg_dir / "calc.py").write_text("def add(a, b):\n    # add comments\n    return a + b\n\ndef sub(a, b):\n    return a - b\n", encoding="utf-8")
        self.commits["modify"] = self._commit("fix: add sub function")

        # Commit 3: Add utils.py
        (pkg_dir / "utils.py").write_text("def format_num(n):\n    return str(n)\n", encoding="utf-8")
        self.commits["add"] = self._commit("feat: add utils module")

        # Commit 4: Rename utils.py -> helpers.py
        self.repo.git.mv("src/pkg/utils.py", "src/pkg/helpers.py")
        self.commits["rename"] = self._commit("refactor: rename utils to helpers")

        # Commit 5: Delete README.md
        self.repo.git.rm("README.md")
        self.commits["delete"] = self._commit("docs: remove readme")

        # Commit 6: Binary file
        assets_dir = self.repo_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01")
        self.commits["binary"] = self._commit("chore: add binary logo")

        # Commit 7: Empty commit
        self.repo.git.commit("--allow-empty", "-m", "chore: empty commit trigger")
        self.commits["empty"] = self.repo.head.commit.hexsha

        # Commit 8: Branch and merge
        self.repo.git.checkout("-b", "feature-branch")
        (pkg_dir / "feature.py").write_text("FEATURE = True\n", encoding="utf-8")
        self._commit("feat: add feature module on branch")

        self.repo.git.checkout("main")
        (pkg_dir / "main_change.py").write_text("MAIN = True\n", encoding="utf-8")
        self._commit("chore: change on main")

        self.repo.git.merge("feature-branch", "--no-ff", "-m", "merge: merge feature-branch into main")
        self.commits["merge"] = self.repo.head.commit.hexsha

        return self.commits
