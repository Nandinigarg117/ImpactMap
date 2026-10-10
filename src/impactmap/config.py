"""ImpactMap configuration: centralized limits, exclusions, weights, and thresholds."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Sequence, Tuple
import os

# Deterministic Seed
RANDOM_SEED: int = 42

# Resource and Safety Limits
MAX_FILE_BYTES: int = 1 * 1024 * 1024  # 1 MB per file blob
MAX_FILES: int = 500                    # Max files analyzed per commit
MAX_HISTORY_DEPTH: int = 500            # Bounded git history walk
MAX_DIFF_LINES: int = 2000              # Max lines displayed in UI diff viewer
MAX_PATHS_PER_NODE: int = 5             # Cap on enumerated dependency paths per node
MAX_DEPTH: int = 10                     # Max BFS distance for dependents
PATH_CAP: int = 10                      # Scaling constant for log1p(path_count)
MAGNITUDE_CAP: int = 500                # Scaling constant for log1p(churn)
CLONE_TIMEOUT_SECONDS: int = 60         # Safe subprocess clone timeout

# Allowed Remote Hosts for Repository Clones
ALLOWED_REMOTE_HOSTS: Tuple[str, ...] = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
)

# Allowed Local Roots (empty tuple allows any readable directory with UI notice)
ALLOWED_LOCAL_ROOTS: Tuple[Path, ...] = ()

# Directories excluded from Python module indexing
EXCLUDED_DIRS: FrozenSet[str] = frozenset({
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".git",
    "build",
    "dist",
    ".tox",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "site-packages",
    "migrations",
    ".eggs",
    "*.egg-info",
})

# File Categories and Classification Rules
PYTHON_SOURCE_EXTENSIONS: FrozenSet[str] = frozenset({".py", ".pyw", ".pyi"})
TEST_PATH_INDICATORS: Tuple[str, ...] = ("test_", "_test.py", "/tests/", "/test/", "/testing/")
CONFIG_EXTENSIONS: FrozenSet[str] = frozenset({
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".env", ".properties"
})
DOC_EXTENSIONS: FrozenSet[str] = frozenset({".md", ".rst", ".txt", ".adoc", ".pdf", ".html"})
GENERATED_INDICATORS: Tuple[str, ...] = (
    "/vendor/", "/vendored/", "/third_party/", "/dist/", "/build/", ".min.js", ".min.css"
)
BINARY_EXTENSIONS: FrozenSet[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".zip", ".tar", ".gz",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin", ".whl", ".db", ".sqlite"
})
OTHER_SOURCE_EXTENSIONS: FrozenSet[str] = frozenset({
    ".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp", ".h", ".hpp",
    ".go", ".rs", ".rb", ".php", ".sh", ".sql"
})

# Default Priority Ranking Weights (sums to 1.0)
DEFAULT_RANKING_WEIGHTS: Dict[str, float] = {
    "status": 0.35,
    "proximity": 0.30,
    "path_support": 0.10,
    "change_magnitude": 0.10,
    "history": 0.05,
    "commit_risk": 0.10,
}

# Priority Tier Thresholds
PRIORITY_TIER_THRESHOLDS: Dict[str, float] = {
    "Critical": 0.70,
    "High": 0.45,
    "Medium": 0.20,
}
# A higher commit risk lowers the High boundary by at most this amount. This
# changes tier labels; commit_risk itself remains a shared per-commit feature.
COMMIT_RISK_HIGH_TIER_SHIFT: float = 0.15
# Explicit additional evidence for imports broken by a removed/renamed module.
BROKEN_IMPORT_BOOST: float = 0.15

# Commit Risk Tier Thresholds
COMMIT_RISK_TIER_THRESHOLDS: Dict[str, float] = {
    "High": 0.60,
    "Medium": 0.30,
}

# Git Empty Tree SHA (standard SHA-1 empty tree)
EMPTY_TREE_SHA: str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


@dataclass(frozen=True)
class AppConfig:
    """Immutable application configuration with overrides."""
    max_file_bytes: int = MAX_FILE_BYTES
    max_files: int = MAX_FILES
    max_history_depth: int = MAX_HISTORY_DEPTH
    max_diff_lines: int = MAX_DIFF_LINES
    max_paths_per_node: int = MAX_PATHS_PER_NODE
    max_depth: int = MAX_DEPTH
    clone_timeout_seconds: int = CLONE_TIMEOUT_SECONDS
    allowed_remote_hosts: Tuple[str, ...] = ALLOWED_REMOTE_HOSTS
    allowed_local_roots: Tuple[Path, ...] = ALLOWED_LOCAL_ROOTS
    excluded_dirs: FrozenSet[str] = EXCLUDED_DIRS
    ranking_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_RANKING_WEIGHTS))
    random_seed: int = RANDOM_SEED

    def config_hash(self) -> str:
        """Hash representation of active configuration for cache invalidation."""
        import hashlib
        import json
        from dataclasses import asdict
        values = asdict(self)
        values["excluded_dirs"] = sorted(self.excluded_dirs)
        values["allowed_local_roots"] = sorted(str(p) for p in self.allowed_local_roots)
        values["allowed_remote_hosts"] = list(self.allowed_remote_hosts)
        payload = json.dumps(values, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
