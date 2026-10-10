"""Feature schema and extractor: shared commit-level metrics for training and inference."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import git

from impactmap.config import MAX_HISTORY_DEPTH
from impactmap.diff_analyzer import CommitDiffSummary, analyze_commit_diff

SCHEMA_VERSION: str = "1.0.0"

# Ordered list of features used by the production ML model
FEATURE_COLUMNS: Tuple[str, ...] = (
    "la",    # Lines added in commit
    "ld",    # Lines deleted in commit
    "nf",    # Number of files modified
    "nd",    # Number of directories modified
    "ns",    # Number of subsystems modified (top-level directories)
    "ent",   # Shannon entropy of change across files
    "ndev",  # Mean distinct previous developers per modified file
    "age",   # Mean days since previous modification per modified file
    "nuc",   # Mean unique prior changes per modified file
    "aexp",  # Author prior commit experience in repository
)

FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "la": "Lines added across all modified files in the commit.",
    "ld": "Lines deleted across all modified files in the commit.",
    "nf": "Number of files modified in the commit.",
    "nd": "Number of directories modified in the commit.",
    "ns": "Number of top-level subsystems touched by the commit.",
    "ent": "Shannon entropy (base 2) measuring how diffusely changes are spread across files.",
    "ndev": "Average number of distinct past developers who previously changed the touched files.",
    "age": "Average age (in days) since the modified files were last touched.",
    "nuc": "Average number of prior commits that previously modified the touched files.",
    "aexp": "Number of prior commits authored by this developer in the repository.",
}

FEATURE_DTYPES: Dict[str, str] = {
    "la": "int64",
    "ld": "int64",
    "nf": "int64",
    "nd": "int64",
    "ns": "int64",
    "ent": "float64",
    "ndev": "float64",
    "age": "float64",
    "nuc": "float64",
    "aexp": "int64",
}


@dataclass(frozen=True)
class ExtractedFeatures:
    """Standardized commit features with degradation tracking for inference and training."""
    values: Dict[str, float]
    is_complete: bool
    missing_or_degraded: List[str]
    notes: List[str]

    def to_ordered_vector(self) -> List[float]:
        """Returns values in exact schema column order."""
        return [self.values[col] for col in FEATURE_COLUMNS]


def compute_shannon_entropy(churn_per_file: Sequence[int]) -> float:
    """Computes Shannon entropy (base 2) of changes across modified files.
    
    Formula:
    H = - sum(p_i * log2(p_i)) where p_i = churn_i / sum(churn)
    If sum(churn) == 0 or len(churn_per_file) <= 1, H = 0.0.
    """
    total_churn = sum(churn_per_file)
    if total_churn == 0 or len(churn_per_file) <= 1:
        return 0.0

    entropy = 0.0
    for churn in churn_per_file:
        if churn > 0:
            p_i = churn / total_churn
            entropy -= p_i * math.log2(p_i)

    return float(round(entropy, 6))


def extract_commit_features(
    repo: git.Repo,
    commit_sha: str,
    diff_summary: Optional[CommitDiffSummary] = None,
    max_history_depth: int = MAX_HISTORY_DEPTH,
) -> ExtractedFeatures:
    """Extracts the 10 Kamei metrics from git commit and diff identically to training data.
    
    - la, ld, nf, nd, ns, ent: computed directly from diff.
    - ndev, age, nuc, aexp: reconstructed from bounded git history.
    - Gracefully tracks when history is shallow or features are degraded.
    """
    try:
        commit = repo.commit(commit_sha)
    except Exception as e:
        raise ValueError(f"Commit '{commit_sha}' not found: {e}") from e

    diff = diff_summary or analyze_commit_diff(repo, commit_sha)
    notes: List[str] = []
    degraded: List[str] = []

    # 1. Size and Diffusion metrics
    la = float(diff.lines_added_total)
    ld = float(diff.lines_deleted_total)
    nf = float(diff.files_changed_count)

    file_paths = [fc.path for fc in diff.file_changes]
    churns = [fc.lines_added + fc.lines_deleted for fc in diff.file_changes]

    # Directories touched
    dirs: Set[str] = set()
    subsystems: Set[str] = set()
    for p in file_paths:
        parts = p.replace("\\", "/").strip("/").split("/")
        if len(parts) > 1:
            dirs.add("/".join(parts[:-1]))
            subsystems.add(parts[0])
        else:
            dirs.add(".")
            subsystems.add(".")

    nd = float(len(dirs)) if file_paths else 0.0
    ns = float(len(subsystems)) if file_paths else 0.0
    ent = compute_shannon_entropy(churns)

    # 2. History metrics from bounded git log
    if diff.is_initial_commit or not commit.parents:
        ndev = 0.0
        age = 0.0
        nuc = 0.0
        aexp = 0.0
        degraded.extend(["ndev", "age", "nuc", "aexp"])
        notes.append("Initial commit: past history features set to 0.0 and marked degraded.")
    else:
        parent = commit.parents[0]
        # Check historical ancestry depth
        try:
            sample_history = list(repo.iter_commits(parent, max_count=5))
            is_shallow = len(sample_history) < 3
        except Exception:
            is_shallow = True

        if is_shallow:
            degraded.extend(["ndev", "age", "nuc", "aexp"])
            notes.append("Shallow git history detected: history features may have reduced accuracy.")

        # Compute author experience (aexp): prior commits by this author
        author_name = commit.author.name if commit.author else ""
        try:
            author_commits_cnt = 0
            for c in repo.iter_commits(parent, max_count=max_history_depth):
                if c.author and c.author.name == author_name:
                    author_commits_cnt += 1
            aexp = float(author_commits_cnt)
        except Exception:
            aexp = 0.0
            degraded.append("aexp")

        # Per-file metrics: ndev, age, nuc
        file_ndev_list: List[int] = []
        file_age_list: List[float] = []
        file_nuc_list: List[int] = []

        commit_ts = commit.authored_date

        for path in file_paths:
            try:
                prior_file_commits = list(repo.iter_commits(parent, paths=path, max_count=max_history_depth))
                if prior_file_commits:
                    unique_devs = {c.author.name for c in prior_file_commits if c.author}
                    file_ndev_list.append(len(unique_devs))
                    file_nuc_list.append(len(prior_file_commits))

                    last_commit_ts = prior_file_commits[0].authored_date
                    age_days = max(0.0, (commit_ts - last_commit_ts) / 86400.0)
                    file_age_list.append(age_days)
                else:
                    file_ndev_list.append(0)
                    file_age_list.append(0.0)
                    file_nuc_list.append(0)
            except Exception:
                file_ndev_list.append(0)
                file_age_list.append(0.0)
                file_nuc_list.append(0)

        ndev = float(sum(file_ndev_list) / len(file_ndev_list)) if file_ndev_list else 0.0
        age = float(round(sum(file_age_list) / len(file_age_list), 2)) if file_age_list else 0.0
        nuc = float(round(sum(file_nuc_list) / len(file_nuc_list), 2)) if file_nuc_list else 0.0

    values: Dict[str, float] = {
        "la": la,
        "ld": ld,
        "nf": nf,
        "nd": nd,
        "ns": ns,
        "ent": ent,
        "ndev": ndev,
        "age": age,
        "nuc": nuc,
        "aexp": aexp,
    }

    is_complete = len(degraded) == 0

    return ExtractedFeatures(
        values=values,
        is_complete=is_complete,
        missing_or_degraded=sorted(list(set(degraded))),
        notes=notes,
    )


def validate_feature_dict(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates that a dictionary matches FEATURE_COLUMNS schema exactly."""
    errors: List[str] = []
    for col in FEATURE_COLUMNS:
        if col not in features:
            errors.append(f"Missing required feature: '{col}'")
        else:
            val = features[col]
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                errors.append(f"Feature '{col}' has invalid non-finite value: {val}")

    return len(errors) == 0, errors
