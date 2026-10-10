"""Unit tests for feature extractor: Shannon entropy math, commit feature extraction, and schema parity."""

import json
import math
from pathlib import Path
import pytest
import git
import pandas as pd

from impactmap.feature_extractor import (
    FEATURE_COLUMNS,
    FEATURE_DTYPES,
    compute_shannon_entropy,
    extract_commit_features,
    validate_feature_dict,
)
from tests.fixtures.repo_fixtures import GitRepoBuilder


def test_hand_worked_shannon_entropy():
    # 1. Single file: entropy should be 0.0
    assert compute_shannon_entropy([100]) == 0.0
    assert compute_shannon_entropy([]) == 0.0
    assert compute_shannon_entropy([0]) == 0.0

    # 2. Two equal files: p1 = 0.5, p2 = 0.5
    # H = - (0.5 * log2(0.5) + 0.5 * log2(0.5)) = - (-0.5 - 0.5) = 1.0
    assert compute_shannon_entropy([50, 50]) == 1.0

    # 3. Four equal files: p = 0.25 each
    # H = - 4 * (0.25 * log2(0.25)) = - 4 * (0.25 * -2) = 2.0
    assert compute_shannon_entropy([25, 25, 25, 25]) == 2.0

    # 4. Hand-worked unequal distribution [90, 10]
    # p1 = 0.9, p2 = 0.1
    # - (0.9 * log2(0.9) + 0.1 * log2(0.1)) = 0.4689955935892812
    expected = - (0.9 * math.log2(0.9) + 0.1 * math.log2(0.1))
    assert abs(compute_shannon_entropy([90, 10]) - expected) < 1e-5


def test_extract_commit_features(tmp_path: Path):
    builder = GitRepoBuilder(tmp_path / "repo")
    commits = builder.build_standard_repo()

    repo = git.Repo(tmp_path / "repo")

    # Initial commit features
    init_features = extract_commit_features(repo, commits["initial"])
    assert "la" in init_features.values
    assert init_features.values["nf"] == 4.0
    assert init_features.values["la"] > 0
    assert init_features.values["ld"] == 0.0
    assert init_features.values["ent"] > 0.0  # Multiple files added
    # Initial commit history is degraded
    assert "ndev" in init_features.missing_or_degraded

    # Modified commit features
    mod_features = extract_commit_features(repo, commits["modify"])
    assert mod_features.values["nf"] == 1.0
    assert mod_features.values["ent"] == 0.0  # 1 file modified -> entropy 0.0
    assert mod_features.values["la"] > 0.0
    assert mod_features.values["aexp"] >= 1.0  # Author has at least 1 prior commit


def test_schema_parity():
    # Load schema json
    schema_path = Path("models/feature_schema.json")
    assert schema_path.exists()
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_data = json.load(f)

    # 1. Feature columns match exactly in name and order
    assert list(FEATURE_COLUMNS) == schema_data["feature_columns"]

    # 2. Dtypes match
    for col in FEATURE_COLUMNS:
        assert FEATURE_DTYPES[col] == schema_data["feature_dtypes"][col]

    # 3. Check processed dataset if present
    train_path = Path("data/processed/train.csv")
    if train_path.exists():
        df_train = pd.read_csv(train_path, nrows=5)
        for col in FEATURE_COLUMNS:
            assert col in df_train.columns
        assert schema_data["target_column"] in df_train.columns


def test_validate_feature_dict():
    valid = {col: 1.0 for col in FEATURE_COLUMNS}
    ok, errors = validate_feature_dict(valid)
    assert ok is True
    assert len(errors) == 0

    # Missing column
    invalid = {col: 1.0 for col in FEATURE_COLUMNS if col != "ent"}
    ok, errors = validate_feature_dict(invalid)
    assert ok is False
    assert any("ent" in e for e in errors)

    # Non-finite value
    invalid_nan = dict(valid)
    invalid_nan["la"] = float("nan")
    ok, errors = validate_feature_dict(invalid_nan)
    assert ok is False
    assert any("la" in e for e in errors)
