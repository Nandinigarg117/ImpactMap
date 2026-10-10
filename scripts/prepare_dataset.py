"""Dataset preparation script for ApacheJIT: schema validation, deduplication, and time-aware splitting."""

from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import shutil
import urllib.request

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
SCRATCH_FALLBACK = Path("/Users/nandinigarg/.gemini/antigravity-ide/brain/59e586eb-31d5-4b97-97fb-e8211396f771/scratch/apachejit/dataset/apachejit_total.csv")
APACHEJIT_URL = "https://raw.githubusercontent.com/hosseinkshvrz/apachejit/master/dataset/apachejit_total.csv"

EXPECTED_COLUMNS = [
    "commit_id", "project", "buggy", "fix", "year", "author_date",
    "la", "ld", "nf", "nd", "ns", "ent", "ndev", "age", "nuc", "aexp", "arexp", "asexp"
]


def download_or_locate_dataset() -> Path:
    """Idempotently ensures raw ApacheJIT dataset is present in data/raw/."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_csv = RAW_DATA_DIR / "apachejit_total.csv"

    if raw_csv.exists() and raw_csv.stat().st_size > 1000:
        logger.info("Raw dataset already present at %s (%d bytes)", raw_csv, raw_csv.stat().st_size)
        return raw_csv

    if SCRATCH_FALLBACK.exists() and SCRATCH_FALLBACK.stat().st_size > 1000:
        logger.info("Copying dataset from scratch cache: %s", SCRATCH_FALLBACK)
        shutil.copy2(SCRATCH_FALLBACK, raw_csv)
        return raw_csv

    logger.info("Downloading ApacheJIT from %s ...", APACHEJIT_URL)
    urllib.request.urlretrieve(APACHEJIT_URL, raw_csv)
    logger.info("Downloaded %d bytes to %s", raw_csv.stat().st_size, raw_csv)
    return raw_csv


def prepare_data():
    """Validates, cleans, splits, and writes dataset preparation reports."""
    raw_path = download_or_locate_dataset()
    logger.info("Loading raw dataset from %s", raw_path)
    df = pd.read_csv(raw_path)
    raw_count = len(df)
    logger.info("Raw row count: %d", raw_count)

    # 1. Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw dataset is missing expected columns: {missing_cols}")

    # 2. Check and drop duplicates by commit_id
    dups_count = int(df.duplicated(subset=["commit_id"]).sum())
    if dups_count > 0:
        logger.info("Dropping %d duplicate commit_ids", dups_count)
        df = df.drop_duplicates(subset=["commit_id"]).reset_index(drop=True)

    # 3. Handle data types & nulls
    df["buggy"] = df["buggy"].astype(int)
    null_count = int(df.isnull().sum().sum())
    if null_count > 0:
        logger.warning("Found %d null cells, dropping invalid rows", null_count)
        df = df.dropna().reset_index(drop=True)

    # Drop any impossible values (e.g. negative churn or files)
    valid_mask = (df["la"] >= 0) & (df["ld"] >= 0) & (df["nf"] >= 0) & (df["nd"] >= 0) & (df["ns"] >= 0) & (df["ent"] >= 0)
    invalid_rows = int((~valid_mask).sum())
    if invalid_rows > 0:
        logger.info("Dropping %d rows with negative metric values", invalid_rows)
        df = df[valid_mask].reset_index(drop=True)

    cleaned_count = len(df)
    logger.info("Cleaned row count: %d", cleaned_count)

    # 4. Time-aware split (70% train, 15% validation, 15% test by author_date)
    df = df.sort_values("author_date").reset_index(drop=True)

    n_train = int(cleaned_count * 0.70)
    n_val = int(cleaned_count * 0.15)

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    logger.info("Splits: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))

    # Save processed splits
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DATA_DIR / "val.csv", index=False)
    test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

    # Generate preparation report
    def _split_stats(split_df: pd.DataFrame) -> dict:
        min_ts = int(split_df["author_date"].min())
        max_ts = int(split_df["author_date"].max())
        return {
            "row_count": len(split_df),
            "buggy_count": int(split_df["buggy"].sum()),
            "clean_count": int(len(split_df) - split_df["buggy"].sum()),
            "buggy_ratio": round(float(split_df["buggy"].mean()), 4),
            "min_author_date_utc": datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat(),
            "max_author_date_utc": datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat(),
        }

    prep_report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": str(raw_path),
        "raw_row_count": raw_count,
        "cleaned_row_count": cleaned_count,
        "duplicates_dropped": dups_count,
        "overall_buggy_ratio": round(float(df["buggy"].mean()), 4),
        "project_counts": df["project"].value_counts().to_dict(),
        "train_split": _split_stats(train_df),
        "val_split": _split_stats(val_df),
        "test_split": _split_stats(test_df),
    }

    report_path = PROCESSED_DATA_DIR / "prep_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(prep_report, f, indent=2)

    logger.info("Preparation report saved to %s", report_path)
    return prep_report


if __name__ == "__main__":
    prepare_data()
