"""
Phase 1 — Dataset Acquisition & Inventory
Parse all REFERENCE.csv files, build master label dataframe, validate files.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import soundfile as sf

from src.config import (
    TRAINING_DIRS, VALIDATION_DIR, SAMPLE_RATE,
    LABEL_MAP, LABEL_NAMES, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def parse_reference_csv(csv_path: Path) -> pd.DataFrame:
    """Parse a single REFERENCE.csv file.

    Args:
        csv_path: Path to REFERENCE.csv (two columns: filename, label).

    Returns:
        DataFrame with columns [filename, label, database, wav_path].

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If CSV has unexpected format.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"REFERENCE.csv not found: {csv_path}")

    df = pd.read_csv(csv_path, header=None, names=["filename", "label"])

    if df.shape[1] != 2:
        raise ValueError(f"Expected 2 columns, got {df.shape[1]} in {csv_path}")

    if not df["label"].isin([-1, 1]).all():
        bad = df[~df["label"].isin([-1, 1])]["label"].unique()
        raise ValueError(f"Unexpected labels {bad} in {csv_path}")

    # Derive database name from parent folder (e.g., "training-a" → "a")
    db_folder = csv_path.parent.name
    db_key = db_folder.replace("training-", "")

    df["database"] = db_key
    df["wav_path"] = df["filename"].apply(
        lambda f: str(csv_path.parent / f"{f}.wav")
    )

    return df


def build_master_labels(
    training_dirs: Optional[dict] = None,
) -> pd.DataFrame:
    """Merge all REFERENCE.csv files into a single master DataFrame.

    Args:
        training_dirs: Dict mapping db_key → Path. Defaults to config.

    Returns:
        DataFrame with columns [filename, label, database, wav_path, class_idx, class_name].
    """
    if training_dirs is None:
        training_dirs = TRAINING_DIRS

    frames = []
    for key, dir_path in sorted(training_dirs.items()):
        csv_path = dir_path / "REFERENCE.csv"
        if not csv_path.exists():
            logger.warning("No REFERENCE.csv in %s — skipping", dir_path)
            continue
        df = parse_reference_csv(csv_path)
        frames.append(df)
        logger.info("Database %s: %d recordings", key, len(df))

    if not frames:
        raise RuntimeError("No REFERENCE.csv files found in any training directory")

    master = pd.concat(frames, ignore_index=True)

    # Map labels to class indices
    master["class_idx"] = master["label"].map(LABEL_MAP)
    master["class_name"] = master["class_idx"].map(LABEL_NAMES)

    return master


def validate_wav_files(master_df: pd.DataFrame) -> pd.DataFrame:
    """Check that all .wav files exist and are readable.

    Args:
        master_df: Master labels DataFrame with 'wav_path' column.

    Returns:
        DataFrame with added columns [exists, readable, duration, sr].
    """
    exists_list = []
    readable_list = []
    duration_list = []
    sr_list = []

    for _, row in master_df.iterrows():
        wav = Path(row["wav_path"])
        file_exists = wav.exists()
        exists_list.append(file_exists)

        if file_exists:
            try:
                info = sf.info(str(wav))
                readable_list.append(True)
                duration_list.append(info.duration)
                sr_list.append(info.samplerate)
            except Exception as e:
                logger.warning("Cannot read %s: %s", wav, e)
                readable_list.append(False)
                duration_list.append(0.0)
                sr_list.append(0)
        else:
            readable_list.append(False)
            duration_list.append(0.0)
            sr_list.append(0)

    master_df = master_df.copy()
    master_df["exists"] = exists_list
    master_df["readable"] = readable_list
    master_df["duration"] = duration_list
    master_df["sr"] = sr_list

    return master_df


def get_dataset_stats(master_df: pd.DataFrame) -> dict:
    """Compute summary statistics of the dataset.

    Args:
        master_df: Validated master DataFrame.

    Returns:
        Dict with total, per-database counts, class distribution, duration stats.
    """
    stats = {
        "total_recordings": len(master_df),
        "total_existing": int(master_df["exists"].sum()) if "exists" in master_df.columns else len(master_df),
        "per_database": {},
        "class_distribution": {},
        "duration": {},
    }

    # Per-database breakdown
    for db, group in master_df.groupby("database"):
        normal_count = int((group["label"] == -1).sum())
        abnormal_count = int((group["label"] == 1).sum())
        stats["per_database"][db] = {
            "total": len(group),
            "normal": normal_count,
            "abnormal": abnormal_count,
        }

    # Overall class distribution
    stats["class_distribution"] = {
        "normal": int((master_df["label"] == -1).sum()),
        "abnormal": int((master_df["label"] == 1).sum()),
    }

    # Duration stats (if validated)
    if "duration" in master_df.columns:
        valid = master_df[master_df["readable"] == True]
        if len(valid) > 0:
            stats["duration"] = {
                "min": float(valid["duration"].min()),
                "max": float(valid["duration"].max()),
                "mean": float(valid["duration"].mean()),
                "median": float(valid["duration"].median()),
            }

    return stats


def save_master_labels(master_df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
    """Save master labels to CSV.

    Args:
        master_df: Master labels DataFrame.
        output_path: Optional save path. Defaults to output/master_labels.csv.

    Returns:
        Path where file was saved.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "master_labels.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(output_path, index=False)
    logger.info("Saved master labels to %s", output_path)
    return output_path


def run_phase1() -> pd.DataFrame:
    """Execute the full Phase 1 pipeline.

    Returns:
        Validated master labels DataFrame.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=" * 60)
    logger.info("PHASE 1 — Dataset Acquisition & Inventory")
    logger.info("=" * 60)

    # Build master labels
    master = build_master_labels()
    logger.info("Total recordings found: %d", len(master))

    # Validate wav files
    logger.info("Validating .wav files...")
    master = validate_wav_files(master)

    missing = len(master) - int(master["exists"].sum())
    unreadable = int(master["exists"].sum()) - int(master["readable"].sum())
    logger.info("Missing files: %d | Unreadable: %d", missing, unreadable)

    # Stats
    stats = get_dataset_stats(master)
    logger.info("\n--- Dataset Summary ---")
    logger.info("Total: %d | Existing: %d", stats["total_recordings"], stats["total_existing"])

    for db, info in sorted(stats["per_database"].items()):
        logger.info(
            "  Database %s: %d total (%d normal, %d abnormal)",
            db, info["total"], info["normal"], info["abnormal"],
        )

    cd = stats["class_distribution"]
    logger.info("Class distribution: %d normal, %d abnormal", cd["normal"], cd["abnormal"])

    if stats["duration"]:
        d = stats["duration"]
        logger.info(
            "Duration: min=%.1fs, max=%.1fs, mean=%.1fs, median=%.1fs",
            d["min"], d["max"], d["mean"], d["median"],
        )

    # Save
    save_path = save_master_labels(master)
    logger.info("Master labels saved to: %s", save_path)

    return master


if __name__ == "__main__":
    run_phase1()
