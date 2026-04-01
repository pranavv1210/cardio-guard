"""
Phase 3 — Mel Spectrogram Generation & Dataset Split
Convert .npy segments to mel spectrogram images, split into train/val/test.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import librosa
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

from src.config import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    SPEC_HEIGHT, SPEC_WIDTH,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED,
    SPECTROGRAMS_DIR, SEGMENTS_DIR, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute a log-power mel spectrogram from audio.

    Args:
        audio: 1D audio signal (float32).
        sr: Sample rate.
        n_mels: Number of mel bands.
        n_fft: FFT window size.
        hop_length: Hop length.

    Returns:
        2D numpy array (n_mels, time_frames) in dB scale.
    """
    if len(audio) == 0:
        return np.zeros((n_mels, 1), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, power=2.0,
    )

    # Convert to log scale (dB)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db.astype(np.float32)


def spectrogram_to_image(
    mel_db: np.ndarray,
    height: int = SPEC_HEIGHT,
    width: int = SPEC_WIDTH,
) -> np.ndarray:
    """Convert mel spectrogram to a resized uint8 image (0-255).

    Args:
        mel_db: 2D mel spectrogram in dB.
        height: Output image height.
        width: Output image width.

    Returns:
        2D uint8 array (height, width) normalized to 0-255.
    """
    # Normalize to 0-255
    min_val = mel_db.min()
    max_val = mel_db.max()
    if max_val - min_val < 1e-6:
        img = np.zeros((height, width), dtype=np.uint8)
    else:
        normalized = (mel_db - min_val) / (max_val - min_val)
        img_raw = (normalized * 255).astype(np.uint8)

        # Resize using PIL
        pil_img = Image.fromarray(img_raw)
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        img = np.array(pil_img)

    return img


def save_spectrogram_png(
    img: np.ndarray,
    output_path: Path,
) -> None:
    """Save a spectrogram image as PNG.

    Args:
        img: 2D uint8 array.
        output_path: Path to save the PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pil_img = Image.fromarray(img)
    pil_img.save(str(output_path))


def patient_wise_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Split segments into train/val/test ensuring no source file leaks across splits.

    Args:
        df: Segments metadata with 'source_file' and 'class_idx' columns.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        seed: Random seed.

    Returns:
        DataFrame with added 'split' column ('train', 'val', 'test').
    """
    df = df.copy()
    groups = df["source_file"].values

    # First split: separate test set
    gss_test = GroupShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=seed,
    )
    train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

    # Second split: separate val from train
    val_fraction_of_remaining = val_ratio / (train_ratio + val_ratio)
    train_val_df = df.iloc[train_val_idx]
    train_val_groups = train_val_df["source_file"].values

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_fraction_of_remaining, random_state=seed,
    )
    train_idx_rel, val_idx_rel = next(
        gss_val.split(train_val_df, groups=train_val_groups)
    )

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    df["split"] = ""
    df.iloc[train_idx, df.columns.get_loc("split")] = "train"
    df.iloc[val_idx, df.columns.get_loc("split")] = "val"
    df.iloc[test_idx, df.columns.get_loc("split")] = "test"

    return df


def generate_spectrograms(
    segments_df: pd.DataFrame,
    output_base: Path = SPECTROGRAMS_DIR,
) -> pd.DataFrame:
    """Generate spectrogram PNGs for all segments.

    Args:
        segments_df: DataFrame with 'segment_path', 'segment_name', 'class_idx', 'split'.
        output_base: Base directory for spectrogram images.

    Returns:
        DataFrame with added 'spectrogram_path' column.
    """
    spec_paths = []
    total = len(segments_df)

    for idx, row in segments_df.iterrows():
        # Load segment
        seg = np.load(row["segment_path"])

        # Compute mel spectrogram
        mel_db = compute_mel_spectrogram(seg)

        # Convert to image
        img = spectrogram_to_image(mel_db)

        # Save to split/class/name.png
        split = row["split"]
        class_name = "normal" if row["class_idx"] == 0 else "abnormal"
        out_path = output_base / split / class_name / f"{row['segment_name']}.png"
        save_spectrogram_png(img, out_path)
        spec_paths.append(str(out_path))

        if (idx + 1) % 5000 == 0 or idx == total - 1:
            logger.info("Generated %d/%d spectrograms", idx + 1, total)

    segments_df = segments_df.copy()
    segments_df["spectrogram_path"] = spec_paths
    return segments_df


def run_phase3(segments_csv: Optional[str] = None) -> pd.DataFrame:
    """Execute the full Phase 3 pipeline.

    Args:
        segments_csv: Path to segments_metadata.csv. Auto-detected if None.

    Returns:
        DataFrame with spectrogram paths and split assignments.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=" * 60)
    logger.info("PHASE 3 — Mel Spectrogram Generation & Dataset Split")
    logger.info("=" * 60)

    # Load segments metadata
    if segments_csv is None:
        segments_csv = str(OUTPUT_DIR / "segments_metadata.csv")

    seg_df = pd.read_csv(segments_csv)
    logger.info("Loaded %d segments", len(seg_df))

    # Patient-wise split
    logger.info("Performing patient-wise split...")
    seg_df = patient_wise_split(seg_df)

    for split in ["train", "val", "test"]:
        subset = seg_df[seg_df["split"] == split]
        n = len(subset)
        normal = (subset["class_idx"] == 0).sum()
        abnormal = (subset["class_idx"] == 1).sum()
        sources = subset["source_file"].nunique()
        logger.info("  %s: %d segments (%d normal, %d abnormal) from %d recordings",
                     split, n, normal, abnormal, sources)

    # Verify no source file leaks
    train_sources = set(seg_df[seg_df["split"] == "train"]["source_file"])
    val_sources = set(seg_df[seg_df["split"] == "val"]["source_file"])
    test_sources = set(seg_df[seg_df["split"] == "test"]["source_file"])

    assert train_sources.isdisjoint(val_sources), "Train/val source leak!"
    assert train_sources.isdisjoint(test_sources), "Train/test source leak!"
    assert val_sources.isdisjoint(test_sources), "Val/test source leak!"
    logger.info("No source file leaks across splits ✓")

    # Generate spectrograms
    logger.info("Generating mel spectrograms...")
    seg_df = generate_spectrograms(seg_df)

    # Save updated metadata
    out_csv = OUTPUT_DIR / "spectrograms_metadata.csv"
    seg_df.to_csv(out_csv, index=False)
    logger.info("Saved spectrogram metadata to %s", out_csv)

    return seg_df


if __name__ == "__main__":
    run_phase3()
