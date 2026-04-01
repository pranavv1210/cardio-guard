"""
Tests for Phase 3 — Mel Spectrogram Generation & Dataset Split
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.config import SAMPLE_RATE, N_MELS, SPEC_HEIGHT, SPEC_WIDTH, SEGMENT_SAMPLES
from src.phase3_spectrograms import (
    compute_mel_spectrogram,
    spectrogram_to_image,
    save_spectrogram_png,
    patient_wise_split,
    generate_spectrograms,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def audio_segment():
    """A 3-second audio segment at 2kHz."""
    np.random.seed(42)
    t = np.linspace(0, 3.0, SEGMENT_SAMPLES, endpoint=False)
    return np.sin(2 * np.pi * 100 * t).astype(np.float32)


@pytest.fixture
def sample_segments_df(tmp_path):
    """Create a small segments DataFrame with .npy files."""
    seg_dir = tmp_path / "segments"
    seg_dir.mkdir()

    rows = []
    for i in range(20):
        source = f"rec{i // 4:03d}"  # 5 unique source files
        class_idx = i % 2
        name = f"{source}_seg{i:04d}"
        path = seg_dir / f"{name}.npy"

        audio = np.random.randn(SEGMENT_SAMPLES).astype(np.float32)
        np.save(str(path), audio)

        rows.append({
            "segment_name": name,
            "source_file": source,
            "segment_idx": i,
            "class_idx": class_idx,
            "label": -1 if class_idx == 0 else 1,
            "segment_path": str(path),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# compute_mel_spectrogram
# ──────────────────────────────────────────────────────────────────────────────
class TestComputeMelSpectrogram:
    def test_output_shape(self, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        assert mel.shape[0] == N_MELS
        assert mel.shape[1] > 0

    def test_output_dtype(self, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        assert mel.dtype == np.float32

    def test_db_scale_negative(self, audio_segment):
        """Log-power with ref=max should produce values <= 0."""
        mel = compute_mel_spectrogram(audio_segment)
        assert mel.max() <= 0.01  # Close to 0 dB max

    def test_empty_audio(self):
        mel = compute_mel_spectrogram(np.array([], dtype=np.float32))
        assert mel.shape[0] == N_MELS


# ──────────────────────────────────────────────────────────────────────────────
# spectrogram_to_image
# ──────────────────────────────────────────────────────────────────────────────
class TestSpectrogramToImage:
    def test_output_shape(self, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        img = spectrogram_to_image(mel)
        assert img.shape == (SPEC_HEIGHT, SPEC_WIDTH)

    def test_output_dtype(self, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        img = spectrogram_to_image(mel)
        assert img.dtype == np.uint8

    def test_range_0_255(self, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        img = spectrogram_to_image(mel)
        assert img.min() >= 0
        assert img.max() <= 255

    def test_constant_input(self):
        """Constant spectrogram should produce all-black image."""
        mel = np.ones((N_MELS, 50), dtype=np.float32) * -80.0
        img = spectrogram_to_image(mel)
        assert img.max() == 0  # All zeros


# ──────────────────────────────────────────────────────────────────────────────
# save_spectrogram_png
# ──────────────────────────────────────────────────────────────────────────────
class TestSaveSpectrogramPng:
    def test_saves_valid_png(self, tmp_path, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        img = spectrogram_to_image(mel)

        out = tmp_path / "test_spec.png"
        save_spectrogram_png(img, out)

        assert out.exists()
        loaded = Image.open(out)
        assert loaded.size == (SPEC_WIDTH, SPEC_HEIGHT)

    def test_creates_subdirs(self, tmp_path, audio_segment):
        mel = compute_mel_spectrogram(audio_segment)
        img = spectrogram_to_image(mel)

        out = tmp_path / "deep" / "nested" / "spec.png"
        save_spectrogram_png(img, out)
        assert out.exists()


# ──────────────────────────────────────────────────────────────────────────────
# patient_wise_split
# ──────────────────────────────────────────────────────────────────────────────
class TestPatientWiseSplit:
    def test_all_segments_assigned(self, sample_segments_df):
        df = patient_wise_split(sample_segments_df)
        assert "split" in df.columns
        assert (df["split"] != "").all()
        assert set(df["split"].unique()).issubset({"train", "val", "test"})

    def test_no_source_leaks(self, sample_segments_df):
        df = patient_wise_split(sample_segments_df)
        train_src = set(df[df["split"] == "train"]["source_file"])
        val_src = set(df[df["split"] == "val"]["source_file"])
        test_src = set(df[df["split"] == "test"]["source_file"])

        assert train_src.isdisjoint(val_src)
        assert train_src.isdisjoint(test_src)
        assert val_src.isdisjoint(test_src)

    def test_train_is_largest(self, sample_segments_df):
        df = patient_wise_split(sample_segments_df)
        train_count = (df["split"] == "train").sum()
        val_count = (df["split"] == "val").sum()
        test_count = (df["split"] == "test").sum()
        assert train_count >= val_count
        assert train_count >= test_count

    def test_reproducible(self, sample_segments_df):
        df1 = patient_wise_split(sample_segments_df, seed=42)
        df2 = patient_wise_split(sample_segments_df, seed=42)
        assert (df1["split"] == df2["split"]).all()

    def test_does_not_mutate_input(self, sample_segments_df):
        original_cols = set(sample_segments_df.columns)
        _ = patient_wise_split(sample_segments_df)
        assert set(sample_segments_df.columns) == original_cols


# ──────────────────────────────────────────────────────────────────────────────
# generate_spectrograms (integration test)
# ──────────────────────────────────────────────────────────────────────────────
class TestGenerateSpectrograms:
    def test_generates_pngs(self, sample_segments_df, tmp_path):
        df = patient_wise_split(sample_segments_df)
        out_dir = tmp_path / "specs"

        result = generate_spectrograms(df, output_base=out_dir)

        assert "spectrogram_path" in result.columns
        for _, row in result.iterrows():
            assert Path(row["spectrogram_path"]).exists()

    def test_correct_folder_structure(self, sample_segments_df, tmp_path):
        df = patient_wise_split(sample_segments_df)
        out_dir = tmp_path / "specs"

        generate_spectrograms(df, output_base=out_dir)

        # Check folder structure: specs/train/normal/, specs/train/abnormal/, etc.
        for split in ["train", "val", "test"]:
            split_dir = out_dir / split
            if split_dir.exists():
                subfolders = {p.name for p in split_dir.iterdir() if p.is_dir()}
                assert subfolders.issubset({"normal", "abnormal"})
