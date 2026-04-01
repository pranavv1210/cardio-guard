"""
Tests for Phase 1 — Dataset Acquisition & Inventory
"""
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.config import TRAINING_DIRS, LABEL_MAP, LABEL_NAMES, SAMPLE_RATE
from src.phase1_data_inventory import (
    parse_reference_csv,
    build_master_labels,
    validate_wav_files,
    get_dataset_stats,
    save_master_labels,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal REFERENCE.csv for testing."""
    db_dir = tmp_path / "training-x"
    db_dir.mkdir()
    csv = db_dir / "REFERENCE.csv"
    csv.write_text("x0001,-1\nx0002,1\nx0003,-1\n")
    return csv


@pytest.fixture
def sample_csv_with_wavs(tmp_path):
    """Create REFERENCE.csv with actual tiny .wav files."""
    import soundfile as sf
    import numpy as np

    db_dir = tmp_path / "training-t"
    db_dir.mkdir()

    csv = db_dir / "REFERENCE.csv"
    csv.write_text("t0001,-1\nt0002,1\n")

    # Create tiny valid .wav files
    for name in ["t0001", "t0002"]:
        data = np.random.randn(2000).astype(np.float32)  # 1 second at 2kHz
        sf.write(str(db_dir / f"{name}.wav"), data, SAMPLE_RATE)

    return csv


# ──────────────────────────────────────────────────────────────────────────────
# parse_reference_csv
# ──────────────────────────────────────────────────────────────────────────────
class TestParseReferenceCsv:
    def test_basic_parsing(self, sample_csv):
        df = parse_reference_csv(sample_csv)
        assert len(df) == 3
        assert list(df.columns) == ["filename", "label", "database", "wav_path"]

    def test_labels_are_valid(self, sample_csv):
        df = parse_reference_csv(sample_csv)
        assert set(df["label"].unique()) == {-1, 1}

    def test_database_key_extracted(self, sample_csv):
        df = parse_reference_csv(sample_csv)
        assert df["database"].iloc[0] == "x"

    def test_wav_paths_constructed(self, sample_csv):
        df = parse_reference_csv(sample_csv)
        for _, row in df.iterrows():
            assert row["wav_path"].endswith(".wav")
            assert row["filename"] in row["wav_path"]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_reference_csv(tmp_path / "nonexistent" / "REFERENCE.csv")

    def test_bad_labels_raise(self, tmp_path):
        db_dir = tmp_path / "training-bad"
        db_dir.mkdir()
        csv = db_dir / "REFERENCE.csv"
        csv.write_text("b0001,0\nb0002,2\n")  # Invalid labels
        with pytest.raises(ValueError, match="Unexpected labels"):
            parse_reference_csv(csv)


# ──────────────────────────────────────────────────────────────────────────────
# build_master_labels
# ──────────────────────────────────────────────────────────────────────────────
class TestBuildMasterLabels:
    def test_merges_multiple_databases(self, tmp_path):
        for name in ["training-a", "training-b"]:
            d = tmp_path / name
            d.mkdir()
            (d / "REFERENCE.csv").write_text("x0001,-1\nx0002,1\n")

        dirs = {
            "a": tmp_path / "training-a",
            "b": tmp_path / "training-b",
        }
        df = build_master_labels(dirs)
        assert len(df) == 4
        assert set(df["database"].unique()) == {"a", "b"}

    def test_has_class_columns(self, tmp_path):
        d = tmp_path / "training-z"
        d.mkdir()
        (d / "REFERENCE.csv").write_text("z0001,-1\nz0002,1\n")
        df = build_master_labels({"z": d})
        assert "class_idx" in df.columns
        assert "class_name" in df.columns
        assert df[df["label"] == -1]["class_idx"].iloc[0] == 0
        assert df[df["label"] == 1]["class_idx"].iloc[0] == 1

    def test_no_csvs_raises(self, tmp_path):
        d = tmp_path / "training-empty"
        d.mkdir()
        with pytest.raises(RuntimeError, match="No REFERENCE.csv"):
            build_master_labels({"empty": d})

    def test_real_dataset_loads(self):
        """Integration test — loads the real dataset."""
        available = {k: v for k, v in TRAINING_DIRS.items() if (v / "REFERENCE.csv").exists()}
        if not available:
            pytest.skip("No training data available")

        df = build_master_labels(available)
        assert len(df) > 0
        assert "class_idx" in df.columns
        assert df["label"].isin([-1, 1]).all()


# ──────────────────────────────────────────────────────────────────────────────
# validate_wav_files
# ──────────────────────────────────────────────────────────────────────────────
class TestValidateWavFiles:
    def test_existing_wav_validated(self, sample_csv_with_wavs):
        df = parse_reference_csv(sample_csv_with_wavs)
        validated = validate_wav_files(df)

        assert "exists" in validated.columns
        assert "readable" in validated.columns
        assert "duration" in validated.columns
        assert "sr" in validated.columns

        assert validated["exists"].all()
        assert validated["readable"].all()
        assert (validated["duration"] > 0).all()
        assert (validated["sr"] == SAMPLE_RATE).all()

    def test_missing_wav_flagged(self, sample_csv):
        df = parse_reference_csv(sample_csv)
        validated = validate_wav_files(df)
        assert not validated["exists"].any()
        assert not validated["readable"].any()

    def test_does_not_mutate_input(self, sample_csv):
        df = parse_reference_csv(sample_csv)
        original_cols = set(df.columns)
        _ = validate_wav_files(df)
        assert set(df.columns) == original_cols  # No mutation


# ──────────────────────────────────────────────────────────────────────────────
# get_dataset_stats
# ──────────────────────────────────────────────────────────────────────────────
class TestGetDatasetStats:
    def test_stats_structure(self, sample_csv_with_wavs):
        df = parse_reference_csv(sample_csv_with_wavs)
        df = validate_wav_files(df)
        stats = get_dataset_stats(df)

        assert "total_recordings" in stats
        assert "per_database" in stats
        assert "class_distribution" in stats
        assert "duration" in stats
        assert stats["total_recordings"] == 2

    def test_class_counts_correct(self, sample_csv_with_wavs):
        df = parse_reference_csv(sample_csv_with_wavs)
        df = validate_wav_files(df)
        stats = get_dataset_stats(df)

        assert stats["class_distribution"]["normal"] == 1
        assert stats["class_distribution"]["abnormal"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# save_master_labels
# ──────────────────────────────────────────────────────────────────────────────
class TestSaveMasterLabels:
    def test_saves_and_reads_back(self, sample_csv_with_wavs, tmp_path):
        df = parse_reference_csv(sample_csv_with_wavs)
        out = tmp_path / "test_labels.csv"
        result_path = save_master_labels(df, out)

        assert result_path.exists()
        loaded = pd.read_csv(result_path)
        assert len(loaded) == len(df)
        assert list(loaded.columns) == list(df.columns)

    def test_creates_parent_dirs(self, sample_csv_with_wavs, tmp_path):
        df = parse_reference_csv(sample_csv_with_wavs)
        out = tmp_path / "deep" / "nested" / "labels.csv"
        result_path = save_master_labels(df, out)
        assert result_path.exists()


# ──────────────────────────────────────────────────────────────────────────────
# Config sanity checks
# ──────────────────────────────────────────────────────────────────────────────
class TestConfig:
    def test_label_map_covers_both(self):
        assert -1 in LABEL_MAP
        assert 1 in LABEL_MAP

    def test_label_names_cover_both(self):
        assert 0 in LABEL_NAMES
        assert 1 in LABEL_NAMES

    def test_sample_rate(self):
        assert SAMPLE_RATE == 2000
