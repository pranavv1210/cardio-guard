"""
Tests for Phase 2 — Audio Preprocessing
"""
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.config import SAMPLE_RATE, SEGMENT_SAMPLES, SEGMENT_HOP
from src.phase2_preprocess import (
    load_wav,
    bandpass_filter,
    wavelet_denoise,
    normalize_amplitude,
    trim_silence,
    preprocess_audio,
    segment_audio,
    process_single_recording,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sine_wave():
    """Generate a clean 100 Hz sine wave (3 seconds at 2kHz)."""
    t = np.linspace(0, 3.0, 3 * SAMPLE_RATE, endpoint=False)
    return np.sin(2 * np.pi * 100 * t).astype(np.float32)


@pytest.fixture
def noisy_signal():
    """Generate a heart-frequency signal with high-freq noise."""
    t = np.linspace(0, 3.0, 3 * SAMPLE_RATE, endpoint=False)
    heart = np.sin(2 * np.pi * 50 * t)  # 50 Hz — in heart range
    noise = 0.3 * np.sin(2 * np.pi * 1500 * t)  # 1500 Hz — above heart range
    return (heart + noise).astype(np.float32)


@pytest.fixture
def long_audio():
    """10 seconds of audio — should produce multiple segments."""
    return np.random.randn(10 * SAMPLE_RATE).astype(np.float32)


@pytest.fixture
def wav_file(tmp_path):
    """Create a temporary .wav file."""
    data = np.random.randn(2 * SAMPLE_RATE).astype(np.float32) * 0.5
    path = tmp_path / "test.wav"
    sf.write(str(path), data, SAMPLE_RATE)
    return str(path)


# ──────────────────────────────────────────────────────────────────────────────
# load_wav
# ──────────────────────────────────────────────────────────────────────────────
class TestLoadWav:
    def test_loads_valid_wav(self, wav_file):
        audio = load_wav(wav_file)
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_correct_sample_count(self, wav_file):
        audio = load_wav(wav_file)
        expected = 2 * SAMPLE_RATE
        assert abs(len(audio) - expected) <= 1  # Allow 1 sample rounding

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_wav("/nonexistent/path/fake.wav")


# ──────────────────────────────────────────────────────────────────────────────
# bandpass_filter
# ──────────────────────────────────────────────────────────────────────────────
class TestBandpassFilter:
    def test_preserves_length(self, sine_wave):
        filtered = bandpass_filter(sine_wave)
        assert len(filtered) == len(sine_wave)

    def test_preserves_in_band_signal(self, sine_wave):
        """100 Hz sine should pass through 20-1000 Hz filter mostly intact."""
        filtered = bandpass_filter(sine_wave)
        correlation = np.corrcoef(sine_wave, filtered)[0, 1]
        assert correlation > 0.9

    def test_attenuates_out_of_band(self):
        """5 Hz signal is below the 20 Hz cutoff and should be attenuated."""
        t = np.linspace(0, 3.0, 3 * SAMPLE_RATE, endpoint=False)
        low_freq = np.sin(2 * np.pi * 5 * t).astype(np.float32)
        filtered = bandpass_filter(low_freq)
        # Sub-cutoff signal energy should drop significantly
        assert np.std(filtered) < 0.3 * np.std(low_freq)

    def test_empty_input(self):
        result = bandpass_filter(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_output_dtype(self, sine_wave):
        filtered = bandpass_filter(sine_wave)
        assert filtered.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# wavelet_denoise
# ──────────────────────────────────────────────────────────────────────────────
class TestWaveletDenoise:
    def test_preserves_length(self, sine_wave):
        denoised = wavelet_denoise(sine_wave)
        assert len(denoised) == len(sine_wave)

    def test_reduces_noise(self):
        """Signal + noise → denoised should have lower noise energy."""
        np.random.seed(42)
        clean = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 2000))
        noisy = clean + 0.5 * np.random.randn(2000)
        denoised = wavelet_denoise(noisy.astype(np.float32))

        noise_before = np.std(noisy - clean)
        noise_after = np.std(denoised - clean)
        assert noise_after < noise_before

    def test_empty_input(self):
        result = wavelet_denoise(np.array([], dtype=np.float32))
        assert len(result) == 0

    def test_output_dtype(self, sine_wave):
        assert wavelet_denoise(sine_wave).dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# normalize_amplitude
# ──────────────────────────────────────────────────────────────────────────────
class TestNormalizeAmplitude:
    def test_peak_is_one(self):
        audio = np.array([0.5, -0.3, 0.8, -0.1], dtype=np.float32)
        normed = normalize_amplitude(audio)
        assert np.isclose(np.max(np.abs(normed)), 1.0)

    def test_silent_returns_zeros(self):
        audio = np.zeros(100, dtype=np.float32)
        normed = normalize_amplitude(audio)
        assert np.all(normed == 0)

    def test_preserves_shape(self, sine_wave):
        normed = normalize_amplitude(sine_wave)
        assert normed.shape == sine_wave.shape

    def test_output_dtype(self):
        normed = normalize_amplitude(np.array([1.0, 2.0], dtype=np.float32))
        assert normed.dtype == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# trim_silence
# ──────────────────────────────────────────────────────────────────────────────
class TestTrimSilence:
    def test_removes_leading_silence(self):
        silence = np.zeros(4000, dtype=np.float32)
        signal = np.random.randn(4000).astype(np.float32)
        audio = np.concatenate([silence, signal])
        trimmed = trim_silence(audio)
        assert len(trimmed) < len(audio)

    def test_does_not_over_trim(self, sine_wave):
        """A full sine wave should not be significantly trimmed."""
        trimmed = trim_silence(sine_wave)
        assert len(trimmed) >= SAMPLE_RATE  # At least 1 second remains

    def test_empty_input(self):
        result = trim_silence(np.array([], dtype=np.float32))
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────────────────
# preprocess_audio
# ──────────────────────────────────────────────────────────────────────────────
class TestPreprocessAudio:
    def test_full_pipeline_runs(self, sine_wave):
        result = preprocess_audio(sine_wave)
        assert isinstance(result, np.ndarray)
        assert len(result) > 0

    def test_output_bounded(self, sine_wave):
        result = preprocess_audio(sine_wave)
        assert np.max(np.abs(result)) <= 1.0 + 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# segment_audio
# ──────────────────────────────────────────────────────────────────────────────
class TestSegmentAudio:
    def test_correct_segment_length(self, long_audio):
        segments = segment_audio(long_audio)
        for seg in segments:
            assert len(seg) == SEGMENT_SAMPLES

    def test_segment_count(self):
        """10s audio → 3s windows, 50% overlap → expect ~6 segments."""
        audio = np.random.randn(10 * SAMPLE_RATE).astype(np.float32)
        segments = segment_audio(audio)
        # With 3s window, 1.5s hop over 10s: ceil((10-3)/1.5) + 1 ≈ 6
        assert 4 <= len(segments) <= 8

    def test_short_audio_padded(self):
        """Audio shorter than one segment should still produce one padded segment."""
        short = np.random.randn(1000).astype(np.float32)
        segments = segment_audio(short)
        assert len(segments) >= 1
        assert len(segments[0]) == SEGMENT_SAMPLES

    def test_empty_audio(self):
        segments = segment_audio(np.array([], dtype=np.float32))
        assert segments == []

    def test_overlap_works(self, long_audio):
        """Adjacent segments should overlap by 50%."""
        segments = segment_audio(long_audio)
        if len(segments) >= 2:
            # Second half of seg 0 should match first half of seg 1
            half = SEGMENT_SAMPLES // 2
            overlap = np.array_equal(segments[0][half:], segments[1][:half])
            assert overlap


# ──────────────────────────────────────────────────────────────────────────────
# process_single_recording
# ──────────────────────────────────────────────────────────────────────────────
class TestProcessSingleRecording:
    def test_produces_segments(self, tmp_path):
        # Create a test .wav
        audio = np.random.randn(5 * SAMPLE_RATE).astype(np.float32) * 0.5
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        out_dir = tmp_path / "segs"
        out_dir.mkdir()

        meta = process_single_recording(
            wav_path=str(wav_path),
            label=-1,
            filename="test",
            output_dir=out_dir,
        )

        assert len(meta) > 0
        assert all(m["class_idx"] == 0 for m in meta)
        assert all(m["source_file"] == "test" for m in meta)

        # Check .npy files exist
        for m in meta:
            assert Path(m["segment_path"]).exists()

    def test_abnormal_label_mapping(self, tmp_path):
        audio = np.random.randn(3 * SAMPLE_RATE).astype(np.float32) * 0.5
        wav_path = tmp_path / "abn.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        out_dir = tmp_path / "segs"
        out_dir.mkdir()

        meta = process_single_recording(
            wav_path=str(wav_path),
            label=1,
            filename="abn",
            output_dir=out_dir,
        )
        assert all(m["class_idx"] == 1 for m in meta)
