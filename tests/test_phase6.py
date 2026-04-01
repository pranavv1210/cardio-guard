"""
Tests for Phase 6 — Gradio App
"""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import soundfile as sf
import torch

from src.config import SAMPLE_RATE, SEGMENT_SAMPLES, LABEL_NAMES
from app import classify_heart_sound, create_app


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def fake_wav(tmp_path):
    """Create a fake 5-second heart sound .wav file."""
    duration = 5.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    # Simulate heart-like signal: low freq oscillation
    audio = (
        0.5 * np.sin(2 * np.pi * 40 * t)
        + 0.3 * np.sin(2 * np.pi * 80 * t)
        + 0.1 * np.random.randn(len(t))
    ).astype(np.float32)

    wav_path = tmp_path / "test_heart.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    return str(wav_path)


@pytest.fixture
def short_wav(tmp_path):
    """Create a too-short .wav file (0.5 seconds)."""
    audio = np.random.randn(int(SAMPLE_RATE * 0.5)).astype(np.float32)
    wav_path = tmp_path / "short.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    return str(wav_path)


@pytest.fixture
def mock_model():
    """Mock the get_model function to avoid loading real weights."""
    from src.phase4_train import build_model, get_transforms

    model = build_model(pretrained=False)
    model.eval()
    device = torch.device("cpu")
    transform = get_transforms(is_train=False)

    with patch("app.get_model", return_value=(model, device, transform)):
        yield


# ──────────────────────────────────────────────────────────────────────────────
# classify_heart_sound
# ──────────────────────────────────────────────────────────────────────────────
class TestClassifyHeartSound:
    def test_returns_tuple_with_text_and_plot(self, fake_wav, mock_model):
        result_text, wave_fig, spec_fig = classify_heart_sound(fake_wav)
        assert isinstance(result_text, str)
        assert "NORMAL" in result_text or "ABNORMAL" in result_text
        assert wave_fig is not None
        assert spec_fig is not None

    def test_spectrogram_is_matplotlib_figure(self, fake_wav, mock_model):
        import matplotlib.pyplot as plt
        _, wave_fig, spec_fig = classify_heart_sound(fake_wav)
        assert isinstance(spec_fig, plt.Figure)
        assert isinstance(wave_fig, plt.Figure)
        plt.close(spec_fig)
        plt.close(wave_fig)

    def test_confidence_shown_in_result(self, fake_wav, mock_model):
        result_text, _, _ = classify_heart_sound(fake_wav)
        assert "%" in result_text

    def test_none_input_returns_message(self, mock_model):
        result_text, wave_fig, spec_fig = classify_heart_sound(None)
        assert "upload" in result_text.lower() or "please" in result_text.lower()
        assert wave_fig is None
        assert spec_fig is None

    def test_short_audio_returns_error(self, short_wav, mock_model):
        result_text, wave_fig, spec_fig = classify_heart_sound(short_wav)
        assert "short" in result_text.lower() or "error" in result_text.lower()
        assert wave_fig is None
        assert spec_fig is None


# ──────────────────────────────────────────────────────────────────────────────
# create_app
# ──────────────────────────────────────────────────────────────────────────────
class TestCreateApp:
    def test_app_creates_without_error(self):
        app = create_app()
        assert app is not None

    def test_app_is_gradio_blocks(self):
        import gradio as gr
        app = create_app()
        assert isinstance(app, gr.Blocks)
