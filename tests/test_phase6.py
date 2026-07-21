"""
Tests for Phase 6 - Streamlit app and inference wrapper.
"""
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import soundfile as sf
import torch

from app import classify_heart_sound, predict_heart_sound, render_streamlit_app
from src.config import SAMPLE_RATE


@pytest.fixture
def fake_wav(tmp_path):
    duration = 5.0
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
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
    audio = np.random.randn(int(SAMPLE_RATE * 0.5)).astype(np.float32)
    wav_path = tmp_path / "short.wav"
    sf.write(str(wav_path), audio, SAMPLE_RATE)
    return str(wav_path)


@pytest.fixture
def mock_model():
    from src.phase4_train import build_model, get_transforms

    model = build_model(pretrained=False)
    model.eval()
    device = torch.device("cpu")
    transform = get_transforms(is_train=False)

    with patch("app.get_model", return_value=(model, device, transform)):
        yield


class TestPredictHeartSound:
    def test_returns_prediction_result(self, fake_wav, mock_model):
        result = predict_heart_sound(fake_wav)
        assert result.label in {"NORMAL", "ABNORMAL"}
        assert 0.0 <= result.normal_probability <= 1.0
        assert 0.0 <= result.abnormal_probability <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.waveform_figure, plt.Figure)
        assert isinstance(result.spectrogram_figure, plt.Figure)
        plt.close(result.waveform_figure)
        plt.close(result.spectrogram_figure)

    def test_low_abnormal_probability_is_normal(self, fake_wav):
        with (
            patch("app._predict_probabilities", return_value=(np.array([0.91, 0.09]), np.zeros((128, 128)))),
            patch("app.load_optimal_threshold", return_value=0.74),
        ):
            result = predict_heart_sound(fake_wav)

        assert result.label == "NORMAL"
        assert result.normal_probability == pytest.approx(0.91)
        assert result.abnormal_probability == pytest.approx(0.09)

    def test_high_abnormal_probability_is_abnormal(self, fake_wav):
        with (
            patch("app._predict_probabilities", return_value=(np.array([0.08, 0.92]), np.zeros((128, 128)))),
            patch("app.load_optimal_threshold", return_value=0.74),
        ):
            result = predict_heart_sound(fake_wav)

        assert result.label == "ABNORMAL"
        assert result.normal_probability == pytest.approx(0.08)
        assert result.abnormal_probability == pytest.approx(0.92)

    def test_low_confidence_prediction_still_returns_binary_label(self, fake_wav):
        with (
            patch("app._predict_probabilities", return_value=(np.array([0.52, 0.48]), np.zeros((128, 128)))),
            patch("app.load_optimal_threshold", return_value=0.74),
        ):
            result = predict_heart_sound(fake_wav)

        assert result.label == "NORMAL"

    def test_none_input_raises_clear_error(self):
        with pytest.raises(ValueError, match="upload or record"):
            predict_heart_sound(None)

    def test_short_audio_raises_clear_error(self, short_wav):
        with pytest.raises(ValueError, match="too short"):
            predict_heart_sound(short_wav)


class TestClassifyHeartSound:
    def test_returns_tuple_with_text_and_plots(self, fake_wav, mock_model):
        result_text, wave_fig, spec_fig = classify_heart_sound(fake_wav)
        assert "NORMAL" in result_text or "ABNORMAL" in result_text
        assert "%" in result_text
        assert wave_fig is not None
        assert spec_fig is not None
        plt.close(wave_fig)
        plt.close(spec_fig)

    def test_none_input_returns_error_tuple(self):
        result_text, wave_fig, spec_fig = classify_heart_sound(None)
        assert "upload" in result_text.lower() or "record" in result_text.lower()
        assert wave_fig is None
        assert spec_fig is None

    def test_short_audio_returns_error_tuple(self, short_wav):
        result_text, wave_fig, spec_fig = classify_heart_sound(short_wav)
        assert "short" in result_text.lower() or "error" in result_text.lower()
        assert wave_fig is None
        assert spec_fig is None


class TestStreamlitApp:
    def test_render_function_exists(self):
        assert callable(render_streamlit_app)
