"""
Streamlit mobile app for heart sound classification.

Inference contract:
- class index 0 = Normal
- class index 1 = Abnormal
- decision uses abnormal probability >= saved optimal threshold
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    MODELS_DIR,
    PROJECT_ROOT,
)
from src.phase2_preprocess import preprocess_audio
from src.phase3_spectrograms import compute_mel_spectrogram, spectrogram_to_image
from src.phase4_train import build_model, get_transforms
from src.phase5_evaluate import load_optimal_threshold

logger = logging.getLogger(__name__)

_model = None
_device = None
_transform = None

_NATIVE_EXTS = {".wav", ".flac", ".ogg", ".mp3"}
MIN_BINARY_CONFIDENCE = 0.60


@dataclass(frozen=True)
class PredictionResult:
    label: str
    normal_probability: float
    abnormal_probability: float
    confidence: float
    threshold: float
    waveform_figure: plt.Figure
    spectrogram_figure: plt.Figure
    status_text: str


def get_model():
    """Load the trained model lazily once per process."""
    global _model, _device, _transform

    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = MODELS_DIR / "best_model.pt"
        if not model_path.exists():
            model_path = PROJECT_ROOT / "models" / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODELS_DIR / 'best_model.pt'} or "
                f"{PROJECT_ROOT / 'models' / 'best_model.pt'}."
            )

        _model = build_model(pretrained=False)
        state_dict = torch.load(str(model_path), map_location=_device, weights_only=True)
        _model.load_state_dict(state_dict)
        _model = _model.to(_device)
        _model.eval()
        _transform = get_transforms(is_train=False)
        logger.info("Model loaded from %s on %s", model_path, _device)

    return _model, _device, _transform


def _ensure_wav(filepath: str) -> str:
    """Convert uncommon audio formats to WAV when ffmpeg is available."""
    ext = Path(filepath).suffix.lower()
    if ext in _NATIVE_EXTS:
        return filepath

    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                filepath,
                "-ar",
                str(SAMPLE_RATE),
                "-ac",
                "1",
                "-f",
                "wav",
                wav_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and Path(wav_path).exists():
            return wav_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("ffmpeg conversion unavailable; trying direct audio load")

    return filepath


def _mel_spectrogram_figure(mel_db: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3))
    im = ax.imshow(
        mel_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        interpolation="nearest",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel band")
    ax.set_title("Mel spectrogram")
    fig.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    return fig


def _waveform_figure(audio: np.ndarray, label: str) -> plt.Figure:
    display_audio = audio
    if len(audio) > 3000:
        step = max(1, len(audio) // 3000)
        display_audio = audio[::step]

    t = np.linspace(0, len(audio) / SAMPLE_RATE, len(display_audio))
    color = "#d83b3b" if label == "ABNORMAL" else "#1f9d55"

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#ffffff")
    ax.plot(t, display_audio, color=color, linewidth=0.9)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Heart sound waveform")
    ax.grid(True, color="#e7e7e7", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _make_segment(audio: np.ndarray) -> np.ndarray:
    if len(audio) >= SEGMENT_SAMPLES:
        center = len(audio) // 2
        start = max(0, center - SEGMENT_SAMPLES // 2)
        return audio[start : start + SEGMENT_SAMPLES]

    segment = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
    segment[: len(audio)] = audio
    return segment


def _predict_segment(segment: np.ndarray) -> np.ndarray:
    mel_db = compute_mel_spectrogram(segment)
    img = spectrogram_to_image(mel_db)
    model, device, transform = get_model()
    pil_img = Image.fromarray(img).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        return torch.softmax(output, dim=1).cpu().numpy()[0]


def _predict_probabilities(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return averaged class probabilities and the display spectrogram."""
    center_segment = _make_segment(audio)
    display_mel = compute_mel_spectrogram(center_segment)

    if len(audio) < SEGMENT_SAMPLES * 2:
        return _predict_segment(center_segment), display_mel

    probs = []
    hop = SEGMENT_SAMPLES // 2
    for start_idx in range(0, len(audio) - SEGMENT_SAMPLES + 1, hop):
        probs.append(_predict_segment(audio[start_idx : start_idx + SEGMENT_SAMPLES]))

    return np.mean(probs, axis=0), display_mel


def predict_heart_sound(audio_filepath: str | None) -> PredictionResult:
    """Classify a heart sound recording as Normal or Abnormal."""
    if audio_filepath is None:
        raise ValueError("Please upload or record a heart sound recording.")

    audio_filepath = _ensure_wav(audio_filepath)
    try:
        audio, _ = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
    except Exception as exc:
        raise ValueError(f"Could not read audio file: {exc}") from exc

    if len(audio) < SAMPLE_RATE:
        raise ValueError("Audio is too short. Please provide at least 1 second.")

    audio = preprocess_audio(audio, sr=SAMPLE_RATE)
    if len(audio) < SAMPLE_RATE:
        raise ValueError("Usable heart sound is too short after noise/silence removal.")

    probs, display_mel = _predict_probabilities(audio)
    threshold = load_optimal_threshold()
    abnormal_probability = float(probs[1])
    normal_probability = float(probs[0])

    binary_label = "ABNORMAL" if abnormal_probability >= threshold else "NORMAL"
    confidence = abnormal_probability if binary_label == "ABNORMAL" else normal_probability
    label = binary_label if confidence >= MIN_BINARY_CONFIDENCE else "UNCERTAIN"
    status_text = (
        f"{label} | confidence {confidence * 100:.1f}% | "
        f"normal {normal_probability * 100:.1f}% | "
        f"abnormal {abnormal_probability * 100:.1f}% | "
        f"threshold {threshold:.3f}"
    )

    return PredictionResult(
        label=label,
        normal_probability=normal_probability,
        abnormal_probability=abnormal_probability,
        confidence=confidence,
        threshold=threshold,
        waveform_figure=_waveform_figure(audio, label),
        spectrogram_figure=_mel_spectrogram_figure(display_mel),
        status_text=status_text,
    )


def classify_heart_sound(audio_filepath: str | None) -> tuple[str, plt.Figure | None, plt.Figure | None]:
    """Backward-compatible wrapper used by tests and simple scripts."""
    try:
        result = predict_heart_sound(audio_filepath)
    except ValueError as exc:
        return f"Error: {exc}", None, None

    icon = "!" if result.label == "ABNORMAL" else "OK" if result.label == "NORMAL" else "?"
    text = (
        f"## {icon} {result.label}\n"
        f"Confidence: **{result.confidence * 100:.1f}%**\n\n"
        f"Normal: **{result.normal_probability * 100:.1f}%**  \n"
        f"Abnormal: **{result.abnormal_probability * 100:.1f}%**"
    )
    return text, result.waveform_figure, result.spectrogram_figure


def _write_uploaded_audio(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def render_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="CardioGuard",
        page_icon="heart",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 680px;
            padding: 1rem 0.85rem 2rem;
        }
        h1 {
            font-size: 1.85rem !important;
            line-height: 1.15 !important;
            margin-bottom: 0.2rem !important;
        }
        .cg-subtitle {
            color: #5f6368;
            font-size: 0.95rem;
            margin-bottom: 1rem;
        }
        .result-card {
            border: 1px solid #e4e7ec;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
        }
        .result-normal {
            border-left: 6px solid #1f9d55;
        }
        .result-abnormal {
            border-left: 6px solid #d83b3b;
        }
        .result-uncertain {
            border-left: 6px solid #b7791f;
        }
        .result-label {
            font-size: 1.55rem;
            font-weight: 800;
            margin-bottom: 0.35rem;
        }
        .metric-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.65rem;
            margin-top: 0.85rem;
        }
        .metric-box {
            border: 1px solid #edf0f3;
            border-radius: 8px;
            padding: 0.75rem;
            background: #fafafa;
        }
        .metric-box span {
            display: block;
            color: #667085;
            font-size: 0.78rem;
        }
        .metric-box strong {
            display: block;
            font-size: 1.2rem;
            margin-top: 0.15rem;
        }
        .disclaimer {
            color: #667085;
            font-size: 0.82rem;
            line-height: 1.35;
            margin-top: 1rem;
        }
        @media (max-width: 480px) {
            .metric-row { grid-template-columns: 1fr; }
            h1 { font-size: 1.6rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("CardioGuard")
    st.markdown(
        "<div class='cg-subtitle'>Mobile-first heart sound screening for Normal vs Abnormal recordings.</div>",
        unsafe_allow_html=True,
    )

    tab_upload, tab_record = st.tabs(["Upload", "Record"])
    audio_path = None

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Heart sound audio",
            type=["wav", "flac", "ogg", "mp3", "m4a", "opus", "amr"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            st.audio(uploaded_file)
            audio_path = _write_uploaded_audio(uploaded_file)

    with tab_record:
        audio_input = getattr(st, "audio_input", None) or getattr(st, "experimental_audio_input", None)
        if audio_input is None:
            st.info("Recording requires a newer Streamlit version. Upload audio instead.")
        else:
            recording = audio_input("Record heart sound", label_visibility="collapsed")
            if recording is not None:
                st.audio(recording)
                audio_path = _write_uploaded_audio(recording)

    analyze = st.button("Analyze", type="primary", use_container_width=True)

    if analyze:
        if audio_path is None:
            st.error("Upload or record at least 1 second of heart sound audio.")
            return

        with st.spinner("Analyzing heart sound..."):
            try:
                result = predict_heart_sound(audio_path)
            except Exception as exc:
                st.error(str(exc))
                return

        if result.label == "ABNORMAL":
            result_class = "result-abnormal"
        elif result.label == "NORMAL":
            result_class = "result-normal"
        else:
            result_class = "result-uncertain"
        st.markdown(
            f"""
            <div class="result-card {result_class}">
                <div class="result-label">{result.label}</div>
                <div>Confidence: <strong>{result.confidence * 100:.1f}%</strong></div>
                <div class="metric-row">
                    <div class="metric-box"><span>Normal probability</span><strong>{result.normal_probability * 100:.1f}%</strong></div>
                    <div class="metric-box"><span>Abnormal probability</span><strong>{result.abnormal_probability * 100:.1f}%</strong></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.pyplot(result.waveform_figure, clear_figure=True, use_container_width=True)
        st.pyplot(result.spectrogram_figure, clear_figure=True, use_container_width=True)

    st.markdown(
        "<div class='disclaimer'>Research prototype only. This is not a medical device and must not be used as a clinical diagnosis.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    render_streamlit_app()
