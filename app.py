"""
Phase 6 — Gradio Web App for Heart Sound Classification
Upload a .wav file → preprocessing → mel spectrogram → CNN inference → result.
"""
import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
import librosa
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    SPEC_HEIGHT, SPEC_WIDTH,
    SEGMENT_SAMPLES, MODELS_DIR, LABEL_NAMES, PROJECT_ROOT,
)
from src.phase2_preprocess import preprocess_audio
from src.phase3_spectrograms import compute_mel_spectrogram, spectrogram_to_image
from src.phase4_train import build_model, get_transforms
from src.phase5_evaluate import load_optimal_threshold
from PIL import Image

logger = logging.getLogger(__name__)

APP_CSS = """
.main-title { text-align: center; margin-bottom: 0.5em; }
.subtitle { text-align: center; color: #aaa; margin-bottom: 1.5em; font-size: 0.95em; }
.result-box { min-height: 120px; display: flex; align-items: center; justify-content: center;
              border: 2px solid #333; border-radius: 12px; padding: 20px; }
.disclaimer { text-align: center; font-size: 0.8em; color: #888; margin-top: 0; }
.how-it-works { font-size: 0.9em; color: #aaa; }
"""

# ─── Global model (loaded once) ──────────────────────────────────────────────
_model = None
_device = None
_transform = None


def get_model():
    """Load model lazily on first call."""
    global _model, _device, _transform

    if _model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Primary: output/models/ (local training); Fallback: models/ (deployment)
        model_path = MODELS_DIR / "best_model.pt"
        if not model_path.exists():
            model_path = PROJECT_ROOT / "models" / "best_model.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found. Expected {MODELS_DIR / 'best_model.pt'} or "
                f"{PROJECT_ROOT / 'models' / 'best_model.pt'}."
            )

        _model = build_model(pretrained=False)
        state_dict = torch.load(str(model_path), map_location=_device, weights_only=True)
        _model.load_state_dict(state_dict)
        _model = _model.to(_device)
        _model.eval()
        _transform = get_transforms(is_train=False)

        logger.info("Model loaded on %s", _device)

    return _model, _device, _transform


def _mel_spectrogram_figure(mel_db: np.ndarray) -> plt.Figure:
    """Create a matplotlib figure of the mel spectrogram for display."""
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(
        mel_db, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest",
    )
    ax.set_xlabel("Time", fontsize=10)
    ax.set_ylabel("Mel Band", fontsize=10)
    ax.set_title("Mel Spectrogram", fontsize=12)
    fig.colorbar(im, ax=ax, label="dB")
    plt.tight_layout()
    return fig


def _waveform_figure(audio: np.ndarray, label: str) -> plt.Figure:
    """Create an ECG-style waveform plot of the audio signal."""
    # Downsample for display (max 3000 points)
    display_audio = audio
    if len(audio) > 3000:
        step = len(audio) // 3000
        display_audio = audio[::step]

    t = np.linspace(0, len(audio) / SAMPLE_RATE, len(display_audio))

    color = "#e74c3c" if label == "ABNORMAL" else "#2ecc71"

    fig, ax = plt.subplots(figsize=(7, 3), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    ax.plot(t, display_audio, color=color, linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Time (s)", fontsize=10, color="white")
    ax.set_ylabel("Amplitude", fontsize=10, color="white")
    ax.set_title("Heart Sound Waveform", fontsize=12, color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color="#222", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    return fig


# Formats librosa/soundfile can read natively (no ffmpeg needed)
_NATIVE_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def _ensure_wav(filepath: str) -> str:
    """Convert non-WAV audio to WAV using ffmpeg for reliable loading.

    Gradio mic recordings are already WAV. Uploaded files may be AMR, OPUS,
    OGG, M4A, etc. — convert them so librosa always gets a WAV.
    """
    ext = Path(filepath).suffix.lower()
    if ext in _NATIVE_EXTS:
        return filepath

    wav_path = tempfile.mktemp(suffix=".wav")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", filepath, "-ar", str(SAMPLE_RATE),
             "-ac", "1", "-f", "wav", wav_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and Path(wav_path).exists():
            logger.info("Converted %s → WAV", ext)
            return wav_path
    except FileNotFoundError:
        logger.warning("ffmpeg not found — falling back to direct librosa load")
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg conversion timed out")

    return filepath  # fall back to original path (librosa may still handle it)


def classify_heart_sound(audio_filepath: str) -> tuple:
    """Classify a heart sound recording as Normal or Abnormal.

    Args:
        audio_filepath: Path to a .wav file (from Gradio upload).

    Returns:
        Tuple of (result_text, waveform_figure, spectrogram_figure).
    """
    if audio_filepath is None:
        return "Please upload a heart sound recording.", None, None

    # Convert any format to WAV via ffmpeg for reliable loading
    audio_filepath = _ensure_wav(audio_filepath)

    try:
        audio, sr = librosa.load(audio_filepath, sr=SAMPLE_RATE, mono=True)
        audio = audio.astype(np.float32)
    except Exception as e:
        logger.error("Failed to load audio: %s", e)
        return f"Error: Could not read audio file. ({e})", None, None

    if len(audio) < SAMPLE_RATE:
        return "Error: Audio too short (need at least 1 second).", None, None

    # Preprocess
    audio = preprocess_audio(audio, sr=SAMPLE_RATE)

    # Take central 3-second segment (or pad if shorter)
    if len(audio) >= SEGMENT_SAMPLES:
        center = len(audio) // 2
        start = center - SEGMENT_SAMPLES // 2
        segment = audio[start : start + SEGMENT_SAMPLES]
    else:
        segment = np.zeros(SEGMENT_SAMPLES, dtype=np.float32)
        segment[: len(audio)] = audio

    # Compute mel spectrogram
    mel_db = compute_mel_spectrogram(segment)
    img = spectrogram_to_image(mel_db)

    # Build spectrogram figure for display
    spec_fig = _mel_spectrogram_figure(mel_db)

    # Convert to PIL, apply transforms
    model, device, transform = get_model()
    pil_img = Image.fromarray(img).convert("RGB")
    tensor = transform(pil_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]

    # If recording is long, also do majority voting over multiple segments
    if len(audio) >= SEGMENT_SAMPLES * 2:
        all_probs = []
        hop = SEGMENT_SAMPLES // 2  # 50% overlap
        for start_idx in range(0, len(audio) - SEGMENT_SAMPLES + 1, hop):
            seg = audio[start_idx : start_idx + SEGMENT_SAMPLES]
            mel = compute_mel_spectrogram(seg)
            im = spectrogram_to_image(mel)
            pil = Image.fromarray(im).convert("RGB")
            t = transform(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(t)
                p = torch.softmax(out, dim=1).cpu().numpy()[0]
            all_probs.append(p)

        probs = np.mean(all_probs, axis=0)

    # Apply optimized threshold (tuned on validation set via Youden's J)
    threshold = load_optimal_threshold()
    abnormal_prob = float(probs[1])
    normal_prob = 1.0 - abnormal_prob
    confidence = max(abnormal_prob, normal_prob) * 100

    if abnormal_prob >= threshold:
        label = "ABNORMAL"
        result_text = f"## \u2757 {label}\nConfidence: **{confidence:.1f}%**"
    else:
        label = "NORMAL"
        result_text = f"## \u2705 {label}\nConfidence: **{confidence:.1f}%**"

    wave_fig = _waveform_figure(audio, label)

    return result_text, wave_fig, spec_fig


def create_app() -> gr.Blocks:
    """Create the Gradio web application.

    Returns:
        Gradio Blocks app ready to launch.
    """
    with gr.Blocks(
        title="Heart Sound Classifier",
    ) as app:
        gr.Markdown(
            "# \U0001FA7A Heart Sound Classifier",
            elem_classes=["main-title"],
        )
        gr.Markdown(
            "AI-powered heart sound analysis using EfficientNet-B0 trained on PhysioNet 2016",
            elem_classes=["subtitle"],
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="\U0001F3B5 Upload or Record Heart Sound",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                analyze_btn = gr.Button(
                    "\U0001F50D Analyze",
                    variant="primary",
                    size="lg",
                )
            with gr.Column(scale=1):
                output_result = gr.Markdown(
                    value="Upload a .wav file and click **Analyze**",
                    elem_classes=["result-box"],
                )

        gr.Markdown("---")

        with gr.Row(equal_height=True):
            with gr.Column():
                wave_plot = gr.Plot(label="\U0001F4C8 Heart Sound Waveform")
            with gr.Column():
                spec_plot = gr.Plot(label="\U0001F308 Mel Spectrogram")

        analyze_btn.click(
            fn=classify_heart_sound,
            inputs=[audio_input],
            outputs=[output_result, wave_plot, spec_plot],
        )

        gr.Markdown("---")
        with gr.Accordion("How it works", open=False):
            gr.Markdown(
                """
                1. **Preprocessing** — Bandpass filter (20-1000 Hz) + wavelet denoising removes noise
                2. **Mel Spectrogram** — Audio converted to 128-band mel spectrogram (128x128 image)
                3. **CNN Classification** — EfficientNet-B0 analyzes the spectrogram
                4. **Multi-segment Voting** — Longer recordings are split into overlapping segments and averaged
                """,
                elem_classes=["how-it-works"],
            )

        gr.Markdown(
            "\u26A0\uFE0F *This is a research tool, not a medical device. Always consult a cardiologist.*",
            elem_classes=["disclaimer"],
        )

    return app


def run_app(share: bool = False, server_port: int = 7860):
    """Launch the Gradio app.

    Args:
        share: If True, creates a public Gradio link.
        server_port: Local port number.
    """
    app = create_app()
    app.launch(
        share=share,
        server_name="0.0.0.0",
        server_port=server_port,
        css=APP_CSS,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_app(share=True)
