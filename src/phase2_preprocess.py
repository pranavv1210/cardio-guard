"""
Phase 2 — Audio Preprocessing
Bandpass filter, wavelet denoising, normalization, silence trimming, segmentation.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, sosfiltfilt
import pywt

from src.config import (
    SAMPLE_RATE, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER,
    WAVELET_NAME, WAVELET_LEVEL,
    SEGMENT_DURATION, SEGMENT_SAMPLES, SEGMENT_HOP,
    SEGMENTS_DIR, OUTPUT_DIR, LABEL_MAP,
)

logger = logging.getLogger(__name__)


def load_wav(wav_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a .wav file and resample if needed.

    Args:
        wav_path: Path to .wav file.
        target_sr: Target sample rate.

    Returns:
        1D numpy array of audio samples (float32).

    Raises:
        FileNotFoundError: If file doesn't exist.
        RuntimeError: If file can't be read.
    """
    path = Path(wav_path)
    if not path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    try:
        audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load {wav_path}: {e}") from e

    return audio.astype(np.float32)


def bandpass_filter(
    audio: np.ndarray,
    low: int = BANDPASS_LOW,
    high: int = BANDPASS_HIGH,
    sr: int = SAMPLE_RATE,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to isolate heart sounds.

    Args:
        audio: 1D audio signal.
        low: Low cutoff frequency (Hz).
        high: High cutoff frequency (Hz).
        sr: Sample rate (Hz).
        order: Filter order.

    Returns:
        Filtered audio signal (same length).
    """
    if len(audio) == 0:
        return audio

    nyquist = sr / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist

    # Clamp to valid range
    low_norm = max(low_norm, 1e-5)
    high_norm = min(high_norm, 1.0 - 1e-5)

    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")
    filtered = sosfiltfilt(sos, audio).astype(np.float32)
    return filtered


def wavelet_denoise(
    audio: np.ndarray,
    wavelet: str = WAVELET_NAME,
    level: int = WAVELET_LEVEL,
) -> np.ndarray:
    """Remove high-frequency noise using wavelet soft thresholding.

    Args:
        audio: 1D audio signal.
        wavelet: Wavelet family name.
        level: Decomposition level.

    Returns:
        Denoised audio signal.
    """
    if len(audio) == 0:
        return audio

    # Decompose
    coeffs = pywt.wavedec(audio, wavelet, level=level)

    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(audio)))

    # Soft-threshold detail coefficients (keep approximation intact)
    denoised_coeffs = [coeffs[0]]  # approximation — untouched
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, threshold, mode="soft"))

    # Reconstruct
    reconstructed = pywt.waverec(denoised_coeffs, wavelet)

    # waverec can add one extra sample due to rounding
    reconstructed = reconstructed[: len(audio)]

    return reconstructed.astype(np.float32)


def normalize_amplitude(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range by peak amplitude.

    Args:
        audio: 1D audio signal.

    Returns:
        Normalized audio. Returns zeros if input is silent.
    """
    peak = np.max(np.abs(audio))
    if peak < 1e-10:
        return np.zeros_like(audio)
    return (audio / peak).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    top_db: float = 25.0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Trim leading/trailing silence based on energy threshold.

    Args:
        audio: 1D audio signal.
        top_db: Threshold in dB below peak to consider as silence.
        sr: Sample rate.

    Returns:
        Trimmed audio. Returns original if trimming would remove everything.
    """
    if len(audio) == 0:
        return audio

    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)

    # Safety: don't return empty audio
    if len(trimmed) < sr:  # Less than 1 second
        return audio

    return trimmed


def preprocess_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Full preprocessing pipeline for a single audio recording.

    Pipeline: bandpass → wavelet denoise → normalize → trim silence.

    Args:
        audio: Raw 1D audio signal.
        sr: Sample rate.

    Returns:
        Preprocessed audio signal.
    """
    audio = bandpass_filter(audio, sr=sr)
    audio = wavelet_denoise(audio)
    audio = normalize_amplitude(audio)
    audio = trim_silence(audio, sr=sr)
    return audio


def segment_audio(
    audio: np.ndarray,
    segment_samples: int = SEGMENT_SAMPLES,
    hop_samples: int = SEGMENT_HOP,
) -> list[np.ndarray]:
    """Cut audio into fixed-length overlapping windows.

    Args:
        audio: 1D preprocessed audio.
        segment_samples: Samples per segment (default: 3s * 2000Hz = 6000).
        hop_samples: Hop between segment starts (default: 50% overlap = 3000).

    Returns:
        List of segments. Each is zero-padded if needed.
    """
    if len(audio) == 0:
        return []

    segments = []
    start = 0
    while start < len(audio):
        end = start + segment_samples
        segment = audio[start:end]

        # Zero-pad the last segment if shorter
        if len(segment) < segment_samples:
            padded = np.zeros(segment_samples, dtype=np.float32)
            padded[: len(segment)] = segment
            segment = padded

        segments.append(segment)
        start += hop_samples

    return segments


def process_single_recording(
    wav_path: str,
    label: int,
    filename: str,
    output_dir: Path,
) -> list[dict]:
    """Process one recording: load → preprocess → segment → save.

    Args:
        wav_path: Path to .wav file.
        label: Original label (-1 or 1).
        filename: Recording filename (e.g., 'a0001').
        output_dir: Directory to save segments.

    Returns:
        List of dicts with segment metadata.
    """
    class_idx = LABEL_MAP[label]

    audio = load_wav(wav_path)
    audio = preprocess_audio(audio)
    segments = segment_audio(audio)

    metadata = []
    for i, seg in enumerate(segments):
        seg_name = f"{filename}_seg{i:04d}"
        seg_path = output_dir / f"{seg_name}.npy"
        np.save(str(seg_path), seg)
        metadata.append({
            "segment_name": seg_name,
            "source_file": filename,
            "segment_idx": i,
            "class_idx": class_idx,
            "label": label,
            "segment_path": str(seg_path),
        })

    return metadata


def run_phase2(master_csv: Optional[str] = None) -> pd.DataFrame:
    """Execute the full Phase 2 preprocessing pipeline.

    Args:
        master_csv: Path to master_labels.csv. Auto-detected if None.

    Returns:
        DataFrame of all segment metadata.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=" * 60)
    logger.info("PHASE 2 — Audio Preprocessing")
    logger.info("=" * 60)

    # Load master labels
    if master_csv is None:
        master_csv = str(OUTPUT_DIR / "master_labels.csv")

    master = pd.read_csv(master_csv)
    logger.info("Loaded %d recordings from master labels", len(master))

    # Filter to readable files only
    if "readable" in master.columns:
        master = master[master["readable"] == True].reset_index(drop=True)
        logger.info("Readable recordings: %d", len(master))

    # Create output directory
    SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metadata = []
    total = len(master)

    for idx, row in master.iterrows():
        try:
            meta = process_single_recording(
                wav_path=row["wav_path"],
                label=row["label"],
                filename=row["filename"],
                output_dir=SEGMENTS_DIR,
            )
            all_metadata.extend(meta)
        except Exception as e:
            logger.warning("Error processing %s: %s", row["filename"], e)

        if (idx + 1) % 200 == 0 or idx == total - 1:
            logger.info("Processed %d/%d recordings → %d segments so far",
                        idx + 1, total, len(all_metadata))

    # Save segment metadata
    seg_df = pd.DataFrame(all_metadata)
    seg_csv = OUTPUT_DIR / "segments_metadata.csv"
    seg_df.to_csv(seg_csv, index=False)
    logger.info("Saved %d segments metadata to %s", len(seg_df), seg_csv)

    # Print stats
    normal = (seg_df["class_idx"] == 0).sum()
    abnormal = (seg_df["class_idx"] == 1).sum()
    logger.info("Segments: %d normal / %d abnormal / %d total",
                normal, abnormal, len(seg_df))

    return seg_df


if __name__ == "__main__":
    run_phase2()
