"""Acoustic feature extraction for heart sound classification."""
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import welch

from src.config import SAMPLE_RATE


def band_energy_ratio(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    total_mask = (frequencies >= 20) & (frequencies <= 500)
    total = np.trapezoid(power[total_mask], frequencies[total_mask])
    if total <= 1e-12:
        return 0.0
    return float(np.trapezoid(power[mask], frequencies[mask]) / total)


def extract_acoustic_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract compact PCG features from preprocessed audio."""
    audio = audio.astype(np.float32)
    frequencies, power = welch(audio, fs=sr, nperseg=min(1024, len(audio)))

    features: list[float] = []
    for low, high in [
        (20, 60),
        (60, 100),
        (100, 140),
        (140, 180),
        (180, 240),
        (240, 400),
        (400, 800),
    ]:
        features.append(band_energy_ratio(frequencies, power, low, high))

    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=512, hop_length=128)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, n_fft=512, hop_length=128)[0]
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=512, hop_length=128)[0]
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=512, hop_length=128)[0]
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=512, hop_length=128)[0]
    rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=128)[0]
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=512, hop_length=128)

    for arr in [centroid, bandwidth, rolloff, flatness, zcr, rms]:
        features.extend([
            float(np.mean(arr)),
            float(np.std(arr)),
            float(np.percentile(arr, 25)),
            float(np.percentile(arr, 75)),
        ])

    features.extend(np.mean(mfcc, axis=1).astype(float).tolist())
    features.extend(np.std(mfcc, axis=1).astype(float).tolist())

    duration = len(audio) / float(sr)
    active = float(np.mean(rms > (0.10 * np.max(rms)))) if np.max(rms) > 1e-12 else 0.0
    features.extend([duration, active])

    return np.array(features, dtype=np.float32)
