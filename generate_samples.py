import os
import numpy as np
import soundfile as sf
from pathlib import Path
from src.config import SAMPLE_RATE

def create_heart_beat(abnormal=False):
    sr = SAMPLE_RATE
    beat_duration = 0.8  # 75 BPM
    t = np.linspace(0, beat_duration, int(sr * beat_duration), endpoint=False)
    signal = np.zeros_like(t)
    
    # S1: Lub (frequency ~60Hz, duration 150ms)
    s1_len = int(sr * 0.15)
    t_s1 = np.linspace(0, 0.15, s1_len, endpoint=False)
    env_s1 = np.hanning(s1_len)
    s1 = np.sin(2 * np.pi * 60 * t_s1) * env_s1
    signal[:s1_len] = s1
    
    # S2: Dub (frequency ~100Hz, duration 120ms, starts at 250ms)
    s2_start = int(sr * 0.25)
    s2_len = int(sr * 0.12)
    t_s2 = np.linspace(0, 0.12, s2_len, endpoint=False)
    env_s2 = np.hanning(s2_len)
    s2 = np.sin(2 * np.pi * 100 * t_s2) * env_s2
    signal[s2_start:s2_start+s2_len] = s2
    
    # Murmur (systolic murmur between S1 and S2)
    if abnormal:
        murmur_start = int(sr * 0.12)
        murmur_end = int(sr * 0.25)
        murmur_len = murmur_end - murmur_start
        t_murmur = np.linspace(0, murmur_len/sr, murmur_len, endpoint=False)
        noise = np.random.randn(murmur_len)
        murmur = 0.25 * np.sin(2 * np.pi * 150 * t_murmur) * np.hanning(murmur_len) * (0.8 + 0.2 * noise)
        signal[murmur_start:murmur_end] += murmur
        
    return signal

def main():
    duration = 5.0  # 5 seconds
    sr = SAMPLE_RATE
    num_beats = int(np.ceil(duration / 0.8))
    
    # Generate Normal Heart Sound
    normal_signal = []
    for _ in range(num_beats):
        normal_signal.extend(create_heart_beat(abnormal=False))
    normal_signal = np.array(normal_signal[:int(sr * duration)])
    normal_signal += 0.02 * np.random.randn(len(normal_signal))
    normal_signal = normal_signal / np.max(np.abs(normal_signal)) * 0.5
    
    # Generate Abnormal Heart Sound
    abnormal_signal = []
    for _ in range(num_beats):
        abnormal_signal.extend(create_heart_beat(abnormal=True))
    abnormal_signal = np.array(abnormal_signal[:int(sr * duration)])
    abnormal_signal += 0.02 * np.random.randn(len(abnormal_signal))
    abnormal_signal = abnormal_signal / np.max(np.abs(abnormal_signal)) * 0.5
    
    # Save files
    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    path_normal = samples_dir / "normal_heart_sound.wav"
    sf.write(str(path_normal), normal_signal.astype(np.float32), sr)
    print(f"Saved normal heart sound to {path_normal}")
    
    path_abnormal = samples_dir / "abnormal_heart_sound.wav"
    sf.write(str(path_abnormal), abnormal_signal.astype(np.float32), sr)
    print(f"Saved abnormal heart sound to {path_abnormal}")

if __name__ == "__main__":
    main()
