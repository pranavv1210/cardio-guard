"""
Heart Sound CNN Classifier — Project Configuration
All paths, constants, and hyperparameters in one place.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ → project root
DATA_ROOT = PROJECT_ROOT  # training-a, training-b, etc. live here

TRAINING_DIRS = {
    "a": DATA_ROOT / "training-a",
    "b": DATA_ROOT / "training-b",
    "c": DATA_ROOT / "training-c",
    "d": DATA_ROOT / "training-d",
    "e": DATA_ROOT / "training-e",
    "f": DATA_ROOT / "training-f",
}

VALIDATION_DIR = DATA_ROOT / "validation"

# Output directories (created during processing)
OUTPUT_DIR = PROJECT_ROOT / "output"
SEGMENTS_DIR = OUTPUT_DIR / "segments"
SPECTROGRAMS_DIR = OUTPUT_DIR / "spectrograms"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# ─── Audio Constants ─────────────────────────────────────────────────────────
SAMPLE_RATE = 2000          # All PhysioNet 2016 files are 2000 Hz
SEGMENT_DURATION = 3.0      # seconds per window
SEGMENT_OVERLAP = 0.5       # 50% overlap between windows
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 6000 samples
SEGMENT_HOP = int(SEGMENT_SAMPLES * (1 - SEGMENT_OVERLAP))  # 3000 samples

# ─── Bandpass Filter ─────────────────────────────────────────────────────────
BANDPASS_LOW = 20           # Hz — below this is DC drift / room rumble
BANDPASS_HIGH = 1000        # Hz — heart sounds live in 20-1000 Hz
BANDPASS_ORDER = 4          # Butterworth filter order

# ─── Wavelet Denoising ──────────────────────────────────────────────────────
WAVELET_NAME = "db4"        # Daubechies-4 wavelet
WAVELET_LEVEL = 4           # Decomposition levels

# ─── Mel Spectrogram ────────────────────────────────────────────────────────
N_MELS = 128                # Number of mel bands
N_FFT = 512                 # FFT window — smaller for 2kHz SR (not 2048)
HOP_LENGTH = 128            # Hop length for spectrogram
SPEC_HEIGHT = 128           # Output image height
SPEC_WIDTH = 128            # Output image width

# ─── Dataset Split ───────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ─── Training Hyperparameters ────────────────────────────────────────────────
BATCH_SIZE = 16             # Safe for GTX 1650 (4GB VRAM)
NUM_WORKERS = 4             # DataLoader workers
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True
DROPOUT_RATE = 0.5
NUM_CLASSES = 2             # normal vs abnormal

# Stage 1: frozen backbone
STAGE1_EPOCHS = 10
STAGE1_LR = 1e-3

# Stage 2: fine-tune last blocks
STAGE2_EPOCHS = 15
STAGE2_LR = 1e-5

# ─── SpecAugment ─────────────────────────────────────────────────────────────
SPEC_AUG_FREQ_MASK = 15     # Max frequency bands to mask
SPEC_AUG_TIME_MASK = 20     # Max time steps to mask

# ─── Label Mapping ───────────────────────────────────────────────────────────
# PhysioNet 2016: -1 = normal, 1 = abnormal
LABEL_MAP = {-1: 0, 1: 1}   # Map to 0/1 for training
LABEL_NAMES = {0: "Normal", 1: "Abnormal"}
