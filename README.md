<div align="center">

# 🩺 CardioGuard — Heart Sound Classifier

**AI-powered heart sound analysis using EfficientNet-B0 CNN**  
Trained on the [PhysioNet/CinC 2016 Challenge](https://physionet.org/content/challenge-2016/1.0.0/) dataset.

Upload or record a heart sound — the model classifies it as **Normal** or **Abnormal** and visualizes the waveform and mel spectrogram.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-6.0+-orange?logo=gradle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
[![PhysioNet](https://img.shields.io/badge/Dataset-PhysioNet%202016-8A2BE2)](https://physionet.org/content/challenge-2016/1.0.0/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Model Performance](#-model-performance)
- [Pipeline](#-pipeline)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Utilities](#-utilities)
- [Testing](#-testing)
- [Deployment](#-deployment)
- [Technologies Used](#-technologies-used)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🔍 Overview

CardioGuard is a deep learning system that classifies heart sound (phonocardiogram) recordings as **Normal** or **Abnormal**. The system:

- ✅ Uses **EfficientNet-B0** with transfer learning for spectrogram classification
- ✅ Applies **wavelet denoising** and **bandpass filtering** for robust preprocessing
- ✅ Implements **SpecAugment** data augmentation during training
- ✅ Uses **multi-segment majority voting** for robust inference on long recordings
- ✅ Provides an intuitive **Gradio web interface** for upload or live microphone recording
- ✅ Displays **waveform** and **mel spectrogram** visualizations alongside classification results

> **Use case:** Quick, AI-assisted screening of heart sound recordings to flag potential abnormalities.

---

## 📊 Model Performance

The model was evaluated on a held-out test set (15% of the data) with an **optimized decision threshold** (0.74) determined via **Youden's J statistic**.

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **81.10%** |
| **Sensitivity** (Recall for Abnormal) | **95.53%** |
| **Specificity** | **76.71%** |
| **AUC-ROC** | **92.08%** |

The high sensitivity ensures that nearly all abnormal cases are flagged, making this suitable as a screening tool.

---

## 🔬 Pipeline

The system processes heart sounds through a 5-phase pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 1: Data Inventory                                            │
│  Scan and organize PhysioNet recordings from training sets a-f      │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 2: Preprocessing                                             │
│  • Bandpass filter (20–1000 Hz) — removes DC drift & high-freq noise│
│  • Daubechies-4 wavelet denoising — preserves heart sound transients│
│  • Segmentation into 3-second windows with 50% overlap              │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 3: Mel Spectrograms                                          │
│  • 128-band mel spectrogram → 128×128 image                         │
│  • Log-scaled magnitude for improved dynamic range                  │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 4: Training                                                  │
│  • Two-stage EfficientNet-B0 transfer learning                      │
│  • Stage 1: Frozen backbone (10 epochs, LR=1e-3)                   │
│  • Stage 2: Fine-tune last blocks (15 epochs, LR=1e-5)             │
│  • SpecAugment (frequency & time masking) for regularization        │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 5: Evaluation                                                │
│  • Patient-wise split: 70/15/15 (train/val/test)                    │
│  • Threshold optimization via Youden's J statistic                  │
│  • Per-class metrics & confusion matrix                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Dataset

[**PhysioNet/Computing in Cardiology Challenge 2016**](https://physionet.org/content/challenge-2016/1.0.0/)

| Property | Value |
|----------|-------|
| Total Recordings | 3,240 (training sets a–f) |
| Preprocessed Segments | 50,077 (3-second windows) |
| Sample Rate | 2,000 Hz |
| Split | Patient-wise 70/15/15 (train/val/test) |
| Classes | Normal (–1), Abnormal (+1) |

---

## 📁 Project Structure

```
cardio-guard/
│
├── app.py                        # Gradio web application
├── generate_dummy_model.py       # Utility: create a dummy model for testing
├── generate_samples.py           # Utility: generate sample audio for testing
├── requirements.txt              # Python dependencies
├── packages.txt                  # Additional system packages (for Hugging Face Spaces)
├── pytest.ini                    # Pytest configuration
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Centralized configuration & hyperparameters
│   ├── phase1_data_inventory.py  # Phase 1: Data scanning & organization
│   ├── phase2_preprocess.py      # Phase 2: Filtering, denoising, segmentation
│   ├── phase3_spectrograms.py    # Phase 3: Mel spectrogram generation
│   ├── phase4_train.py           # Phase 4: Model training pipeline
│   └── phase5_evaluate.py        # Phase 5: Evaluation & threshold tuning
│
├── models/
│   ├── best_model.pt             # Trained model weights (deployment fallback)
│   └── optimal_threshold.txt     # Optimized decision threshold
│
├── output/
│   └── models/
│       └── best_model.pt         # Trained model weights (primary)
│
├── samples/
│   ├── normal_heart_sound.wav    # Sample normal heart sound
│   └── abnormal_heart_sound.wav  # Sample abnormal heart sound
│
└── tests/
    ├── __init__.py
    ├── test_phase1.py            # Unit tests for data inventory
    ├── test_phase2.py            # Unit tests for preprocessing
    ├── test_phase3.py            # Unit tests for spectrograms
    ├── test_phase4.py            # Unit tests for training
    ├── test_phase5.py            # Unit tests for evaluation
    └── test_phase6.py            # Unit tests for the web app
```

---

## 🛠 Installation

### Prerequisites

- Python **3.10+**
- [PyTorch](https://pytorch.org/) (see requirements.txt for CPU version; for GPU support, adjust the index URL)
- [FFmpeg](https://ffmpeg.org/) (for audio format conversion in the web app)

### Setup

```bash
# Clone the repository
git clone https://github.com/pranavv1210/cardio-guard.git
cd cardio-guard

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# (Optional) Install GPU version of PyTorch for faster inference
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 🚀 Usage

### Run the Gradio Web App

```bash
python app.py
```

This launches a local web interface at `http://localhost:7860`. You can:

- 🎤 **Record** a heart sound using your microphone
- 📂 **Upload** a `.wav`, `.flac`, `.ogg`, or `.mp3` file
- 🔍 Click **Analyze** to get instant classification with visualizations

### Generate Sample Audio (for testing)

```bash
python generate_samples.py
```

Creates sample `normal` and `abnormal` heart sound `.wav` files in the `samples/` directory.

### Generate a Dummy Model (for testing without GPU)

```bash
python generate_dummy_model.py
```

Creates a lightweight placeholder model for testing the pipeline end-to-end.

### Run All Tests

```bash
pytest
```

Runs 100+ unit tests across all 6 phases.

---

## 🧰 Utilities

| Script | Purpose |
|--------|---------|
| `generate_dummy_model.py` | Creates a small random-weight model for CI/testing |
| `generate_samples.py` | Generates synthetic normal/abnormal heart sound WAV files |
| `packages.txt` | System-level dependencies required for Hugging Face Spaces deployment |

---

## 🧪 Testing

The project includes **112 unit tests** across all phases:

| Phase | File | Tests |
|-------|------|-------|
| Phase 1 | `tests/test_phase1.py` | Data inventory & scanning |
| Phase 2 | `tests/test_phase2.py` | Preprocessing, filtering, denoising |
| Phase 3 | `tests/test_phase3.py` | Mel spectrogram generation |
| Phase 4 | `tests/test_phase4.py` | Model building & training pipeline |
| Phase 5 | `tests/test_phase5.py` | Evaluation & threshold optimization |
| Phase 6 | `tests/test_phase6.py` | Web app & inference logic |

```bash
# Run all tests
pytest

# Run with coverage
pip install pytest-cov
pytest --cov=src --cov-report=term-missing
```

---

## ☁ Deployment

This app is designed for easy deployment on **Hugging Face Spaces**:

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Gradio** SDK
3. Upload the repository files (the `README.md` frontmatter is already configured for Spaces)
4. The Space will automatically install dependencies from `requirements.txt` and `packages.txt`

---

## 🧩 Technologies Used

| Technology | Purpose |
|------------|---------|
| [PyTorch](https://pytorch.org/) | Deep learning framework |
| [EfficientNet-B0](https://arxiv.org/abs/1905.11946) (via `timm`) | CNN backbone for spectrogram classification |
| [Librosa](https://librosa.org/) | Audio analysis & mel spectrogram computation |
| [PyWavelets](https://pywavelets.readthedocs.io/) | Wavelet denoising |
| [Gradio](https://gradio.app/) | Web interface |
| [Matplotlib](https://matplotlib.org/) | Visualization |
| [Scikit-learn](https://scikit-learn.org/) | Evaluation metrics & data splitting |
| [OpenCV](https://opencv.org/) | Image processing (via albumentations transforms) |

---

## ⚠️ Disclaimer

> **This is a research project for educational and experimental purposes only.**  
> It is **NOT** a certified medical device. The model is not intended for clinical diagnosis or decision-making.  
> **Always consult a qualified cardiologist or healthcare professional** for any heart-related concerns.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for Advaya 2026**  
[GitHub](https://github.com/pranavv1210/cardio-guard) · [Report Bug](https://github.com/pranavv1210/cardio-guard/issues) · [Request Feature](https://github.com/pranavv1210/cardio-guard/issues)

</div>