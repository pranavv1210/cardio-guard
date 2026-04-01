---
title: Heart Sound Classifier
emoji: 🩺
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.10.0
app_file: app.py
pinned: false
license: mit
---

# 🩺 Heart Sound Classifier

AI-powered heart sound analysis using an EfficientNet-B0 CNN trained on the **PhysioNet/CinC 2016** dataset.

Upload or record a heart sound — the model classifies it as **Normal** or **Abnormal** and displays the waveform and mel spectrogram.

## Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | 81.10% |
| Sensitivity (Recall for Abnormal) | 95.53% |
| Specificity | 76.71% |
| AUC-ROC | 92.08% |

Threshold optimized via Youden's J statistic (threshold = 0.74).

## Dataset

[PhysioNet/Computing in Cardiology Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/)
- 3,240 recordings from training sets a–f
- 50,077 preprocessed 3-second segments
- Patient-wise train/val/test split (70/15/15)

## Pipeline

1. **Preprocessing** — Bandpass filter (20–1000 Hz) + Daubechies-4 wavelet denoising
2. **Segmentation** — 3-second windows with 50% overlap at 2000 Hz
3. **Mel Spectrogram** — 128-band mel spectrogram → 128×128 image
4. **Training** — Two-stage EfficientNet-B0 transfer learning with SpecAugment
5. **Inference** — Multi-segment majority voting for long recordings

## Project Structure

```
├── app.py               # Gradio web app
├── src/
│   ├── config.py
│   ├── phase1_data_inventory.py
│   ├── phase2_preprocess.py
│   ├── phase3_spectrograms.py
│   ├── phase4_train.py
│   └── phase5_evaluate.py
├── models/
│   ├── best_model.pt
│   └── optimal_threshold.txt
└── tests/               # 112 unit tests across all phases
```

## Disclaimer

This is a research project, not a medical device. Always consult a cardiologist for diagnosis.
