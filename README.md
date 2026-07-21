# CardioGuard

Mobile-first heart sound classifier for Normal vs Abnormal phonocardiogram recordings.

The inference pipeline is:

1. Load audio at 2000 Hz mono.
2. Apply bandpass filtering, wavelet denoising, amplitude normalization, and silence trimming.
3. Convert 3-second windows into 128 x 128 mel spectrogram images.
4. Run EfficientNet-B0.
5. Predict `ABNORMAL` when class-1 probability is greater than or equal to the saved optimal threshold.

Class mapping is fixed:

| Class index | Label |
| --- | --- |
| `0` | Normal |
| `1` | Abnormal |

## Accuracy And Validation

Current repository validation:

| Validation set | Accuracy |
| --- | --- |
| Bundled smoke-test samples (`samples/normal_heart_sound.wav`, `samples/abnormal_heart_sound.wav`) | 100% (2/2) |

The repository does not currently include the full held-out PhysioNet test split, prediction CSV, confusion matrix, or metrics artifact needed to recompute a whole-project test accuracy from source. Because of that, the only accuracy that can be verified directly from this checkout is the bundled sample sanity check above.

Important implementation detail: the included CNN checkpoint is low-confidence on the bundled samples, so the app uses an acoustic murmur fallback when the CNN confidence is below 60%. This prevents the app from blindly returning `NORMAL` for murmur-like abnormal recordings when the checkpoint is near 50/50.

## Run Locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\streamlit.exe run app.py
```

Open `http://localhost:8501` on your computer, or use the network URL printed by Streamlit on your phone if both devices are on the same Wi-Fi.

## Deploy On Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app.
3. Select `app.py` as the main file.
4. Ensure these files are included:
   - `app.py`
   - `requirements.txt`
   - `packages.txt`
   - `.streamlit/config.toml`
   - `src/`
   - `models/best_model.pt`
   - `models/optimal_threshold.txt`

## Important Files

| File | Purpose |
| --- | --- |
| `app.py` | Streamlit mobile app and inference wrapper |
| `src/phase2_preprocess.py` | Audio filtering, denoising, normalization, trimming |
| `src/phase3_spectrograms.py` | Mel spectrogram generation |
| `src/phase4_train.py` | EfficientNet model and training helpers |
| `src/phase5_evaluate.py` | Metrics and decision threshold handling |
| `models/best_model.pt` | Deployment model weights |
| `models/optimal_threshold.txt` | Deployment threshold |

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The app tests include explicit guards for the most important behavior:

- low abnormal probability predicts `NORMAL`
- high abnormal probability predicts `ABNORMAL`
- bundled normal sample predicts `NORMAL`
- bundled abnormal sample predicts `ABNORMAL`
- short or missing audio returns a clear error

## Disclaimer

This is a research prototype, not a certified medical device. It must not be used as a clinical diagnosis or replacement for a qualified healthcare professional.
