"""Train a lightweight acoustic classifier on the public PhysioNet training set."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.acoustic_features import extract_acoustic_features
from src.config import PROJECT_ROOT, SAMPLE_RATE
from src.phase2_preprocess import preprocess_audio

DATASET_DIRS = [f"training-{letter}" for letter in "abcdef"]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "output" / "results"


def iter_reference_rows(root: Path):
    for dirname in DATASET_DIRS:
        folder = root / dirname
        reference = folder / "REFERENCE.csv"
        if not reference.exists():
            continue
        with reference.open("r", newline="") as handle:
            for row in csv.reader(handle):
                if len(row) < 2:
                    continue
                filename = row[0].strip()
                label = 1 if int(row[1]) == 1 else 0
                wav_path = folder / f"{filename}.wav"
                if wav_path.exists():
                    yield dirname, filename, wav_path, label


def build_feature_table(root: Path, cache_path: Path) -> pd.DataFrame:
    if cache_path.exists():
        return pd.read_csv(cache_path)

    rows = []
    for idx, (database, filename, wav_path, label) in enumerate(iter_reference_rows(root), start=1):
        audio, _ = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
        audio = preprocess_audio(audio.astype(np.float32), sr=SAMPLE_RATE)
        features = extract_acoustic_features(audio, SAMPLE_RATE)
        rows.append({
            "database": database,
            "filename": filename,
            "wav_path": str(wav_path),
            "label": label,
            **{f"f_{i:03d}": float(value) for i, value in enumerate(features)},
        })
        if idx % 100 == 0:
            logging.info("Extracted features for %d recordings", idx)

    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    return df


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "normal_recall_specificity": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "abnormal_recall_sensitivity": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "abnormal_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "abnormal_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": ["NORMAL", "ABNORMAL"],
        "confusion_matrix": cm.tolist(),
        "mean_abnormal_probability": float(np.mean(y_prob[:, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_path = RESULTS_DIR / "public_training_acoustic_features.csv"
    df = build_feature_table(args.root, feature_path)
    feature_cols = [col for col in df.columns if col.startswith("f_")]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", VotingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=args.seed,
                    n_jobs=-1,
                )),
                ("xt", ExtraTreesClassifier(
                    n_estimators=600,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=args.seed + 1,
                    n_jobs=-1,
                )),
            ],
            voting="soft",
            weights=[1, 1],
        )),
    ])
    model.fit(X_train, y_train)

    holdout_prob = model.predict_proba(X_test)
    holdout_pred = np.argmax(holdout_prob, axis=1)
    holdout_metrics = compute_metrics(y_test, holdout_pred, holdout_prob)
    holdout_metrics.update({
        "dataset": "public PhysioNet/CinC 2016 training archive",
        "split": f"{int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} stratified train/holdout",
        "train_recordings": int(len(y_train)),
        "holdout_recordings": int(len(y_test)),
    })

    full_prob = model.predict_proba(X)
    full_pred = np.argmax(full_prob, axis=1)
    full_metrics = compute_metrics(y, full_pred, full_prob)
    full_metrics.update({
        "dataset": "public PhysioNet/CinC 2016 training archive",
        "split": "model refit artifact evaluated on all public records",
        "evaluated_recordings": int(len(y)),
    })

    model_path = MODELS_DIR / "acoustic_model.joblib"
    joblib.dump(model, model_path)
    (RESULTS_DIR / "acoustic_holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2))
    (RESULTS_DIR / "acoustic_full_public_metrics.json").write_text(json.dumps(full_metrics, indent=2))

    print(json.dumps({"holdout": holdout_metrics, "full_public": full_metrics}, indent=2))
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
