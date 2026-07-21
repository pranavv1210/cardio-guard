"""Evaluate deployed CardioGuard inference on the public PhysioNet training set."""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from app import predict_heart_sound
from src.config import PROJECT_ROOT

DATASET_DIRS = [f"training-{letter}" for letter in "abcdef"]
RESULTS_DIR = PROJECT_ROOT / "output" / "results"


def iter_reference_rows(root: Path):
    for dirname in DATASET_DIRS:
        folder = root / dirname
        reference = folder / "REFERENCE.csv"
        if not reference.exists():
            continue

        with reference.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 2:
                    continue
                filename = row[0].strip()
                label = int(row[1])
                wav_path = folder / f"{filename}.wav"
                if wav_path.exists():
                    yield {
                        "database": dirname,
                        "filename": filename,
                        "wav_path": str(wav_path),
                        "true_label": "ABNORMAL" if label == 1 else "NORMAL",
                    }


def evaluate_public_training(root: Path, limit: int | None = None) -> tuple[pd.DataFrame, dict]:
    rows = []
    for idx, item in enumerate(iter_reference_rows(root), start=1):
        if limit is not None and idx > limit:
            break

        try:
            result = predict_heart_sound(item["wav_path"])
            predicted = result.label
            normal_probability = result.normal_probability
            abnormal_probability = result.abnormal_probability
            decision_source = result.decision_source
            error = ""
            plt.close(result.waveform_figure)
            plt.close(result.spectrogram_figure)
        except Exception as exc:
            predicted = "ERROR"
            normal_probability = 0.0
            abnormal_probability = 0.0
            decision_source = "ERROR"
            error = str(exc)

        rows.append({
            **item,
            "predicted_label": predicted,
            "normal_probability": normal_probability,
            "abnormal_probability": abnormal_probability,
            "decision_source": decision_source,
            "error": error,
        })

        if idx % 100 == 0:
            logging.info("Evaluated %d recordings", idx)

    df = pd.DataFrame(rows)
    valid = df[df["predicted_label"].isin(["NORMAL", "ABNORMAL"])].copy()

    y_true = valid["true_label"].tolist()
    y_pred = valid["predicted_label"].tolist()
    labels = ["NORMAL", "ABNORMAL"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    metrics = {
        "evaluated_recordings": int(len(valid)),
        "errored_recordings": int((df["predicted_label"] == "ERROR").sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "normal_recall_specificity": float(recall_score(y_true, y_pred, pos_label="NORMAL", zero_division=0)) if y_true else 0.0,
        "abnormal_recall_sensitivity": float(recall_score(y_true, y_pred, pos_label="ABNORMAL", zero_division=0)) if y_true else 0.0,
        "abnormal_precision": float(precision_score(y_true, y_pred, pos_label="ABNORMAL", zero_division=0)) if y_true else 0.0,
        "abnormal_f1": float(f1_score(y_true, y_pred, pos_label="ABNORMAL", zero_division=0)) if y_true else 0.0,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    return df, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df, metrics = evaluate_public_training(args.root, args.limit)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit_{args.limit}" if args.limit else ""
    predictions_path = RESULTS_DIR / f"public_training_predictions{suffix}.csv"
    metrics_path = RESULTS_DIR / f"public_training_metrics{suffix}.json"
    df.to_csv(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    print(f"Saved predictions: {predictions_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
