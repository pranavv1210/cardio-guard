"""
Phase 5 — Evaluation & Metrics
Compute accuracy, sensitivity, specificity, F1, AUC-ROC, confusion matrix.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import (
    BATCH_SIZE, NUM_WORKERS, MODEL_NAME, NUM_CLASSES, DROPOUT_RATE,
    SPEC_HEIGHT, SPEC_WIDTH,
    MODELS_DIR, RESULTS_DIR, OUTPUT_DIR, LABEL_NAMES,
)
from src.phase4_train import (
    SpectrogramDataset,
    get_transforms,
    build_model,
)

logger = logging.getLogger(__name__)

OPTIMAL_THRESHOLD_FILE = MODELS_DIR / "optimal_threshold.txt"


def load_trained_model(
    model_path: Path,
    device: torch.device,
) -> nn.Module:
    """Load a trained model from checkpoint.

    Args:
        model_path: Path to .pt checkpoint.
        device: Target device.

    Returns:
        Model in eval mode on the specified device.

    Raises:
        FileNotFoundError: If model_path doesn't exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = build_model(pretrained=False)
    state_dict = torch.load(str(model_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run inference on an entire dataset.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader for the dataset.
        device: torch device.

    Returns:
        Dict with 'predictions', 'true_labels', 'probabilities'.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        "predictions": np.array(all_preds),
        "true_labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


def compute_metrics(
    true_labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        true_labels: Ground truth (0 or 1).
        predictions: Model predictions (0 or 1).
        probabilities: Softmax probabilities (N, 2).

    Returns:
        Dict with accuracy, sensitivity, specificity, f1, auc_roc, etc.
    """
    acc = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, pos_label=1, zero_division=0)
    sensitivity = recall_score(true_labels, predictions, pos_label=1, zero_division=0)
    f1 = f1_score(true_labels, predictions, pos_label=1, zero_division=0)

    # Specificity = recall for normal class
    specificity = recall_score(true_labels, predictions, pos_label=0, zero_division=0)

    # AUC-ROC using probability of abnormal class
    try:
        auc = roc_auc_score(true_labels, probabilities[:, 1])
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(true_labels, predictions)

    report = classification_report(
        true_labels, predictions,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
        output_dict=True,
    )

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def find_optimal_threshold(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    method: str = "youden",
) -> tuple:
    """Find the optimal classification threshold on a validation set.

    Args:
        true_labels: Ground truth (0 or 1).
        probabilities: Softmax probabilities (N, 2).
        method: 'youden' (maximize sensitivity+specificity-1)
                or 'f1' (maximize F1 score).

    Returns:
        (optimal_threshold, best_score)
    """
    fpr, tpr, thresholds = roc_curve(true_labels, probabilities[:, 1])

    if method == "youden":
        # Youden's J statistic = sensitivity + specificity - 1 = TPR - FPR
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx]), float(j_scores[best_idx])
    elif method == "f1":
        best_f1 = 0.0
        best_thresh = 0.5
        for t in np.arange(0.1, 0.95, 0.01):
            preds = (probabilities[:, 1] >= t).astype(int)
            f1_val = f1_score(true_labels, preds, pos_label=1, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_thresh = t
        return best_thresh, best_f1
    else:
        raise ValueError(f"Unknown method: {method}")


def apply_threshold(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply a custom threshold to convert probabilities to predictions.

    Args:
        probabilities: Softmax probabilities (N, 2).
        threshold: Decision boundary for the abnormal class.

    Returns:
        Predictions array (0 or 1).
    """
    return (probabilities[:, 1] >= threshold).astype(int)


def save_optimal_threshold(threshold: float) -> Path:
    """Save optimal threshold to a file for use by the Gradio app.

    Args:
        threshold: Optimal threshold value.

    Returns:
        Path where threshold was saved.
    """
    OPTIMAL_THRESHOLD_FILE.parent.mkdir(parents=True, exist_ok=True)
    OPTIMAL_THRESHOLD_FILE.write_text(str(threshold))
    return OPTIMAL_THRESHOLD_FILE


def load_optimal_threshold() -> float:
    """Load the saved optimal threshold, defaulting to 0.5.

    Returns:
        Threshold value.
    """
    if OPTIMAL_THRESHOLD_FILE.exists():
        return float(OPTIMAL_THRESHOLD_FILE.read_text().strip())
    # Fallback: models/ folder used in deployment
    _deploy_threshold = OPTIMAL_THRESHOLD_FILE.parent.parent.parent / "models" / "optimal_threshold.txt"
    if _deploy_threshold.exists():
        return float(_deploy_threshold.read_text().strip())
    return 0.5


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: Path,
    class_names: list = None,
) -> Path:
    """Plot and save a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array.
        output_path: Path to save the plot.
        class_names: Labels for axes.

    Returns:
        Path where plot was saved.
    """
    if class_names is None:
        class_names = [LABEL_NAMES[0], LABEL_NAMES[1]]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, cbar=True,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Heart Sound Classifier", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    return output_path


def plot_roc_curve(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> Path:
    """Plot and save the ROC curve.

    Args:
        true_labels: Ground truth labels.
        probabilities: Softmax probabilities (N, 2).
        output_path: Path to save the plot.

    Returns:
        Path where plot was saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
    auc = roc_auc_score(true_labels, probabilities[:, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Heart Sound Classifier", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    return output_path


def plot_training_history(
    history_csv: Path,
    output_path: Path,
) -> Path:
    """Plot training/validation loss and accuracy curves.

    Args:
        history_csv: Path to training_history.csv.
        output_path: Path to save the plot.

    Returns:
        Path where plot was saved.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hist = pd.read_csv(str(history_csv))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(hist["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(hist["val_loss"], label="Val Loss", color="red")
    # Mark stage boundary
    stage1_end = len(hist[hist["stage"] == 1])
    if stage1_end > 0:
        axes[0].axvline(x=stage1_end - 0.5, color="gray", linestyle="--", alpha=0.7, label="Stage 1→2")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(hist["train_acc"], label="Train Acc", color="blue")
    axes[1].plot(hist["val_acc"], label="Val Acc", color="red")
    if stage1_end > 0:
        axes[1].axvline(x=stage1_end - 0.5, color="gray", linestyle="--", alpha=0.7, label="Stage 1→2")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

    return output_path


def run_phase5(
    spec_csv: Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """Execute the full Phase 5 evaluation pipeline.

    Args:
        spec_csv: Path to spectrograms_metadata.csv.
        model_path: Path to best_model.pt.

    Returns:
        Dict with all metrics.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("=" * 60)
    logger.info("PHASE 5 — Evaluation & Metrics")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load model
    if model_path is None:
        model_path = MODELS_DIR / "best_model.pt"
    else:
        model_path = Path(model_path)

    model = load_trained_model(model_path, device)
    logger.info("Loaded model from %s", model_path)

    # Load test data
    if spec_csv is None:
        spec_csv = str(OUTPUT_DIR / "spectrograms_metadata.csv")

    df = pd.read_csv(spec_csv)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    logger.info("Test set: %d samples", len(test_df))
    logger.info("Validation set: %d samples", len(val_df))

    test_dataset = SpectrogramDataset(test_df, transform=get_transforms(is_train=False))
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Find optimal threshold on validation set ─────────────────────────────
    val_dataset = SpectrogramDataset(val_df, transform=get_transforms(is_train=False))
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    logger.info("Running inference on validation set for threshold tuning...")
    val_results = predict_dataset(model, val_loader, device)

    optimal_threshold, j_score = find_optimal_threshold(
        val_results["true_labels"],
        val_results["probabilities"],
        method="youden",
    )
    logger.info("Optimal threshold (Youden's J): %.4f (J=%.4f)", optimal_threshold, j_score)
    save_optimal_threshold(optimal_threshold)

    # Run inference on test set
    logger.info("Running inference on test set...")
    results = predict_dataset(model, test_loader, device)

    # ── Metrics with DEFAULT threshold (0.5) ─────────────────────────────────
    metrics = compute_metrics(
        results["true_labels"],
        results["predictions"],
        results["probabilities"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 40)
    logger.info("TEST RESULTS — Default Threshold (0.50)")
    logger.info("=" * 40)
    logger.info("Accuracy:    %.4f", metrics["accuracy"])
    logger.info("Sensitivity: %.4f", metrics["sensitivity"])
    logger.info("Specificity: %.4f", metrics["specificity"])
    logger.info("Precision:   %.4f", metrics["precision"])
    logger.info("F1 Score:    %.4f", metrics["f1_score"])
    logger.info("AUC-ROC:     %.4f", metrics["auc_roc"])
    logger.info("\nConfusion Matrix:\n%s", metrics["confusion_matrix"])

    # ── Metrics with OPTIMIZED threshold ─────────────────────────────────────
    opt_preds = apply_threshold(results["probabilities"], optimal_threshold)
    opt_metrics = compute_metrics(
        results["true_labels"],
        opt_preds,
        results["probabilities"],
    )

    logger.info("\n" + "=" * 40)
    logger.info("TEST RESULTS — Optimized Threshold (%.4f)", optimal_threshold)
    logger.info("=" * 40)
    logger.info("Accuracy:    %.4f", opt_metrics["accuracy"])
    logger.info("Sensitivity: %.4f", opt_metrics["sensitivity"])
    logger.info("Specificity: %.4f", opt_metrics["specificity"])
    logger.info("Precision:   %.4f", opt_metrics["precision"])
    logger.info("F1 Score:    %.4f", opt_metrics["f1_score"])
    logger.info("AUC-ROC:     %.4f", opt_metrics["auc_roc"])
    logger.info("\nConfusion Matrix:\n%s", opt_metrics["confusion_matrix"])

    # Save plots (using optimized results)
    cm_path = plot_confusion_matrix(
        opt_metrics["confusion_matrix"],
        RESULTS_DIR / "confusion_matrix.png",
    )
    logger.info("Saved confusion matrix to %s", cm_path)

    roc_path = plot_roc_curve(
        results["true_labels"],
        results["probabilities"],
        RESULTS_DIR / "roc_curve.png",
    )
    logger.info("Saved ROC curve to %s", roc_path)

    # Training history plot
    hist_csv = OUTPUT_DIR / "training_history.csv"
    if hist_csv.exists():
        hist_path = plot_training_history(hist_csv, RESULTS_DIR / "training_history.png")
        logger.info("Saved training history to %s", hist_path)

    # Save metrics to CSV (both default and optimized)
    metrics_flat = {k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "classification_report")}
    metrics_flat = {"threshold": 0.5, **metrics_flat}

    opt_flat = {k: v for k, v in opt_metrics.items()
                if k not in ("confusion_matrix", "classification_report")}
    opt_flat = {"threshold": optimal_threshold, **opt_flat}

    metrics_df = pd.DataFrame([metrics_flat, opt_flat])
    metrics_df.to_csv(str(RESULTS_DIR / "test_metrics.csv"), index=False)

    # Save full report (optimized)
    report_df = pd.DataFrame(opt_metrics["classification_report"]).transpose()
    report_df.to_csv(str(RESULTS_DIR / "classification_report.csv"))

    logger.info("\nOptimal threshold saved to %s", OPTIMAL_THRESHOLD_FILE)
    logger.info("All results saved to %s", RESULTS_DIR)

    return opt_metrics


if __name__ == "__main__":
    run_phase5()
