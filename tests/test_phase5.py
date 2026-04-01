"""
Tests for Phase 5 — Evaluation & Metrics
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import accuracy_score

from src.config import LABEL_NAMES, RESULTS_DIR
from src.phase5_evaluate import (
    load_trained_model,
    predict_dataset,
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_training_history,
    find_optimal_threshold,
    apply_threshold,
    save_optimal_threshold,
    load_optimal_threshold,
)
from src.phase4_train import build_model, SpectrogramDataset, get_transforms


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_model(tmp_path):
    """Create and save a dummy model for testing."""
    model = build_model(pretrained=False)
    model_path = tmp_path / "test_model.pt"
    torch.save(model.state_dict(), str(model_path))
    return model_path


@pytest.fixture
def dummy_predictions():
    """Create dummy prediction results."""
    np.random.seed(42)
    n = 100
    true_labels = np.array([0] * 60 + [1] * 40)
    predictions = true_labels.copy()
    # Introduce some errors
    predictions[5] = 1   # false positive
    predictions[10] = 1  # false positive
    predictions[65] = 0  # false negative
    predictions[70] = 0  # false negative
    predictions[75] = 0  # false negative

    probs = np.zeros((n, 2))
    for i in range(n):
        if predictions[i] == 0:
            probs[i] = [0.8, 0.2]
        else:
            probs[i] = [0.2, 0.8]

    return true_labels, predictions, probs


@pytest.fixture
def dummy_spectrogram_dataset(tmp_path):
    """Create a tiny dataset of fake spectrogram PNGs."""
    from PIL import Image

    records = []
    for i in range(20):
        cls = 0 if i < 12 else 1
        cls_name = "normal" if cls == 0 else "abnormal"
        img_path = tmp_path / "spectrograms" / "test" / cls_name / f"seg_{i:04d}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img = Image.fromarray(np.random.randint(0, 255, (128, 128), dtype=np.uint8))
        img.save(str(img_path))
        records.append({
            "spectrogram_path": str(img_path),
            "class_idx": cls,
            "split": "test",
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# load_trained_model
# ──────────────────────────────────────────────────────────────────────────────
class TestLoadTrainedModel:
    def test_loads_model(self, dummy_model):
        device = torch.device("cpu")
        model = load_trained_model(dummy_model, device)
        assert model is not None
        assert not model.training  # Should be in eval mode

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_trained_model(tmp_path / "nonexistent.pt", torch.device("cpu"))

    def test_model_on_correct_device(self, dummy_model):
        device = torch.device("cpu")
        model = load_trained_model(dummy_model, device)
        param_device = next(model.parameters()).device
        assert param_device == device


# ──────────────────────────────────────────────────────────────────────────────
# compute_metrics
# ──────────────────────────────────────────────────────────────────────────────
class TestComputeMetrics:
    def test_all_keys_present(self, dummy_predictions):
        true_labels, predictions, probs = dummy_predictions
        metrics = compute_metrics(true_labels, predictions, probs)

        required_keys = [
            "accuracy", "sensitivity", "specificity",
            "precision", "f1_score", "auc_roc",
            "confusion_matrix", "classification_report",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_accuracy_in_range(self, dummy_predictions):
        true_labels, predictions, probs = dummy_predictions
        metrics = compute_metrics(true_labels, predictions, probs)
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_sensitivity_is_recall_for_abnormal(self, dummy_predictions):
        true_labels, predictions, probs = dummy_predictions
        metrics = compute_metrics(true_labels, predictions, probs)
        # Sensitivity = TP / (TP + FN) for abnormal class
        assert 0.0 <= metrics["sensitivity"] <= 1.0

    def test_specificity_is_recall_for_normal(self, dummy_predictions):
        true_labels, predictions, probs = dummy_predictions
        metrics = compute_metrics(true_labels, predictions, probs)
        assert 0.0 <= metrics["specificity"] <= 1.0

    def test_perfect_predictions(self):
        n = 50
        true = np.array([0] * 25 + [1] * 25)
        preds = true.copy()
        probs = np.zeros((n, 2))
        for i in range(n):
            probs[i, true[i]] = 1.0

        metrics = compute_metrics(true, preds, probs)
        assert metrics["accuracy"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["auc_roc"] == 1.0

    def test_confusion_matrix_shape(self, dummy_predictions):
        true_labels, predictions, probs = dummy_predictions
        metrics = compute_metrics(true_labels, predictions, probs)
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_confusion_matrix_sum(self, dummy_predictions):
        true_labels, predictions, probs = dummy_predictions
        metrics = compute_metrics(true_labels, predictions, probs)
        assert metrics["confusion_matrix"].sum() == len(true_labels)


# ──────────────────────────────────────────────────────────────────────────────
# predict_dataset
# ──────────────────────────────────────────────────────────────────────────────
class TestPredictDataset:
    def test_predict_returns_correct_shapes(self, dummy_model, dummy_spectrogram_dataset):
        device = torch.device("cpu")
        model = load_trained_model(dummy_model, device)

        dataset = SpectrogramDataset(
            dummy_spectrogram_dataset, transform=get_transforms(is_train=False)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        results = predict_dataset(model, loader, device)
        n = len(dummy_spectrogram_dataset)

        assert results["predictions"].shape == (n,)
        assert results["true_labels"].shape == (n,)
        assert results["probabilities"].shape == (n, 2)

    def test_probabilities_sum_to_one(self, dummy_model, dummy_spectrogram_dataset):
        device = torch.device("cpu")
        model = load_trained_model(dummy_model, device)

        dataset = SpectrogramDataset(
            dummy_spectrogram_dataset, transform=get_transforms(is_train=False)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

        results = predict_dataset(model, loader, device)
        sums = results["probabilities"].sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────
class TestPlots:
    def test_confusion_matrix_plot(self, tmp_path):
        cm = np.array([[50, 5], [3, 42]])
        path = plot_confusion_matrix(cm, tmp_path / "cm.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_roc_curve_plot(self, tmp_path, dummy_predictions):
        true_labels, _, probs = dummy_predictions
        path = plot_roc_curve(true_labels, probs, tmp_path / "roc.png")
        assert path.exists()
        assert path.stat().st_size > 0

    def test_training_history_plot(self, tmp_path):
        hist = pd.DataFrame({
            "stage": [1, 1, 1, 2, 2, 2],
            "epoch": [1, 2, 3, 1, 2, 3],
            "train_loss": [0.8, 0.6, 0.4, 0.3, 0.2, 0.15],
            "train_acc": [0.6, 0.7, 0.8, 0.85, 0.9, 0.92],
            "val_loss": [0.9, 0.7, 0.5, 0.4, 0.3, 0.25],
            "val_acc": [0.55, 0.65, 0.75, 0.8, 0.85, 0.88],
        })
        csv_path = tmp_path / "history.csv"
        hist.to_csv(str(csv_path), index=False)

        path = plot_training_history(csv_path, tmp_path / "hist.png")
        assert path.exists()
        assert path.stat().st_size > 0


# ──────────────────────────────────────────────────────────────────────────────
# Threshold Optimization
# ──────────────────────────────────────────────────────────────────────────────
class TestThresholdOptimization:
    def test_find_optimal_threshold_youden(self):
        """Optimal threshold should be between 0 and 1."""
        np.random.seed(42)
        n = 200
        true = np.array([0] * 100 + [1] * 100)
        probs = np.zeros((n, 2))
        # Good separation: normals have low abnormal prob, abnormals have high
        probs[:100, 1] = np.random.uniform(0.0, 0.4, 100)
        probs[100:, 1] = np.random.uniform(0.6, 1.0, 100)
        probs[:, 0] = 1.0 - probs[:, 1]

        threshold, j_score = find_optimal_threshold(true, probs, method="youden")
        assert 0.0 < threshold < 1.0
        assert j_score > 0.0

    def test_find_optimal_threshold_f1(self):
        """F1 method should return valid threshold."""
        np.random.seed(42)
        n = 200
        true = np.array([0] * 100 + [1] * 100)
        probs = np.zeros((n, 2))
        probs[:100, 1] = np.random.uniform(0.0, 0.4, 100)
        probs[100:, 1] = np.random.uniform(0.6, 1.0, 100)
        probs[:, 0] = 1.0 - probs[:, 1]

        threshold, best_f1 = find_optimal_threshold(true, probs, method="f1")
        assert 0.1 <= threshold < 0.95
        assert best_f1 > 0.0

    def test_find_optimal_threshold_invalid_method(self):
        true = np.array([0, 1])
        probs = np.array([[0.8, 0.2], [0.3, 0.7]])
        with pytest.raises(ValueError, match="Unknown method"):
            find_optimal_threshold(true, probs, method="invalid")

    def test_apply_threshold_default(self):
        probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        preds = apply_threshold(probs, 0.5)
        np.testing.assert_array_equal(preds, [0, 1, 1])

    def test_apply_threshold_custom(self):
        probs = np.array([[0.4, 0.6], [0.2, 0.8], [0.7, 0.3]])
        preds = apply_threshold(probs, 0.7)
        # Only 0.8 >= 0.7
        np.testing.assert_array_equal(preds, [0, 1, 0])

    def test_save_and_load_threshold(self, tmp_path, monkeypatch):
        """Save and load round-trip."""
        import src.phase5_evaluate as phase5_evaluate
        fake_path = tmp_path / "threshold.txt"
        monkeypatch.setattr(phase5_evaluate, "OPTIMAL_THRESHOLD_FILE", fake_path)

        save_optimal_threshold(0.6234)
        loaded = load_optimal_threshold()
        assert abs(loaded - 0.6234) < 1e-4

    def test_load_threshold_default_when_missing(self, tmp_path, monkeypatch):
        """Load should return 0.5 if no file exists."""
        import src.phase5_evaluate as phase5_evaluate
        fake_path = tmp_path / "nonexistent_threshold.txt"
        monkeypatch.setattr(phase5_evaluate, "OPTIMAL_THRESHOLD_FILE", fake_path)

        loaded = load_optimal_threshold()
        assert loaded == 0.5

    def test_optimal_threshold_improves_accuracy(self):
        """Optimized threshold on well-separated data should beat 0.5."""
        np.random.seed(42)
        n = 400
        true = np.array([0] * 200 + [1] * 200)
        probs = np.zeros((n, 2))
        # Normals cluster around 0.3 abnormal prob
        probs[:200, 1] = np.random.normal(0.3, 0.1, 200).clip(0, 1)
        # Abnormals cluster around 0.7 abnormal prob
        probs[200:, 1] = np.random.normal(0.7, 0.1, 200).clip(0, 1)
        probs[:, 0] = 1.0 - probs[:, 1]

        # Default threshold
        default_preds = apply_threshold(probs, 0.5)
        default_acc = accuracy_score(true, default_preds)

        # Optimized threshold
        opt_thresh, _ = find_optimal_threshold(true, probs, method="youden")
        opt_preds = apply_threshold(probs, opt_thresh)
        opt_acc = accuracy_score(true, opt_preds)

        assert opt_acc >= default_acc
