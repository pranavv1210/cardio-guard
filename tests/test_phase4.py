"""
Tests for Phase 4 — CNN Training
"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path

from src.config import (
    NUM_CLASSES, SPEC_HEIGHT, SPEC_WIDTH, BATCH_SIZE, MODEL_NAME,
    DROPOUT_RATE, SEGMENT_SAMPLES,
)
from src.phase4_train import (
    SpecAugment,
    SpectrogramDataset,
    get_transforms,
    build_model,
    get_weighted_sampler,
    get_class_weights,
    train_one_epoch,
    evaluate,
    freeze_backbone,
    unfreeze_last_blocks,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def dummy_spec_df(tmp_path):
    """Create a small DataFrame with actual .png spectrogram files."""
    rows = []
    for i in range(16):
        class_idx = i % 2
        path = tmp_path / f"spec_{i:03d}.png"
        # Create a random 128x128 grayscale image → RGB
        arr = np.random.randint(0, 255, (SPEC_HEIGHT, SPEC_WIDTH), dtype=np.uint8)
        img = Image.fromarray(arr)
        img.save(str(path))
        rows.append({"spectrogram_path": str(path), "class_idx": class_idx})
    return pd.DataFrame(rows)


@pytest.fixture
def small_model():
    """Build model for testing (pretrained=False to speed up)."""
    return build_model(pretrained=False)


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────────────────────────────────────
# SpecAugment
# ──────────────────────────────────────────────────────────────────────────────
class TestSpecAugment:
    def test_output_shape(self):
        aug = SpecAugment(freq_mask=10, time_mask=10)
        img = torch.randn(3, SPEC_HEIGHT, SPEC_WIDTH)
        result = aug(img)
        assert result.shape == img.shape

    def test_has_zeros(self):
        """After masking, some values should be zero."""
        aug = SpecAugment(freq_mask=20, time_mask=20)
        img = torch.ones(3, SPEC_HEIGHT, SPEC_WIDTH)
        result = aug(img)
        assert (result == 0).any()

    def test_no_mask(self):
        aug = SpecAugment(freq_mask=0, time_mask=0)
        img = torch.ones(3, 64, 64)
        result = aug(img)
        assert torch.equal(result, img)


# ──────────────────────────────────────────────────────────────────────────────
# SpectrogramDataset
# ──────────────────────────────────────────────────────────────────────────────
class TestSpectrogramDataset:
    def test_length(self, dummy_spec_df):
        ds = SpectrogramDataset(dummy_spec_df, transform=get_transforms(is_train=False))
        assert len(ds) == 16

    def test_getitem_returns_tensor(self, dummy_spec_df):
        ds = SpectrogramDataset(dummy_spec_df, transform=get_transforms(is_train=False))
        img, label = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, SPEC_HEIGHT, SPEC_WIDTH)
        assert isinstance(label, int)

    def test_labels_correct(self, dummy_spec_df):
        ds = SpectrogramDataset(dummy_spec_df, transform=get_transforms(is_train=False))
        for i in range(len(ds)):
            _, label = ds[i]
            assert label == i % 2


# ──────────────────────────────────────────────────────────────────────────────
# build_model
# ──────────────────────────────────────────────────────────────────────────────
class TestBuildModel:
    def test_output_classes(self, small_model):
        # Check classifier outputs correct number of classes
        x = torch.randn(1, 3, SPEC_HEIGHT, SPEC_WIDTH)
        out = small_model(x)
        assert out.shape == (1, NUM_CLASSES)

    def test_has_dropout(self, small_model):
        has_dropout = False
        for module in small_model.classifier.modules():
            if isinstance(module, nn.Dropout):
                has_dropout = True
                assert module.p == DROPOUT_RATE
        assert has_dropout

    def test_is_nn_module(self, small_model):
        assert isinstance(small_model, nn.Module)


# ──────────────────────────────────────────────────────────────────────────────
# get_weighted_sampler
# ──────────────────────────────────────────────────────────────────────────────
class TestGetWeightedSampler:
    def test_returns_sampler(self):
        labels = [0, 0, 0, 0, 1, 1]
        sampler = get_weighted_sampler(labels)
        assert len(sampler) == 6

    def test_imbalanced_weights(self):
        labels = [0] * 100 + [1] * 10
        sampler = get_weighted_sampler(labels)
        weights = list(sampler.weights)
        # Minority class should have higher weight
        assert weights[100] > weights[0]


# ──────────────────────────────────────────────────────────────────────────────
# get_class_weights
# ──────────────────────────────────────────────────────────────────────────────
class TestGetClassWeights:
    def test_shape(self, device):
        weights = get_class_weights([0, 0, 0, 1, 1], device)
        assert weights.shape == (2,)

    def test_minority_gets_higher_weight(self, device):
        weights = get_class_weights([0] * 100 + [1] * 10, device)
        assert weights[1] > weights[0]


# ──────────────────────────────────────────────────────────────────────────────
# freeze / unfreeze
# ──────────────────────────────────────────────────────────────────────────────
class TestFreezeUnfreeze:
    def test_freeze_backbone(self, small_model):
        freeze_backbone(small_model)
        for name, param in small_model.named_parameters():
            if "classifier" in name:
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_unfreeze_last_blocks(self, small_model):
        freeze_backbone(small_model)
        unfreeze_last_blocks(small_model, num_blocks=2)
        # Classifier should be trainable
        for param in small_model.classifier.parameters():
            assert param.requires_grad
        # Some backbone params should now be trainable
        trainable_backbone = sum(
            1 for name, p in small_model.named_parameters()
            if p.requires_grad and "classifier" not in name
        )
        assert trainable_backbone > 0


# ──────────────────────────────────────────────────────────────────────────────
# train_one_epoch / evaluate (smoke test)
# ──────────────────────────────────────────────────────────────────────────────
class TestTrainEval:
    def test_train_one_epoch_smoke(self, dummy_spec_df, small_model, device):
        small_model = small_model.to(device)
        ds = SpectrogramDataset(dummy_spec_df, transform=get_transforms(is_train=True))
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler()

        metrics = train_one_epoch(small_model, loader, criterion, optimizer, device, scaler)
        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_evaluate_smoke(self, dummy_spec_df, small_model, device):
        small_model = small_model.to(device)
        ds = SpectrogramDataset(dummy_spec_df, transform=get_transforms(is_train=False))
        loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(small_model, loader, criterion, device)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "predictions" in metrics
        assert "true_labels" in metrics
        assert "probabilities" in metrics
        assert len(metrics["predictions"]) == 16
        assert metrics["probabilities"].shape == (16, NUM_CLASSES)
