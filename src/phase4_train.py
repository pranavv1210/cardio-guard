"""
Phase 4 — CNN Training (EfficientNet-B0 on local GTX 1650)
Two-stage transfer learning with spectrogram augmentation.
"""
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from PIL import Image

from src.config import (
    BATCH_SIZE, NUM_WORKERS, MODEL_NAME, PRETRAINED, DROPOUT_RATE, NUM_CLASSES,
    STAGE1_EPOCHS, STAGE1_LR, STAGE2_EPOCHS, STAGE2_LR,
    SPEC_AUG_FREQ_MASK, SPEC_AUG_TIME_MASK,
    SPEC_HEIGHT, SPEC_WIDTH, RANDOM_SEED,
    SPECTROGRAMS_DIR, MODELS_DIR, OUTPUT_DIR,
)

logger = logging.getLogger(__name__)

# Reproducibility
def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SpecAugment:
    """Apply SpecAugment-style time/frequency masking to spectrograms."""

    def __init__(self, freq_mask: int = SPEC_AUG_FREQ_MASK, time_mask: int = SPEC_AUG_TIME_MASK):
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply masks to a (C, H, W) tensor."""
        _, h, w = img.shape

        # Frequency masking (horizontal band)
        if self.freq_mask > 0:
            f = random.randint(0, min(self.freq_mask, h - 1))
            f0 = random.randint(0, h - f)
            img[:, f0:f0 + f, :] = 0

        # Time masking (vertical band)
        if self.time_mask > 0:
            t = random.randint(0, min(self.time_mask, w - 1))
            t0 = random.randint(0, w - t)
            img[:, :, t0:t0 + t] = 0

        return img


class SpectrogramDataset(Dataset):
    """PyTorch Dataset for spectrogram images."""

    def __init__(self, df: pd.DataFrame, transform=None):
        """
        Args:
            df: DataFrame with 'spectrogram_path' and 'class_idx' columns.
            transform: Optional torchvision transforms.
        """
        self.paths = df["spectrogram_path"].tolist()
        self.labels = df["class_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """Get image transforms for training or evaluation.

    Args:
        is_train: If True, includes augmentation.

    Returns:
        torchvision Compose transform.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((SPEC_HEIGHT, SPEC_WIDTH)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            SpecAugment(),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((SPEC_HEIGHT, SPEC_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def build_model(
    model_name: str = MODEL_NAME,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = PRETRAINED,
    dropout: float = DROPOUT_RATE,
) -> nn.Module:
    """Build EfficientNet model with custom classifier head.

    Args:
        model_name: timm model name.
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate before classifier.

    Returns:
        nn.Module model.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    num_features = model.num_features

    model.classifier = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(num_features, num_classes),
    )

    return model


def get_weighted_sampler(labels: list) -> WeightedRandomSampler:
    """Create a weighted random sampler to handle class imbalance.

    Args:
        labels: List of class indices.

    Returns:
        WeightedRandomSampler for DataLoader.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels_arr]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


def get_class_weights(labels: list, device: torch.device) -> torch.Tensor:
    """Compute class weights for loss function.

    Args:
        labels: List of class indices.
        device: torch device.

    Returns:
        Tensor of class weights.
    """
    labels_arr = np.array(labels)
    class_counts = np.bincount(labels_arr)
    total = len(labels_arr)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights).to(device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> dict:
    """Train model for one epoch.

    Returns:
        Dict with 'loss' and 'accuracy'.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset.

    Returns:
        Dict with 'loss', 'accuracy', 'predictions', 'true_labels', 'probabilities'.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "predictions": np.array(all_preds),
        "true_labels": np.array(all_labels),
        "probabilities": np.array(all_probs),
    }


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_last_blocks(model: nn.Module, num_blocks: int = 2) -> None:
    """Unfreeze the last N blocks of EfficientNet for fine-tuning."""
    # Unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Unfreeze last blocks
    blocks = list(model.blocks) if hasattr(model, 'blocks') else []
    for block in blocks[-num_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def run_phase4(spec_csv: Optional[str] = None) -> dict:
    """Execute the full Phase 4 training pipeline.

    Args:
        spec_csv: Path to spectrograms_metadata.csv. Auto-detected if None.

    Returns:
        Dict with training history and best metrics.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    set_seed()

    logger.info("=" * 60)
    logger.info("PHASE 4 — CNN Training (EfficientNet-B0)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load metadata
    if spec_csv is None:
        spec_csv = str(OUTPUT_DIR / "spectrograms_metadata.csv")

    df = pd.read_csv(spec_csv)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    logger.info("Train: %d | Val: %d", len(train_df), len(val_df))

    # Datasets & DataLoaders
    train_dataset = SpectrogramDataset(train_df, transform=get_transforms(is_train=True))
    val_dataset = SpectrogramDataset(val_df, transform=get_transforms(is_train=False))

    sampler = get_weighted_sampler(train_df["class_idx"].tolist())

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )

    # Model
    model = build_model()
    model = model.to(device)

    class_weights = get_class_weights(train_df["class_idx"].tolist(), device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    history = {"stage": [], "epoch": [], "train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # ─── Stage 1: Frozen backbone ─────────────────────────────────
    logger.info("\n--- Stage 1: Train classifier head (frozen backbone) ---")
    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE1_LR,
    )

    for epoch in range(STAGE1_EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)

        logger.info(
            "S1 Epoch %d/%d — Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f",
            epoch + 1, STAGE1_EPOCHS,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
        )

        history["stage"].append(1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), str(MODELS_DIR / "best_model.pt"))
            logger.info("  → New best val accuracy: %.4f", best_val_acc)

    # ─── Stage 2: Fine-tune last blocks ───────────────────────────
    logger.info("\n--- Stage 2: Fine-tune last blocks ---")
    unfreeze_last_blocks(model, num_blocks=2)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=STAGE2_LR,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=STAGE2_EPOCHS,
    )

    for epoch in range(STAGE2_EPOCHS):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            "S2 Epoch %d/%d — Train Loss: %.4f Acc: %.4f | Val Loss: %.4f Acc: %.4f",
            epoch + 1, STAGE2_EPOCHS,
            train_metrics["loss"], train_metrics["accuracy"],
            val_metrics["loss"], val_metrics["accuracy"],
        )

        history["stage"].append(2)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), str(MODELS_DIR / "best_model.pt"))
            logger.info("  → New best val accuracy: %.4f", best_val_acc)

    # Save final model and history
    torch.save(model.state_dict(), str(MODELS_DIR / "final_model.pt"))
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(str(OUTPUT_DIR / "training_history.csv"), index=False)

    logger.info("\nTraining complete. Best val accuracy: %.4f", best_val_acc)
    logger.info("Models saved to: %s", MODELS_DIR)

    return {"best_val_acc": best_val_acc, "history": history}


if __name__ == "__main__":
    run_phase4()
