import os
import torch
from pathlib import Path
from src.phase4_train import build_model
from src.config import MODELS_DIR, PROJECT_ROOT

def main():
    print("Initializing dummy model...")
    # Instantiate the model with pretrained=False
    model = build_model(pretrained=False)
    model.eval()

    # Save to output/models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path1 = MODELS_DIR / "best_model.pt"
    print(f"Saving dummy model weights to {path1}...")
    torch.save(model.state_dict(), str(path1))

    # Also save to models/
    models_root = PROJECT_ROOT / "models"
    models_root.mkdir(parents=True, exist_ok=True)
    path2 = models_root / "best_model.pt"
    print(f"Saving dummy model weights to {path2}...")
    torch.save(model.state_dict(), str(path2))

    print("Done! Dummy model created successfully.")

if __name__ == "__main__":
    main()
