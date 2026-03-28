"""
Eğitilmiş modelin test seti üzerinde değerlendirilmesi.
Confusion matrix, per-class F1, precision, recall.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

import config
from models.terrain_classifier import load_model
from training.dataset import create_dataloaders


def evaluate(data_dir=None, model_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR / "ai4mars"

    model = load_model(model_path, device)

    _, val_loader = create_dataloaders(
        data_dir / "images", data_dir / "labels", batch_size=32)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("=" * 60)
    print("TERRAIN MODEL DEĞERLENDİRMESİ")
    print("=" * 60)
    print(classification_report(all_labels, all_preds,
                                labels=list(range(config.NUM_CLASSES)),
                                target_names=config.TERRAIN_CLASSES,
                                zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds,
                           labels=list(range(config.NUM_CLASSES))))


if __name__ == "__main__":
    evaluate()
