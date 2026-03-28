"""
AI4Mars terrain modeli eğitim scripti.
Kullanım: python training/train_terrain.py --epochs 15 --batch-size 32
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import config
from models.terrain_classifier import TerrainClassifier
from training.dataset import create_dataloaders, compute_class_weights


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    # Data
    images_dir = Path(args.data_dir) / "images"
    labels_dir = Path(args.data_dir) / "labels"

    if not images_dir.exists():
        print(f"HATA: {images_dir} bulunamadı. Önce veri indirin:")
        print("  python data/download_ai4mars.py")
        sys.exit(1)

    # Sınıf ağırlıkları hesapla (dengesizliği telafi)
    class_weights, class_counts = compute_class_weights(images_dir, labels_dir)
    print(f"Sınıf dağılımı: {dict(zip(config.TERRAIN_CLASSES, class_counts.astype(int).tolist()))}")
    print(f"Sınıf ağırlıkları: {dict(zip(config.TERRAIN_CLASSES, [f'{w:.2f}' for w in class_weights]))}")

    train_loader, val_loader = create_dataloaders(
        images_dir, labels_dir, batch_size=args.batch_size,
        use_fraction=args.fraction)

    print(f"Eğitim: {len(train_loader.dataset)} görüntü")
    print(f"Doğrulama: {len(val_loader.dataset)} görüntü")

    # Model
    model = TerrainClassifier(num_classes=config.NUM_CLASSES).to(device)

    # Overfitting önleme: classifier katmanına dropout ekle
    model.backbone.classifier = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(1280, config.NUM_CLASSES),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

    best_acc = 0.0
    config.PRETRAINED_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        train_acc = 100 * correct / total

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)

        print(f"  → Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}% | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  ✓ En iyi model kaydedildi ({val_acc:.1f}%)")

    print(f"\nEğitim tamamlandı. En iyi doğrulama: {best_acc:.1f}%")
    print(f"Model: {config.MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4Mars Terrain Eğitimi")
    parser.add_argument("--data-dir", type=str,
                        default=str(config.DATA_DIR / "ai4mars"))
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Verinin yuzde kaci kullanilacak (0.3 = %%30)")
    args = parser.parse_args()
    train(args)
