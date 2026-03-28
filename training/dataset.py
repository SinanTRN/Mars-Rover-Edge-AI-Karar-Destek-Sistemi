"""
AI4Mars PyTorch Dataset sınıfı.
NASA AI4Mars dataset'ini yükler ve eğitim için hazırlar.
"""
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# AI4Mars etiket haritası (piksel değerleri → sınıf)
LABEL_MAP = {
    0: "soil",
    1: "bedrock",
    2: "sand",
    3: "big_rock",
    255: "ignore",
}


class AI4MarsDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform or self._default_transform()

        self.image_paths = sorted(self.images_dir.glob("*.jpg")) + \
                           sorted(self.images_dir.glob("*.jpeg")) + \
                           sorted(self.images_dir.glob("*.png")) + \
                           sorted(self.images_dir.glob("*.JPG"))
        self.label_paths = []
        self.valid_pairs = []

        for img_path in self.image_paths:
            label_path = self.labels_dir / (img_path.stem + ".png")
            if not label_path.exists():
                label_path = self.labels_dir / (img_path.stem + ".npy")
            if label_path.exists():
                self.valid_pairs.append((img_path, label_path))

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        img_path, label_path = self.valid_pairs[idx]

        # Grayscale olarak oku, sonra 3 kanala genişlet (inference ile tutarlı)
        image = Image.open(img_path).convert("L").convert("RGB")

        # Etiketi yükle ve baskın sınıfı bul (image-level)
        if label_path.suffix == ".npy":
            label_array = np.load(label_path)
        else:
            label_array = np.array(Image.open(label_path))

        # ignore (255) hariç en sık sınıfı bul
        valid_pixels = label_array[label_array != 255]
        if len(valid_pixels) > 0:
            label = int(np.bincount(valid_pixels.flatten()).argmax())
        else:
            label = 0  # fallback

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def _default_transform():
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def train_transform():
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def create_dataloaders(images_dir, labels_dir, batch_size=config.BATCH_SIZE,
                       val_split=0.2, use_fraction=1.0):
    # Eğitim ve validation için ayrı dataset nesneleri (transform paylaşım hatası önlenir)
    train_dataset = AI4MarsDataset(images_dir, labels_dir,
                                   transform=AI4MarsDataset.train_transform())
    val_dataset = AI4MarsDataset(images_dir, labels_dir,
                                 transform=AI4MarsDataset._default_transform())

    total = len(train_dataset)

    # Verinin sadece bir kısmını kullan
    if use_fraction < 1.0:
        subset_size = int(total * use_fraction)
        generator_sub = torch.Generator().manual_seed(42)
        all_indices = torch.randperm(total, generator=generator_sub)[:subset_size].tolist()
        total = subset_size
    else:
        all_indices = list(range(total))

    val_size = int(total * val_split)
    train_size = total - val_size

    train_idx = all_indices[:train_size]
    val_idx = all_indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_idx)
    val_subset = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    return train_loader, val_loader


def compute_class_weights(images_dir, labels_dir, max_samples=2000):
    """Sınıf dengesizliğini telafi eden ağırlıklar hesapla (örneklem ile)."""
    dataset = AI4MarsDataset(images_dir, labels_dir)
    counts = np.zeros(config.NUM_CLASSES)
    n = min(max_samples, len(dataset))
    indices = np.random.RandomState(42).permutation(len(dataset))[:n]
    for i in indices:
        _, label = dataset[i]
        counts[label] += 1
    # Inverse frequency weighting
    total = counts.sum()
    weights = total / (config.NUM_CLASSES * np.maximum(counts, 1))
    return torch.FloatTensor(weights), counts
