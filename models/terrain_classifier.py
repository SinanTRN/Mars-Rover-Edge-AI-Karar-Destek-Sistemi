"""
MobileNetV2 tabanlı Mars terrain sınıflandırıcı.
Sınıflar: soil, bedrock, sand, big_rock
Edge AI uyumlu: ~3.4M parametre, ~7MB
"""
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1280, num_classes),
        )
        self.classes = config.TERRAIN_CLASSES
        self.classes_tr = config.TERRAIN_CLASSES_TR

        self.transform = transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        return self.backbone(x)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        # Model gri tonlamalı Mars görüntüleriyle eğitildi.
        # Renkli görüntü gelirse önce tek kanala düşür, sonra 3 kanala geri aç.
        gray = image.convert("L")
        rgb_from_gray = gray.convert("RGB")
        return self.transform(rgb_from_gray).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image: Image.Image, device="cpu"):
        self.eval()
        self.to(device)
        tensor = self.preprocess(image).to(device)
        logits = self.forward(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        idx = probs.argmax().item()
        return {
            "class": self.classes[idx],
            "class_tr": self.classes_tr[idx],
            "confidence": round(probs[idx].item(), 4),
            "probabilities": {c: round(p.item(), 4) for c, p in zip(self.classes, probs)},
        }

    @torch.no_grad()
    def predict_with_probabilities(self, image: Image.Image, device="cpu"):
        result = self.predict(image, device)
        return result["probabilities"]


def load_model(model_path=None, device="cpu"):
    model = TerrainClassifier()
    path = Path(model_path) if model_path else config.MODEL_PATH
    if path.exists():
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model
