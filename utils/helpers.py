"""
Genel yardımcı fonksiyonlar.
"""
import numpy as np
from PIL import Image
from pathlib import Path


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def get_sample_images(sample_dir: Path) -> list:
    if not sample_dir.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted([f for f in sample_dir.iterdir()
                   if f.suffix.lower() in exts])


def format_risk_emoji(zone: str) -> str:
    mapping = {
        "güvenli bölge": "🟢",
        "incelenmeli": "🟡",
        "kritik bölge": "🔴",
    }
    return mapping.get(zone, "⚪")


def format_priority_emoji(priority: str) -> str:
    mapping = {
        "düşük": "🔽",
        "orta": "➡️",
        "yüksek": "🔼",
    }
    return mapping.get(priority, "➡️")
