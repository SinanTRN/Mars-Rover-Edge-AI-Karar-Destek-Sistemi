"""Retry failed downloads - check which labels are missing and re-download."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

images_dir = Path("data/ai4mars/images")
labels_dir = Path("data/ai4mars/labels")

imgs = {f.stem for f in images_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg")}
lbls = {f.stem for f in labels_dir.glob("*.png")}

paired = imgs & lbls
imgs_only = imgs - lbls
lbls_only = lbls - imgs

print(f"Goruntu sayisi: {len(imgs)}")
print(f"Etiket sayisi:  {len(lbls)}")
print(f"Eslesenler:     {len(paired)}")
print(f"Etiketsiz img:  {len(imgs_only)}")
print(f"Gorselsiz lbl:  {len(lbls_only)}")

# Class distribution
import numpy as np
from PIL import Image

class_names = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
counts = {n: 0 for n in class_names.values()}

for stem in sorted(paired):
    lbl = np.array(Image.open(labels_dir / f"{stem}.png"))
    valid = lbl[lbl < 255]
    if len(valid) > 0:
        dom = np.bincount(valid.flatten(), minlength=4).argmax()
        counts[class_names[dom]] += 1

print(f"\nSinif dagilimi ({len(paired)} goruntu):")
for name, cnt in counts.items():
    print(f"  {name:10s}: {cnt}")
