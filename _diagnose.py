"""Diagnose overfitting - check predictions on all images."""
import os
from PIL import Image
from models.terrain_classifier import load_model

model = load_model()

print("=" * 80)
print("DEMO IMAGES PREDICTIONS")
print("=" * 80)

demo_dir = "data/sample_images"
for f in sorted(os.listdir(demo_dir)):
    if f.endswith((".jpg", ".jpeg", ".png")):
        img = Image.open(os.path.join(demo_dir, f)).convert("RGB")
        r = model.predict(img)
        p = r["probabilities"]
        print(f"  {f:30s} -> {r['class_tr']:10s} ({r['confidence']*100:.1f}%)  "
              f"soil={p['soil']:.3f} bed={p['bedrock']:.3f} sand={p['sand']:.3f} rock={p['big_rock']:.3f}")

print()
print("=" * 80)
print("TRAINING SET CLASS DISTRIBUTION")
print("=" * 80)

import numpy as np
label_dir = "data/ai4mars/labels"
class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
class_names = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
dominant_counts = {"soil": 0, "bedrock": 0, "sand": 0, "big_rock": 0}

for f in sorted(os.listdir(label_dir)):
    if f.endswith(".png"):
        lbl = np.array(Image.open(os.path.join(label_dir, f)))
        valid = lbl[lbl < 255]
        if len(valid) == 0:
            continue
        for c in range(4):
            class_counts[c] += (valid == c).sum()
        dominant = np.bincount(valid, minlength=4).argmax()
        dominant_counts[class_names[dominant]] += 1

total_px = sum(class_counts.values())
print("  Pixel distribution:")
for c in range(4):
    pct = class_counts[c] / total_px * 100 if total_px > 0 else 0
    print(f"    {class_names[c]:10s}: {class_counts[c]:>10d} px ({pct:.1f}%)")

print("  Dominant class per image:")
for name, cnt in dominant_counts.items():
    print(f"    {name:10s}: {cnt} images")

print()
print("=" * 80)
print("RANDOM AI4MARS IMAGES PREDICTIONS (to check diversity)")
print("=" * 80)

import random
img_dir = "data/ai4mars/images"
all_imgs = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".JPG"))]
random.seed(42)
sample = random.sample(all_imgs, min(20, len(all_imgs)))

for f in sorted(sample):
    img = Image.open(os.path.join(img_dir, f)).convert("RGB")
    r = model.predict(img)
    p = r["probabilities"]
    print(f"  {f[:40]:40s} -> {r['class_tr']:10s} ({r['confidence']*100:.1f}%)  "
          f"soil={p['soil']:.3f} bed={p['bedrock']:.3f} sand={p['sand']:.3f} rock={p['big_rock']:.3f}")
