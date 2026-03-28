"""
Kaggle AI4Mars verisini ai4mars klasörüne kopyala.
EDR görüntüleri ve eşleşen train label'ları eşleştirip kopyalar.
"""
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from collections import Counter

SRC = Path("data/ai4mars-dataset-merged-0.1")
DST = Path("data/ai4mars")

img_src = SRC / "msl" / "images" / "edr"
lbl_src = SRC / "msl" / "labels" / "train"
img_dst = DST / "images"
lbl_dst = DST / "labels"

# Temizle (eski verileri sil)
if img_dst.exists():
    shutil.rmtree(img_dst)
if lbl_dst.exists():
    shutil.rmtree(lbl_dst)

img_dst.mkdir(parents=True, exist_ok=True)
lbl_dst.mkdir(parents=True, exist_ok=True)

# Label dosyalarının stem'lerini al
label_stems = {f.stem: f for f in lbl_src.glob("*.png")}
print(f"Label sayisi: {len(label_stems)}")

# EDR görüntülerinden label'ı olanları eşleştir
paired = 0
skipped = 0

for img_file in sorted(img_src.iterdir()):
    if img_file.suffix.upper() not in (".JPG", ".JPEG", ".PNG"):
        continue
    stem = img_file.stem
    if stem in label_stems:
        shutil.copy2(img_file, img_dst / (stem + ".jpg"))
        shutil.copy2(label_stems[stem], lbl_dst / (stem + ".png"))
        paired += 1
    else:
        skipped += 1

print(f"Eslesen cift: {paired}")
print(f"Etiketsiz (atlanan): {skipped}")

# Sınıf dağılımı
class_names = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
counts = Counter()

for lbl_file in sorted(lbl_dst.glob("*.png")):
    lbl = np.array(Image.open(lbl_file))
    valid = lbl[lbl < 255]
    if len(valid) > 0:
        dom = int(np.bincount(valid.flatten(), minlength=4).argmax())
        counts[class_names[dom]] += 1

print(f"\nSinif dagilimi ({paired} goruntu):")
for name in ["soil", "bedrock", "sand", "big_rock"]:
    print(f"  {name:10s}: {counts[name]}")
