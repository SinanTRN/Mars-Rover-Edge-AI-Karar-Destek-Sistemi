"""Copy diverse demo images from ai4mars to sample_images for Streamlit UI."""
import shutil
import random
from pathlib import Path
from PIL import Image
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def main():
    src_images = Path("data/ai4mars/images")
    src_labels = Path("data/ai4mars/labels")
    dest = Path("data/sample_images")
    dest.mkdir(parents=True, exist_ok=True)

    # Group by dominant class
    class_map = {0: "soil", 1: "bedrock", 2: "sand", 3: "big_rock"}
    class_files = {0: [], 1: [], 2: [], 3: []}

    for lbl_path in sorted(src_labels.glob("*.png")):
        img_path = src_images / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            continue
        arr = np.array(Image.open(lbl_path))
        valid = arr[arr != 255]
        if len(valid) == 0:
            continue
        dominant = int(np.bincount(valid.flatten()).argmax())
        if dominant in class_files:
            class_files[dominant].append(img_path)

    # Pick 3 per class
    selected = []
    for cls_id, files in class_files.items():
        random.seed(42 + cls_id)
        n = min(3, len(files))
        chosen = random.sample(files, n)
        for i, f in enumerate(chosen):
            new_name = f"mars_{class_map[cls_id]}_{i+1}.jpg"
            dest_path = dest / new_name
            shutil.copy2(f, dest_path)
            selected.append((new_name, class_map[cls_id]))
            print(f"  {new_name} <- {f.name}")

    print(f"\n{len(selected)} demo goruntu kopyalandi -> {dest}")

if __name__ == "__main__":
    main()
