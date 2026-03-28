"""Test user's sample images from ornek/ folder."""
import os
from PIL import Image
from models.terrain_classifier import load_model

model = load_model()

print("=" * 85)
print("ORNEK KLASORU - MODEL TAHMiNLERi")
print("=" * 85)

ornek_dir = "ornek"
for f in sorted(os.listdir(ornek_dir)):
    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        path = os.path.join(ornek_dir, f)
        img = Image.open(path)
        orig_mode = img.mode
        orig_size = img.size
        img_rgb = img.convert("RGB")

        r = model.predict(img_rgb)
        p = r["probabilities"]

        print(f"\n  {f}")
        print(f"    Boyut: {orig_size}, Mod: {orig_mode}")
        print(f"    Tahmin: {r['class_tr']:10s} (guven: %{r['confidence']*100:.1f})")
        print(f"    Olasiliklar: toprak={p['soil']:.3f}  ana_kaya={p['bedrock']:.3f}  "
              f"kum={p['sand']:.3f}  buyuk_kaya={p['big_rock']:.3f}")

print("\n" + "=" * 85)
