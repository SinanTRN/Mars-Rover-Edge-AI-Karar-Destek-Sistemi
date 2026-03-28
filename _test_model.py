"""Quick test for trained model."""
from pathlib import Path
from PIL import Image
from models.terrain_classifier import load_model

model = load_model()
print("Model basariyla yuklendi (egitilmis agirliklarla)")

img_dir = Path("data/ai4mars/images")
test_imgs = sorted(img_dir.glob("*"))[:5]
for img_path in test_imgs:
    img = Image.open(img_path)
    result = model.predict(img)
    cls = result["class"]
    conf = result["confidence"]
    print(f"  {img_path.name}: {cls} ({conf:.2%})")
