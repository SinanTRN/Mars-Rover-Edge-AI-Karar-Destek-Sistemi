"""
NASA REMS (Rover Environmental Monitoring Station) veri indirme scripti.
Curiosity rover — Mars yüzey hava durumu verileri.
"""
import sys
from pathlib import Path
import requests
import pandas as pd
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# NASA REMS veri kaynağı — Mars Weather Service API
REMS_API_URL = "https://mars.nasa.gov/rss/api/?feed=weather&category=msl&feedtype=json"


def download_rems():
    dest_dir = config.DATA_DIR / "rems"
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_path = dest_dir / "rems_data.csv"

    if output_path.exists():
        print(f"REMS verisi zaten mevcut: {output_path}")
        return

    print("=" * 60)
    print("NASA REMS Veri İndirme")
    print("=" * 60)

    try:
        print("Mars Weather API'den veri çekiliyor...")
        resp = requests.get(REMS_API_URL, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # JSON → DataFrame dönüşümü
        sols = data.get("soles", [])
        if sols:
            df = pd.DataFrame(sols)
            # Sütun isimlerini düzenle
            rename_map = {
                "sol": "sol",
                "min_temp": "air_temp",
                "max_temp": "ground_temp",
                "pressure": "pressure",
                "abs_humidity": "humidity",
                "atmo_opacity": "opacity",
            }
            existing_renames = {k: v for k, v in rename_map.items() if k in df.columns}
            df = df.rename(columns=existing_renames)
            df.to_csv(output_path, index=False)
            print(f"Kaydedildi: {output_path} ({len(df)} sol)")
            return
    except Exception as e:
        print(f"API'den indirme başarısız: {e}")

    # Fallback: sample_rems.csv zaten projede var
    print("Fallback: Örnek REMS verisi kullanılacak (data/sample_rems.csv)")


if __name__ == "__main__":
    download_rems()
