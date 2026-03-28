"""
AI4Mars dataset indirme scripti — Zenodo v0.6
HTTP range request ile sadece gerekli dosyaları çeker (16 GB yerine ~1-2 GB).
Fallback: Tam ZIP indirme.
"""
import os
import sys
import struct
import zlib
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DATASET_URL = "https://zenodo.org/records/15995036/files/ai4mars-dataset-merged-0.6.zip"
MAX_IMAGES_PER_ROVER = 1500


class RemoteZipReader:
    """HTTP Range Request ile uzaktaki ZIP'ten seçici dosya çıkarma."""

    def __init__(self, url):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "AI4Mars-Downloader/1.0"})
        resp = self.session.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        self.size = int(resp.headers.get("Content-Length", 0))
        self.url = resp.url
        # Range desteğini test et
        test = self.session.get(self.url, headers={"Range": "bytes=0-3"}, timeout=30, stream=True)
        self.accept_ranges = test.status_code == 206
        test.close()

    def _read_range(self, start, end):
        headers = {"Range": f"bytes={start}-{end}"}
        resp = self.session.get(self.url, headers=headers, timeout=120)
        if resp.status_code not in (200, 206):
            raise RuntimeError(f"Range request başarısız: {resp.status_code}")
        return resp.content

    def list_entries(self):
        """ZIP central directory'den dosya listesini oku."""
        tail_size = min(65536 * 4, self.size)
        tail_data = self._read_range(self.size - tail_size, self.size - 1)

        # End of Central Directory bul
        eocd_sig = b"\x50\x4b\x05\x06"
        eocd_idx = tail_data.rfind(eocd_sig)
        zip64_locator_sig = b"\x50\x4b\x06\x07"
        loc_idx = tail_data.rfind(zip64_locator_sig)

        if loc_idx >= 0:
            locator = tail_data[loc_idx:loc_idx + 20]
            eocd64_abs = struct.unpack_from("<Q", locator, 8)[0]
            eocd64_data = self._read_range(eocd64_abs, eocd64_abs + 76)
            cd_size = struct.unpack_from("<Q", eocd64_data, 40)[0]
            cd_offset = struct.unpack_from("<Q", eocd64_data, 48)[0]
        elif eocd_idx >= 0:
            eocd = tail_data[eocd_idx:]
            cd_size = struct.unpack_from("<I", eocd, 12)[0]
            cd_offset = struct.unpack_from("<I", eocd, 16)[0]
            if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
                raise RuntimeError("ZIP64 gerekli ama locator bulunamadı")
        else:
            raise RuntimeError("ZIP EOCD bulunamadı")

        print(f"Central directory okunuyor ({cd_size / 1024 / 1024:.1f} MB)...")
        cd_data = self._read_range(cd_offset, cd_offset + cd_size - 1)

        entries = []
        pos = 0
        while pos < len(cd_data) - 46:
            if cd_data[pos:pos + 4] != b"\x50\x4b\x01\x02":
                break

            comp = struct.unpack_from("<H", cd_data, pos + 10)[0]
            c_size = struct.unpack_from("<I", cd_data, pos + 20)[0]
            u_size = struct.unpack_from("<I", cd_data, pos + 24)[0]
            fn_len = struct.unpack_from("<H", cd_data, pos + 28)[0]
            ex_len = struct.unpack_from("<H", cd_data, pos + 30)[0]
            cm_len = struct.unpack_from("<H", cd_data, pos + 32)[0]
            offset = struct.unpack_from("<I", cd_data, pos + 42)[0]

            filename = cd_data[pos + 46:pos + 46 + fn_len].decode("utf-8", errors="replace")

            # ZIP64 extra field
            if c_size == 0xFFFFFFFF or u_size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
                extra = cd_data[pos + 46 + fn_len:pos + 46 + fn_len + ex_len]
                ep = 0
                while ep < len(extra) - 4:
                    tag = struct.unpack_from("<H", extra, ep)[0]
                    sz = struct.unpack_from("<H", extra, ep + 2)[0]
                    if tag == 0x0001:
                        fp = ep + 4
                        if u_size == 0xFFFFFFFF:
                            u_size = struct.unpack_from("<Q", extra, fp)[0]; fp += 8
                        if c_size == 0xFFFFFFFF:
                            c_size = struct.unpack_from("<Q", extra, fp)[0]; fp += 8
                        if offset == 0xFFFFFFFF:
                            offset = struct.unpack_from("<Q", extra, fp)[0]
                        break
                    ep += 4 + sz

            entries.append({
                "filename": filename,
                "compressed_size": c_size,
                "uncompressed_size": u_size,
                "offset": offset,
                "compression": comp,
            })
            pos += 46 + fn_len + ex_len + cm_len

        return entries

    def extract_file(self, entry, dest_path):
        """Tek dosyayı range request ile indir."""
        if dest_path.exists() and dest_path.stat().st_size > 0:
            return True
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        header = self._read_range(entry["offset"], entry["offset"] + 29)
        fn_len = struct.unpack_from("<H", header, 26)[0]
        ex_len = struct.unpack_from("<H", header, 28)[0]
        data_start = entry["offset"] + 30 + fn_len + ex_len

        if entry["compressed_size"] == 0:
            dest_path.write_bytes(b"")
            return True

        raw = self._read_range(data_start, data_start + entry["compressed_size"] - 1)

        if entry["compression"] == 0:
            dest_path.write_bytes(raw)
        elif entry["compression"] == 8:
            dest_path.write_bytes(zlib.decompress(raw, -zlib.MAX_WBITS))
        else:
            return False
        return True


def smart_download(dest_dir, max_per_rover=MAX_IMAGES_PER_ROVER):
    """Akıllı indirme: ZIP'ten sadece gerekli dosyaları çeker."""
    images_dir = dest_dir / "images"
    labels_dir = dest_dir / "labels"

    if images_dir.exists() and len(list(images_dir.glob("*"))) > 100:
        n = len(list(images_dir.glob("*")))
        print(f"Veri zaten mevcut: {n} görüntü")
        return True

    print("=" * 60)
    print("AI4Mars Dataset — Akıllı İndirme (Range Request)")
    print(f"Kaynak: Zenodo v0.6")
    print("=" * 60)

    reader = RemoteZipReader(DATASET_URL)
    if not reader.accept_ranges:
        print("Range request desteklenmiyor!")
        return False

    print(f"ZIP boyutu: {reader.size / 1e9:.1f} GB")
    entries = reader.list_entries()
    print(f"ZIP içinde {len(entries)} dosya/klasör")

    # Dosyaları sınıflandır
    img_entries = {}
    lbl_entries = {}

    for e in entries:
        fn = e["filename"]
        if fn.endswith("/") or e["uncompressed_size"] == 0:
            continue
        name = Path(fn).name
        stem = Path(fn).stem

        # images klasöründe mi?
        lower = fn.lower()
        if "/images/" in lower and (name.endswith(".jpeg") or name.endswith(".jpg") or name.endswith(".png")):
            # Rover tespiti
            rover = "msl"
            if "m2020" in lower or "m20" in lower:
                rover = "m2020"
            elif "mer" in lower:
                rover = "mer"
            img_entries.setdefault(rover, {})[stem] = e
        elif "/labels/" in lower and name.endswith(".png"):
            # Label: train/ veya test/ altında olabilir
            lbl_entries[stem] = e

    for rover, imgs in img_entries.items():
        print(f"  {rover}: {len(imgs)} görüntü")
    print(f"  Etiket: {len(lbl_entries)} toplam")

    # Çiftleri eşleştir
    paired = []
    for rover in ["msl", "m2020", "mer"]:
        imgs = img_entries.get(rover, {})
        count = 0
        for stem, img_e in imgs.items():
            if stem in lbl_entries and count < max_per_rover:
                paired.append((img_e, lbl_entries[stem], rover))
                count += 1
        print(f"  {rover}: {count} eşleşen çift")

    if not paired:
        print("\nEşleşen çift bulunamadı! Sadece görüntü indiriliyor...")
        # Eşleşme yoksa doğrudan image indir
        all_imgs = []
        for rover_imgs in img_entries.values():
            all_imgs.extend(list(rover_imgs.values())[:max_per_rover])
        paired = [(e, None, "any") for e in all_imgs[:max_per_rover * 3]]

    print(f"\nToplam {len(paired)} çift indiriliyor...")
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0
    pbar = tqdm(paired, desc="İndiriliyor", unit="çift")

    for item in pbar:
        if len(item) == 3:
            img_e, lbl_e, rover = item
        else:
            img_e, lbl_e = item
        try:
            img_name = Path(img_e["filename"]).name
            img_dest = images_dir / img_name
            if reader.extract_file(img_e, img_dest):
                if lbl_e is not None:
                    lbl_dest = labels_dir / (Path(img_name).stem + ".png")
                    reader.extract_file(lbl_e, lbl_dest)
                success += 1
            pbar.set_postfix(ok=success, err=errors)
        except Exception as ex:
            errors += 1
            pbar.set_postfix(ok=success, err=errors)
            continue

    pbar.close()

    img_count = len(list(images_dir.glob("*")))
    lbl_count = len(list(labels_dir.glob("*")))
    print(f"\nTamamlandı! {img_count} görüntü, {lbl_count} etiket")
    print(f"Başarılı: {success}, Hata: {errors}")
    return img_count > 0


def main():
    dest_dir = config.DATA_DIR / "ai4mars"
    dest_dir.mkdir(parents=True, exist_ok=True)

    ok = smart_download(dest_dir)
    if not ok:
        print("\nİndirme başarısız.")
        print(f"Manuel indirme: {DATASET_URL}")
        print(f"ZIP'i çıkarıp şuraya koyun: {dest_dir}")
        sys.exit(1)

    print(f"\nVeri konumu: {dest_dir}")
    print("Eğitim başlatmak için:")
    print("  python training/train_terrain.py")


if __name__ == "__main__":
    main()
