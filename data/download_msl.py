"""
AI4Mars v0.6 — MSL Ncam veri çekme (HTTP range request).
Sadece EDR görüntüleri ve eşleşen label'ları indirir.
"""
import sys
import struct
import zlib
from pathlib import Path
import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

DATASET_URL = "https://zenodo.org/records/15995036/files/ai4mars-dataset-merged-0.6.zip"
MAX_PAIRS = 5000  # Kaç çift indirilecek


def read_range(session, url, start, end):
    resp = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    if resp.status_code not in (200, 206):
        raise RuntimeError(f"HTTP {resp.status_code}")
    return resp.content


def read_central_directory(session, url, size):
    """Tüm central directory'yi oku ve dosya listesini döndür."""
    # EOCD bul
    tail = read_range(session, url, size - 256 * 1024, size - 1)

    loc_sig = b"\x50\x4b\x06\x07"
    loc_idx = tail.rfind(loc_sig)
    if loc_idx >= 0:
        eocd64_abs = struct.unpack_from("<Q", tail, loc_idx + 8)[0]
        eocd64 = read_range(session, url, eocd64_abs, eocd64_abs + 76)
        cd_size = struct.unpack_from("<Q", eocd64, 40)[0]
        cd_offset = struct.unpack_from("<Q", eocd64, 48)[0]
    else:
        sig = b"\x50\x4b\x05\x06"
        eidx = tail.rfind(sig)
        if eidx < 0:
            raise RuntimeError("EOCD bulunamadi")
        cd_size = struct.unpack_from("<I", tail, eidx + 12)[0]
        cd_offset = struct.unpack_from("<I", tail, eidx + 16)[0]

    print(f"Central directory: {cd_size / 1024 / 1024:.1f} MB (offset {cd_offset})")

    # CD'yi parça parça indir
    cd_data = bytearray()
    chunk = 2 * 1024 * 1024  # 2MB chunks
    pbar = tqdm(total=cd_size, desc="CD okunuyor", unit="B", unit_scale=True)
    pos = 0
    while pos < cd_size:
        end = min(pos + chunk - 1, cd_size - 1)
        data = read_range(session, url, cd_offset + pos, cd_offset + end)
        cd_data.extend(data)
        pbar.update(len(data))
        pos += chunk
    pbar.close()

    # Parse
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

        # ZIP64 extra
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

        entries.append((filename, c_size, u_size, offset, comp))
        pos += 46 + fn_len + ex_len + cm_len

    return entries


def extract_entry(session, url, entry, dest_path):
    """Tek dosyayı range request ile çıkar."""
    filename, c_size, u_size, offset, comp = entry
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Local file header
    header = read_range(session, url, offset, offset + 29)
    fn_len = struct.unpack_from("<H", header, 26)[0]
    ex_len = struct.unpack_from("<H", header, 28)[0]
    data_start = offset + 30 + fn_len + ex_len

    if c_size == 0:
        dest_path.write_bytes(b"")
        return True

    raw = read_range(session, url, data_start, data_start + c_size - 1)

    if comp == 0:
        dest_path.write_bytes(raw)
    elif comp == 8:
        dest_path.write_bytes(zlib.decompress(raw, -zlib.MAX_WBITS))
    else:
        return False
    return True


def main():
    dest_dir = config.DATA_DIR / "ai4mars"
    images_dir = dest_dir / "images"
    labels_dir = dest_dir / "labels"

    if images_dir.exists() and len(list(images_dir.glob("*"))) > 1000:
        n = len(list(images_dir.glob("*")))
        print(f"Veri zaten mevcut: {n} görüntü. Atlanıyor.")
        return

    print("=" * 60)
    print("AI4Mars v0.6 — MSL EDR Dataset İndirme")
    print("=" * 60)

    session = requests.Session()
    session.headers.update({"User-Agent": "AI4Mars-Downloader/1.0"})

    resp = session.head(DATASET_URL, allow_redirects=True, timeout=30)
    resp.raise_for_status()
    file_size = int(resp.headers["Content-Length"])
    final_url = resp.url
    print(f"ZIP: {file_size / 1e9:.1f} GB")

    # Central directory oku
    entries = read_central_directory(session, final_url, file_size)
    print(f"Toplam {len(entries)} dosya/klasör")

    # ─── MSL Ncam EDR images & labels eşleştir ─────────────
    # Image: msl/ncam/images/edr/NLB_*EDR_*.JPG
    # Label: msl/ncam/labels/train/NLB_*EDR_*.png  (same stem)
    msl_edr_images = {}
    msl_ncam_labels = {}

    # MCAM images & labels
    msl_mcam_images = {}
    msl_mcam_labels = {}

    # MER images & labels
    mer_images = {}
    mer_labels = {}

    for e in entries:
        fn = e[0]
        u_size = e[2]
        if fn.endswith("/") or u_size == 0:
            continue

        lower = fn.lower()

        # MSL Ncam EDR
        if "msl/ncam/images/edr/" in lower and (fn.endswith(".JPG") or fn.endswith(".jpg") or fn.endswith(".jpeg")):
            stem = Path(fn).stem
            msl_edr_images[stem] = e
        elif "msl/ncam/labels/train/" in lower and fn.endswith(".png"):
            stem = Path(fn).stem
            msl_ncam_labels[stem] = e

        # MSL Mcam
        elif "msl/mcam/images/" in lower and not fn.endswith("/"):
            stem = Path(fn).stem
            msl_mcam_images[stem] = e
        elif "msl/mcam/labels/train/" in lower and fn.endswith(".png"):
            stem = Path(fn).stem.replace("_merged", "")
            msl_mcam_labels[stem] = e

        # MER
        elif "mer/" in lower and "/images/" in lower and not fn.endswith("/"):
            stem = Path(fn).stem
            mer_images[stem] = e
        elif "mer/" in lower and "/labels/" in lower and fn.endswith(".png"):
            stem = Path(fn).stem
            # Remove _merged suffix
            if "_merged" in stem:
                stem = stem[:stem.rfind("_merged")]
            mer_labels[stem] = e

    print(f"\nMSL Ncam EDR: {len(msl_edr_images)} img, {len(msl_ncam_labels)} lbl")
    print(f"MSL Mcam:     {len(msl_mcam_images)} img, {len(msl_mcam_labels)} lbl")
    print(f"MER:          {len(mer_images)} img, {len(mer_labels)} lbl")

    # ─── Eşleştir ─────────────
    paired = []

    # MSL Ncam (en büyük / en iyi kalite)
    for stem, img_e in msl_edr_images.items():
        if stem in msl_ncam_labels:
            paired.append((img_e, msl_ncam_labels[stem]))

    print(f"\nMSL Ncam eşleşen: {len(paired)}")

    # MSL Mcam ekle
    for stem, img_e in msl_mcam_images.items():
        if stem in msl_mcam_labels and len(paired) < MAX_PAIRS:
            paired.append((img_e, msl_mcam_labels[stem]))
    print(f"MSL Mcam eklendi, toplam: {len(paired)}")

    # Yetmezse MER ekle
    if len(paired) < MAX_PAIRS:
        for stem, img_e in mer_images.items():
            if stem in mer_labels and len(paired) < MAX_PAIRS:
                paired.append((img_e, mer_labels[stem]))
        print(f"MER eklendi, toplam: {len(paired)}")

    # Max limit
    paired = paired[:MAX_PAIRS]
    print(f"\nToplam {len(paired)} çift indirilecek...")

    # ─── İndir ─────────────
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    errors = 0
    pbar = tqdm(paired, desc="İndiriliyor", unit="çift")

    for img_e, lbl_e in pbar:
        try:
            img_name = Path(img_e[0]).stem + ".jpg"
            lbl_name = Path(img_e[0]).stem + ".png"

            img_dest = images_dir / img_name
            lbl_dest = labels_dir / lbl_name

            ok1 = extract_entry(session, final_url, img_e, img_dest)
            ok2 = extract_entry(session, final_url, lbl_e, lbl_dest)

            if ok1 and ok2:
                success += 1
            else:
                errors += 1

            pbar.set_postfix(ok=success, err=errors)
        except Exception as ex:
            errors += 1
            pbar.set_postfix(ok=success, err=errors)
            continue

    pbar.close()

    img_count = len(list(images_dir.glob("*")))
    lbl_count = len(list(labels_dir.glob("*")))
    print(f"\n{'=' * 60}")
    print(f"Tamamlandı! {img_count} görüntü, {lbl_count} etiket")
    print(f"Başarılı: {success}, Hata: {errors}")
    print(f"Konum: {dest_dir}")
    print(f"\nEğitim başlatmak için:")
    print(f"  python training/train_terrain.py")


if __name__ == "__main__":
    main()
