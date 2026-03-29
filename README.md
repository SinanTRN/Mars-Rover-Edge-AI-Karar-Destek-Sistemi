# 🔴 Edge AI ve LLM Tabanlı Mars Rover Akıllı Veri Analizi ve Karar Destek Sistemi

Mars rover tarafından toplanan görüntü ve sensör verilerini rover üzerinde (edge) analiz ederek kritik verileri seçen ve HuggingFace LLM ile yorumlayarak bilimsel anlam çıkaran multi-modal AI sistemi.

## 🏗️ Mimari

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────┐
│  NASA AI4Mars   │────▶│  Terrain CNN     │────▶│               │
│  Görüntü Verisi │     │  (MobileNetV2)   │     │   FUSION      │     ┌──────────────┐
└─────────────────┘     │  + Grad-CAM      │     │   ENGINE      │────▶│  HuggingFace │
                        └──────────────────┘     │               │     │  Llama-3.1   │
┌─────────────────┐     ┌──────────────────┐     │  Risk Matrisi │     │              │
│  NASA REMS      │────▶│  Anomaly Det.    │────▶│  Zone/Öncelik │    │              │
│  Sensör Verisi  │     │  (IsolationForest│     │               │     │  Bilimsel    │
└─────────────────┘     │   + Threshold)   │     └───────────────┘     │  Yorumlama   │
                        └──────────────────┘                           └──────────────┘
                                                                              │
                                                                              ▼
                                                                     ┌──────────────┐
                                                                     │  Streamlit   │
                                                                     │  Dashboard   │
                                                                     └──────────────┘
```

## 📊 Veri Setleri

| Dataset | Kaynak | Kullanım |
|---------|--------|----------|
| **AI4Mars** | [NASA](https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars) | Terrain sınıflandırma (soil/bedrock/sand/big_rock) — ~32K görüntü |
| **REMS** | [NASA Mars Weather](https://mars.nasa.gov/) | Sıcaklık, basınç, UV anomali tespiti |

## 🧩 Sistem Bileşenleri

1. **Görüntü Analizi** — MobileNetV2 (PyTorch) ile 4 sınıflı terrain sınıflandırma + Grad-CAM
2. **Sensör Analizi** — Isolation Forest + Mars-spesifik threshold ile anomali tespiti
3. **Fusion Engine** — Rule-based risk matrisi (zone + priority + score)
4. **LLM Yorumlama** — HuggingFace Llama-3.1-8B-Instruct (ücretsiz API) ile astrobiyolojik yorum

## 🚀 Kurulum

### 1. Bağımlılıkları yükle
```bash
pip install -r requirements.txt
```

### 2. HuggingFace Token (ücretsiz)
```bash
# .env dosyası oluştur
cp .env.example .env
# .env içine token'ını yaz (https://huggingface.co/settings/tokens)
```

### 3. Veri indirme (opsiyonel — model eğitimi için)
```bash
# Yöntem A: Kaggle'dan (önerilen — kaggle.json gerekli)
kaggle datasets download -d yash92328/ai4mars-terrainaware-autonomous-driving-on-mars
# İndirilen ZIP'i data/ai4mars-dataset-merged-0.1/ altına çıkarın, ardından:
python _prepare_kaggle.py

# Yöntem B: Zenodo'dan (HTTP range request ile)
python data/download_msl.py
```

### 4. Model eğitimi (opsiyonel)
```bash
# Tüm veri ile (CPU'da uzun sürer)
python training/train_terrain.py --epochs 10 --batch-size 32

# Verinin %30'u ile (hızlı — ~7dk CPU'da)
python training/train_terrain.py --epochs 3 --batch-size 32 --fraction 0.3
```

### 5. Demo çalıştırma
```bash
streamlit run streamlit_app.py
```

## 🖥️ Arayüz

Uygulama 4 ana sekmeden oluşur:

| Sekme | İçerik |
|-------|--------|
| 🛰️ Görüntü Analizi | Mars görüntüsü yükleme, terrain sınıflandırma, Grad-CAM |
| 📊 Sensör Analizi | REMS veri görselleştirme, anomali tespiti |
| 🔗 Fusion & Risk | Multi-modal birleştirme, risk skoru, öncelik |
| 🤖 LLM Yorum | Llama-3.1-8B ile bilimsel yorumlama ve rapor |

**Tek Tıkla Demo**: Sidebar'dan demo butonu ile tüm pipeline'ı anında çalıştırabilirsiniz.

## 📁 Proje Yapısı

```
├── config.py                  # Merkezi konfigürasyon
├── streamlit_app.py           # Ana uygulama
├── models/
│   ├── terrain_classifier.py  # MobileNetV2 CNN
│   └── grad_cam.py            # Explainability
├── sensors/
│   ├── rems_parser.py         # REMS veri işleme
│   └── anomaly_detector.py    # Anomali tespiti
├── fusion/
│   └── decision_engine.py     # Risk matrisi
├── llm/
│   ├── interpreter.py         # HuggingFace LLM wrapper
│   └── prompts.py             # Prompt şablonları
├── training/
│   ├── dataset.py             # AI4Mars Dataset
│   ├── train_terrain.py       # Eğitim scripti
│   └── evaluate.py            # Değerlendirme
├── ui/
│   ├── components.py          # UI bileşenleri
│   ├── styles.py              # Mars temalı CSS
│   └── pages/                 # Sayfa modülleri
└── utils/
    ├── visualization.py       # Plotly grafikleri
    └── helpers.py             # Yardımcı fonksiyonlar
```

## 🔬 Teknik Detaylar

- **CNN**: MobileNetV2 (~3.4M parametre) — edge AI uyumlu
- **LLM**: Llama-3.1-8B-Instruct (primary) + Qwen-2.5-7B (fallback) via HuggingFace Inference API (ücretsiz)
- **Anomali**: Isolation Forest + Mars referans değerlerine göre threshold
- **Fusion**: Açıklanabilir rule-based risk matrisi
- **Explainability**: Grad-CAM dikkat haritaları

## 📜 Lisans

Bu proje eğitim ve yarışma amaçlıdır. AI4Mars veri seti CC-BY-4.0 lisansı ile sunulmaktadır.
