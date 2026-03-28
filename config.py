"""
Merkezi konfigürasyon — Mars Rover Edge AI Karar Destek Sistemi
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PRETRAINED_DIR = MODEL_DIR / "pretrained"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"

# ── Terrain model ──────────────────────────────────────
TERRAIN_CLASSES = ["soil", "bedrock", "sand", "big_rock"]
TERRAIN_CLASSES_TR = ["Toprak", "Ana Kaya", "Kum", "Büyük Kaya"]
NUM_CLASSES = len(TERRAIN_CLASSES)
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
MODEL_PATH = PRETRAINED_DIR / "terrain_model.pth"

# ── Sensor analysis ───────────────────────────────────
ANOMALY_CONTAMINATION = 0.1
MARS_REFERENCE = {
    "air_temp_mean": -60.0,       # °C
    "air_temp_min": -120.0,
    "air_temp_max": 20.0,
    "ground_temp_mean": -50.0,
    "ground_temp_min": -120.0,
    "ground_temp_max": 30.0,
    "pressure_mean": 636.0,       # Pa  (Gale Crater)
    "pressure_min": 600.0,
    "pressure_max": 780.0,
    "uv_index_max": 16.0,
    "humidity_max": 5.0,          # % (Mars'ta çok düşük)
}

# ── HuggingFace LLM (ücretsiz) ────────────────────────
HF_API_TOKEN = "" # BURAYA HUGGINGFACE TOKEN'INIZI YAZIN VEYA .env DOSYASINI KULLANIN
HF_MODEL_PRIMARY = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL_FALLBACK = "Qwen/Qwen2.5-7B-Instruct"
LLM_MAX_TOKENS = 1500
LLM_TEMPERATURE = 0.7

# ── Fusion thresholds ─────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6  # altında → öncelik yükselt
