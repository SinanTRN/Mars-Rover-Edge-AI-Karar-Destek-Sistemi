"""
🔴 Mars Rover Edge AI — Akıllı Veri Analizi ve Karar Destek Sistemi
Ana Streamlit uygulaması.
"""
import streamlit as st
import sys
from pathlib import Path

# Proje kök dizinini path'e ekle
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
from ui.styles import inject_css

# ── Sayfa Konfigürasyonu ──────────────────────────────
st.set_page_config(
    page_title="Mars Rover AI Karar Destek",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()


# ── Model Yükleme (cached) ───────────────────────────
@st.cache_resource
def load_terrain_model():
    from models.terrain_classifier import TerrainClassifier, load_model
    if config.MODEL_PATH.exists():
        return load_model(config.MODEL_PATH)
    # Pretrained model yoksa → ImageNet weight'li model (demo için)
    model = TerrainClassifier()
    model.eval()
    return model


@st.cache_resource
def load_grad_cam(_model):
    from models.grad_cam import GradCAM
    return GradCAM(_model)


# ── Tek Tıkla Demo Fonksiyonu ────────────────────────
def _run_quick_demo():
    """Demo sensör verileriyle tam pipeline çalıştır."""
    from sensors.anomaly_detector import SensorAnomalyDetector
    from fusion.decision_engine import FusionEngine

    # Demo sensör değerleri (anormal senaryo)
    demo_sensors = {
        "air_temp": -95.8,
        "ground_temp": -82.4,
        "pressure": 685.2,
        "uv_index": 8.9,
        "humidity": 0.1,
    }

    # Terrain — demo sonuç (eğitilmiş model yoksa)
    if "terrain_result" not in st.session_state:
        st.session_state["terrain_result"] = {
            "class": "sand",
            "class_tr": "Kum",
            "confidence": 0.73,
            "probabilities": {"soil": 0.12, "bedrock": 0.08,
                              "sand": 0.73, "big_rock": 0.07},
        }

    # Sensör analizi
    detector = SensorAnomalyDetector()
    sensor_result = detector.detect(demo_sensors)
    st.session_state["sensor_result"] = sensor_result
    st.session_state["sensor_values"] = demo_sensors

    # Fusion
    engine = FusionEngine()
    fusion_result = engine.fuse(
        st.session_state["terrain_result"], sensor_result)
    st.session_state["fusion_result"] = fusion_result

    st.success("✅ Demo verileri yüklendi! Sekmeler arasında gezinin.")


# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔴 Mars Rover AI")
    st.markdown("**Edge AI Karar Destek Sistemi**")
    st.divider()

    # Tek Tıkla Demo
    st.markdown("### 🚀 Hızlı Demo")
    if st.button("⚡ Tek Tıkla Demo", width='stretch'):
        _run_quick_demo()

    st.divider()

    # Bilgi
    st.markdown("### ℹ️ Bilgi")
    st.markdown("""
    **Veri Setleri:**
    - NASA AI4Mars (Terrain)
    - NASA REMS (Sensör)

    **Model:** MobileNetV2
    **LLM:** Mistral-7B (HF Free)

    **Modüller:**
    1. Görüntü Analizi
    2. Sensör Anomali Tespiti
    3. Multi-Modal Fusion
    4. LLM Astrobiyoloji Yorumu
    """)

    st.divider()
    st.caption("© 2026 Mars Rover AI — Yarışma Projesi")


# ── Ana İçerik — Sekmeler ─────────────────────────────
st.markdown("# 🔴 Mars Rover — Edge AI Karar Destek Sistemi")
st.markdown("*Mars rover verilerini analiz eden, yorumlayan ve bilimsel çıkarım yapan AI sistemi*")

tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ Görüntü Analizi",
    "📊 Sensör Analizi",
    "🔗 Fusion & Risk",
    "🤖 LLM Bilimsel Yorum",
])

# Tab 1: Görüntü Analizi
with tab1:
    from ui.pages import image_analysis
    model = load_terrain_model()
    gcam = load_grad_cam(model)
    image_analysis.render(model, gcam)

# Tab 2: Sensör Analizi
with tab2:
    from ui.pages import sensor_analysis
    sensor_analysis.render()

# Tab 3: Fusion & Risk
with tab3:
    from ui.pages import fusion_view
    fusion_view.render()

# Tab 4: LLM Bilimsel Yorum
with tab4:
    from ui.pages import llm_report
    llm_report.render()
