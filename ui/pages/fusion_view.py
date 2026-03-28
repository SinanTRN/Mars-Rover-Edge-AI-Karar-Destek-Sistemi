"""
🔗 Fusion & Risk sayfası — Görüntü + sensör sonuçlarını birleştirir
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from fusion.decision_engine import FusionEngine
from ui.components import render_risk_badge, render_info_card
from utils.visualization import risk_gauge_chart
from utils.helpers import format_risk_emoji, format_priority_emoji


def render():
    st.header("🔗 Fusion — Veri Birleştirme & Risk Analizi")
    st.caption("Görüntü ve sensör sonuçlarını birleştirerek risk değerlendirmesi yapar")

    terrain_result = st.session_state.get("terrain_result")
    sensor_result = st.session_state.get("sensor_result")

    if terrain_result is None:
        st.warning("⚠️ Önce **Görüntü Analizi** sekmesinde bir görüntü analiz edin.")
        return None
    if sensor_result is None:
        st.warning("⚠️ Önce **Sensör Analizi** sekmesinde sensör verilerini değerlendirin.")
        return None

    # Fusion
    engine = FusionEngine()
    fusion_result = engine.fuse(terrain_result, sensor_result)
    st.session_state["fusion_result"] = fusion_result

    # Sonuçlar
    st.divider()

    # Üst satır: Risk badge'leri
    render_risk_badge(fusion_result["zone"], fusion_result["priority"])

    # Metrikler
    col1, col2, col3 = st.columns(3)
    with col1:
        zone_emoji = format_risk_emoji(fusion_result["zone"])
        st.metric("🗺️ Bölge Durumu", f"{zone_emoji} {fusion_result['zone']}")
    with col2:
        pri_emoji = format_priority_emoji(fusion_result["priority"])
        st.metric("📡 Veri Önceliği", f"{pri_emoji} {fusion_result['priority']}")
    with col3:
        st.metric("⚡ Risk Skoru", f"{fusion_result['risk_score']:.2f} / 1.0")

    # Risk gauge
    fig = risk_gauge_chart(fusion_result["risk_score"])
    st.plotly_chart(fig, width='stretch')

    # Giriş verileri özeti
    st.divider()
    st.subheader("📋 Giriş Verileri Özeti")

    col_a, col_b = st.columns(2)
    with col_a:
        render_info_card(
            "Görüntü Analizi",
            f"Arazi: <strong>{terrain_result['class_tr']}</strong> | "
            f"Güvenilirlik: %{terrain_result['confidence']*100:.1f}",
            "🛰️"
        )
    with col_b:
        anomalous = sensor_result.get("anomalous_sensors", [])
        anomalous_str = ", ".join(anomalous) if anomalous else "Yok"
        render_info_card(
            "Sensör Analizi",
            f"Durum: <strong>{sensor_result['status']}</strong> | "
            f"Anormal: {anomalous_str}",
            "📊"
        )

    # Reasoning
    st.divider()
    st.subheader("💡 Karar Gerekçesi")
    st.info(fusion_result["reasoning"])

    return fusion_result
