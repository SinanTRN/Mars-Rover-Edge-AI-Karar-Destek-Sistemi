"""
📊 Sensör Analizi sayfası — REMS anomali tespiti
"""
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from sensors.rems_parser import load_rems_data, preprocess
from sensors.anomaly_detector import SensorAnomalyDetector
from utils.visualization import sensor_timeseries_chart, anomaly_highlight_chart
from ui.components import render_sensor_status_cards
import config


def render():
    st.header("📊 Sensör Analizi")
    st.caption("Mars rover REMS sensör verilerini analiz ederek anomali tespit eder")

    # Veri kaynağı seç
    mode = st.radio("Veri Kaynağı", ["📂 Demo REMS Verisi", "✏️ Manuel Giriş"],
                    horizontal=True, key="sensor_mode")

    if mode == "📂 Demo REMS Verisi":
        result = _demo_mode()
    else:
        result = _manual_mode()

    return result


def _demo_mode():
    # Demo verisini yükle
    sample_path = config.DATA_DIR / "sample_rems.csv"
    if not sample_path.exists():
        st.error("Demo verisi bulunamadı: data/sample_rems.csv")
        return None

    df = load_rems_data(sample_path)
    df = preprocess(df)

    # Zaman serisi grafik
    fig = sensor_timeseries_chart(df)
    st.plotly_chart(fig, width='stretch')

    # Satır seçimi
    sol_values = df["sol"].tolist()
    selected_sol = st.select_slider("Mars Günü (Sol) Seç",
                                     options=sol_values,
                                     value=sol_values[len(sol_values) // 2],
                                     key="sol_slider")

    row = df[df["sol"] == selected_sol].iloc[0]
    sensor_values = {
        "air_temp": float(row.get("air_temp", -60)),
        "ground_temp": float(row.get("ground_temp", -50)),
        "pressure": float(row.get("pressure", 636)),
        "uv_index": float(row.get("uv_index", 5)),
        "humidity": float(row.get("humidity", 0.8)),
    }

    return _analyze_and_display(sensor_values, df)


def _manual_mode():
    st.markdown("**Sensör değerlerini ayarlayın:**")

    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.slider("🌡️ Hava Sıcaklığı (°C)", -150.0, 30.0, -60.0,
                             step=1.0, key="air_temp_slider")
        ground_temp = st.slider("🌍 Zemin Sıcaklığı (°C)", -150.0, 40.0, -50.0,
                                step=1.0, key="ground_temp_slider")
        pressure = st.slider("⏲️ Basınç (Pa)", 500.0, 800.0, 636.0,
                             step=1.0, key="pressure_slider")
    with col2:
        uv_index = st.slider("☀️ UV İndeksi", 0.0, 20.0, 5.0,
                             step=0.1, key="uv_slider")
        humidity = st.slider("💧 Nem (%)", 0.0, 10.0, 0.8,
                             step=0.1, key="humidity_slider")

    sensor_values = {
        "air_temp": air_temp,
        "ground_temp": ground_temp,
        "pressure": pressure,
        "uv_index": uv_index,
        "humidity": humidity,
    }

    return _analyze_and_display(sensor_values)


def _analyze_and_display(sensor_values, df=None):
    st.divider()
    st.subheader("🔎 Anomali Tespiti Sonucu")

    # Anomali tespiti
    detector = SensorAnomalyDetector()

    # Demo verisiyle fit et
    if df is not None:
        detector.fit(df)

    result = detector.detect(sensor_values)
    st.session_state["sensor_result"] = result
    st.session_state["sensor_values"] = sensor_values

    # Durum göstergeleri
    status_icon = {"normal": "🟢", "anormal": "🟡", "kritik": "🔴"}
    icon = status_icon.get(result["status"], "⚪")

    st.markdown(f"### {icon} Genel Durum: **{result['status'].upper()}**")

    if result.get("anomalous_sensors"):
        st.warning(f"Anormal sensörler: {', '.join(result['anomalous_sensors'])}")

    # Her sensörün detayı
    if result.get("details"):
        render_sensor_status_cards(result["details"])

    # Anomali skoru
    score = result.get("anomaly_score", 0)
    st.metric("Anomali Skoru", f"{score:.4f}",
              delta="Normal" if score > -0.1 else "Dikkat!")

    # Anomali grafik (df varsa)
    if df is not None:
        anomaly_indices = []
        for i, row in df.iterrows():
            vals = {
                "air_temp": row.get("air_temp", -60),
                "ground_temp": row.get("ground_temp", -50),
                "pressure": row.get("pressure", 636),
                "uv_index": row.get("uv_index", 5),
                "humidity": row.get("humidity", 0.8),
            }
            r = detector.detect(vals)
            if r["status"] != "normal":
                anomaly_indices.append(i)

        fig = anomaly_highlight_chart(df, anomaly_indices)
        st.plotly_chart(fig, width='stretch')

    return result
