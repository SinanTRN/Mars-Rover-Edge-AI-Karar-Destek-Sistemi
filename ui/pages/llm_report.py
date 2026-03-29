"""
🤖 LLM Bilimsel Yorum sayfası — HuggingFace Mistral ile Mars yorumlama
"""
import streamlit as st
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm.interpreter import MarsInterpreter
from ui.components import render_llm_report
import config


def render():
    st.header("🤖 LLM Bilimsel Yorumlama")
    st.caption("HuggingFace Mistral-7B ile rover verilerini bilimsel olarak yorumlar")

    terrain_result = st.session_state.get("terrain_result")
    sensor_result = st.session_state.get("sensor_result")
    fusion_result = st.session_state.get("fusion_result")
    sensor_values = st.session_state.get("sensor_values", {})

    # Ön koşul kontrolleri
    missing = []
    if terrain_result is None:
        missing.append("Görüntü Analizi")
    if sensor_result is None:
        missing.append("Sensör Analizi")
    if fusion_result is None:
        missing.append("Fusion & Risk")

    if missing:
        st.warning(f"⚠️ Önce şu sekmeleri tamamlayın: **{', '.join(missing)}**")
        st.info("💡 Veya sidebar'dan **Tek Tıkla Demo** butonunu kullanın.")
        return

    # API token (config'den)
    api_token = config.HF_API_TOKEN

    # Mevcut verilerin özeti
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Arazi", terrain_result.get("class_tr", "?"))
    with col2:
        st.metric("Sensör", sensor_result.get("status", "?"))
    with col3:
        st.metric("Risk", fusion_result.get("zone", "?"))

    # LLM çağrısı
    st.divider()

    if st.button("🧬 Bilimsel Yorum Oluştur", type="primary", width='stretch'):
        with st.spinner("🤖 Mistral-7B Mars verilerini yorumluyor..."):
            interpreter = MarsInterpreter(api_token=api_token)

            if not interpreter.is_available():
                st.error("LLM bağlantısı kurulamadı. Token'ı kontrol edin.\n Kullanılan yapay zeka modeli lokalde çalışanbir model olduğundan github üzerinden canlıya aldığımız bu demoda çalıştırılamıyor. ")
                return

            response = interpreter.interpret(
                terrain_result, sensor_result, fusion_result, sensor_values)

        if response["success"]:
            st.success(f"✅ Yorum oluşturuldu — Model: {response['model_used']}")
            st.divider()

            # Rapor göster
            render_llm_report(response["report"])

            # Rapor indirme
            st.divider()
            st.download_button(
                "📥 Raporu İndir (Markdown)",
                data=response["report"],
                file_name="mars_bilimsel_rapor.md",
                mime="text/markdown",
            )

            st.session_state["llm_report"] = response["report"]
        else:
            st.warning("⚠️ LLM yanıt veremedi. Fallback rapor gösteriliyor:")
            render_llm_report(response["report"])

    # Önceki rapor varsa göster
    elif "llm_report" in st.session_state:
        st.info("📄 Önceki rapor:")
        render_llm_report(st.session_state["llm_report"])
