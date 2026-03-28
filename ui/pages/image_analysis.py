"""
🛰️ Görüntü Analizi sayfası — Mars terrain sınıflandırma + Grad-CAM
"""
import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from utils.visualization import terrain_probability_chart
from utils.helpers import pil_to_numpy, get_sample_images
import config


def render(model, grad_cam_module):
    st.header("🛰️ Görüntü Analizi")
    st.caption("Mars arazi görüntüsünü analiz ederek terrain tipini sınıflandırır")

    # Görüntü kaynağı
    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded = st.file_uploader(
            "Mars görüntüsü yükle", type=["jpg", "jpeg", "png"],
            key="image_upload")

    with col2:
        sample_images = get_sample_images(config.SAMPLE_IMAGES_DIR)
        sample_names = ["Seçiniz..."] + [f.name for f in sample_images]
        selected_sample = st.selectbox("veya Demo görüntüsü seç",
                                       sample_names, key="sample_select")

    # Görüntü yükle
    image = None
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
    elif selected_sample != "Seçiniz...":
        img_path = config.SAMPLE_IMAGES_DIR / selected_sample
        if img_path.exists():
            image = Image.open(img_path).convert("RGB")

    if image is None:
        st.info("👆 Bir Mars görüntüsü yükleyin veya demo görüntüsü seçin")
        return None

    # Kullanıcıya gösterilecek orijinal (renkli) görüntü
    display_image = image.copy()

    # Analiz
    st.divider()
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.subheader("📷 Orijinal Görüntü")
        st.image(display_image, width='stretch')

    # Model prediction
    with st.spinner("Arazi analiz ediliyor..."):
        result = model.predict(image)
        st.session_state["terrain_result"] = result

    with col_result:
        st.subheader("📊 Analiz Sonucu")

        # Ana sonuç
        conf_pct = result["confidence"] * 100
        if conf_pct > 80:
            conf_color = "🟢"
        elif conf_pct > 60:
            conf_color = "🟡"
        else:
            conf_color = "🔴"

        st.metric("Arazi Tipi", result["class_tr"],
                  delta=f"{conf_color} %{conf_pct:.1f} güvenilirlik")

        # Olasılık grafiği
        fig = terrain_probability_chart(result["probabilities"])
        st.plotly_chart(fig, width='stretch')

    # Grad-CAM
    st.divider()
    st.subheader("🔍 Grad-CAM — Model Dikkat Haritası")
    st.caption("Modelin hangi bölgeye bakarak karar verdiğini gösterir")

    try:
        input_tensor = model.preprocess(image)
        cam_map = grad_cam_module.generate(input_tensor)
        original_np = pil_to_numpy(display_image.resize((224, 224)))
        overlay_img = grad_cam_module.overlay(original_np, cam_map)

        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.image(original_np, caption="Orijinal", width='stretch')
        with gcol2:
            st.image(overlay_img, caption="Grad-CAM Overlay",
                     width='stretch')
    except Exception as e:
        st.warning(f"Grad-CAM oluşturulamadı: {e}")

    return result
