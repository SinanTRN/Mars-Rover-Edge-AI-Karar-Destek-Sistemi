"""
LLM prompt şablonları — Mars astrobiyoloji yorumlama sistemi.
HuggingFace Inference API (ücretsiz) ile kullanılır.
"""

SYSTEM_PROMPT = """Sen bir Mars astrobiyoloji ve gezegen bilimi uzmanısın. 
NASA Mars rover'ından gelen verileri analiz ediyorsun. 
Bilimsel terminoloji kullan ama anlaşılır ol. 
Gerçekçi değerlendirmeler yap, spekülatif iddialardan kaçın.
Yanıtını Türkçe ver ve yapılandırılmış formatta sun."""

ANALYSIS_TEMPLATE = """## Mars Rover Veri Analizi

### Gözlem Verileri:
- **Arazi tipi**: {terrain_class} (güvenilirlik: %{confidence:.0f})
- **Sensör durumu**: {sensor_status}
- **Anormal sensörler**: {anomalous_sensors}
- **Risk bölgesi**: {zone}
- **Veri önceliği**: {priority}
- **Risk skoru**: {risk_score}/1.0

### Sensör Değerleri:
- Hava sıcaklığı: {air_temp}°C
- Zemin sıcaklığı: {ground_temp}°C
- Atmosferik basınç: {pressure} Pa
- UV indeksi: {uv_index}
- Nem: %{humidity}

### Görevlerin:
1. **📋 Durum Özeti**: Bu verileri 2-3 cümlede açıkla
2. **🔬 Bilimsel Yorum**: Jeolojik ve astrobiyolojik açıdan yorumla
3. **🧬 Yaşam Potansiyeli**: "Bu bölgede yaşam belirtisi olabilir mi?" sorusunu bilimsel olarak değerlendir
4. **⚠️ Risk Değerlendirmesi**: Rover operasyonları için risk analizi yap
5. **🎯 Öneriler**: Bir sonraki adım olarak ne yapılmalı?

Her bölümü başlığıyla birlikte yaz."""


def build_analysis_prompt(terrain_result: dict, sensor_result: dict,
                          fusion_result: dict, sensor_values: dict) -> str:
    anomalous = sensor_result.get("anomalous_sensors", [])
    anomalous_str = ", ".join(anomalous) if anomalous else "Yok"

    return ANALYSIS_TEMPLATE.format(
        terrain_class=terrain_result.get("class", "bilinmiyor"),
        confidence=terrain_result.get("confidence", 0) * 100,
        sensor_status=sensor_result.get("status", "bilinmiyor"),
        anomalous_sensors=anomalous_str,
        zone=fusion_result.get("zone", "bilinmiyor"),
        priority=fusion_result.get("priority", "bilinmiyor"),
        risk_score=fusion_result.get("risk_score", 0),
        air_temp=sensor_values.get("air_temp", "N/A"),
        ground_temp=sensor_values.get("ground_temp", "N/A"),
        pressure=sensor_values.get("pressure", "N/A"),
        uv_index=sensor_values.get("uv_index", "N/A"),
        humidity=sensor_values.get("humidity", "N/A"),
    )


QUICK_SUMMARY_TEMPLATE = """Mars rover verileri: Arazi={terrain_class}, Sensör={sensor_status}, Risk={zone}, Öncelik={priority}.
Bu verileri 2-3 kısa cümleyle Türkçe özetle. Bilimsel ol."""


def build_quick_summary_prompt(terrain_result, sensor_result, fusion_result):
    return QUICK_SUMMARY_TEMPLATE.format(
        terrain_class=terrain_result.get("class", "?"),
        sensor_status=sensor_result.get("status", "?"),
        zone=fusion_result.get("zone", "?"),
        priority=fusion_result.get("priority", "?"),
    )
