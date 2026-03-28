"""
Streamlit yeniden kullanılabilir UI bileşenleri.
"""
import streamlit as st


def render_metric_card(title, value, icon="📊", delta=None, color="normal"):
    st.metric(label=f"{icon} {title}", value=value, delta=delta)


def render_risk_badge(zone, priority):
    zone_class = {
        "güvenli bölge": "risk-safe",
        "incelenmeli": "risk-warning",
        "kritik bölge": "risk-critical",
    }.get(zone, "risk-warning")

    priority_class = {
        "düşük": "risk-safe",
        "orta": "risk-warning",
        "yüksek": "risk-critical",
    }.get(priority, "risk-warning")

    html = f"""
    <div style="display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0;">
        <span class="risk-badge {zone_class}">🗺️ {zone.upper()}</span>
        <span class="risk-badge {priority_class}">📡 Öncelik: {priority.upper()}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_llm_report(report_text):
    st.markdown(f'<div class="llm-report">{_md_to_safe_html(report_text)}</div>',
                unsafe_allow_html=True)


def render_info_card(title, content, icon="ℹ️"):
    st.markdown(f"""
    <div class="info-card">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


def render_sensor_status_cards(details: dict):
    cols = st.columns(len(details))
    status_icons = {"normal": "🟢", "anormal": "🟡", "kritik": "🔴"}
    for col, (sensor, info) in zip(cols, details.items()):
        with col:
            icon = status_icons.get(info["status"], "⚪")
            st.metric(
                label=f"{icon} {sensor}",
                value=f"{info['value']}",
                delta=info["status"],
            )


def _md_to_safe_html(text):
    """Basit markdown → güvenli HTML dönüşümü."""
    import re
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Başlıklar
    text = re.sub(r"^### (.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Yeni satır
    text = text.replace("\n", "<br>")
    return text
