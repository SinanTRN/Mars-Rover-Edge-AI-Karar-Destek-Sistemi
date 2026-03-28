"""
Mars temalı Streamlit custom CSS.
"""

MARS_CSS = """
<style>
/* Ana tema */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #1a0a0a 0%, #2d1810 50%, #1a0a0a 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1c1008 0%, #2a1a10 100%);
    border-right: 1px solid #c2825a33;
}

/* Başlıklar */
h1, h2, h3 {
    color: #E8C4A0 !important;
}

/* Metrik kartları */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #2a1a1055, #3a2a2055);
    border: 1px solid #c2825a44;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

div[data-testid="stMetric"] label {
    color: #c2825a !important;
    font-weight: 600;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #E8C4A0 !important;
}

/* Tab stili */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #1a0a0a88;
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #c2825a;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: #c2825a33 !important;
    color: #E8C4A0 !important;
}

/* Butonlar */
.stButton > button {
    background: linear-gradient(135deg, #c2825a, #8B5E3C);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #d4956b, #a0714d);
    box-shadow: 0 4px 15px rgba(194, 130, 90, 0.4);
}

/* Bilgi kartları */
.info-card {
    background: linear-gradient(135deg, #2a1a10, #3a2a20);
    border: 1px solid #c2825a44;
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.risk-badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 14px;
    margin: 4px;
}

.risk-safe { background: #1a4a2a; color: #6BCB77; border: 1px solid #6BCB7744; }
.risk-warning { background: #4a3a1a; color: #FFD93D; border: 1px solid #FFD93D44; }
.risk-critical { background: #4a1a1a; color: #FF4444; border: 1px solid #FF444444; }

/* LLM rapor alanı */
.llm-report {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #4a9eff44;
    border-radius: 12px;
    padding: 24px;
    margin: 10px 0;
    line-height: 1.7;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}

::-webkit-scrollbar-thumb {
    background: #c2825a55;
    border-radius: 3px;
}

/* Expander */
.streamlit-expanderHeader {
    color: #E8C4A0 !important;
    font-weight: 600;
}
</style>
"""


def inject_css():
    import streamlit as st
    st.markdown(MARS_CSS, unsafe_allow_html=True)
