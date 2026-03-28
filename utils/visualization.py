"""
Plotly grafikleri ve görselleştirme yardımcıları.
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd


def terrain_probability_chart(probabilities: dict) -> go.Figure:
    classes = list(probabilities.keys())
    values = list(probabilities.values())
    colors = ["#c2825a", "#8B7355", "#DEB887", "#696969"]

    fig = go.Figure(go.Bar(
        x=values, y=classes, orientation="h",
        marker_color=colors[:len(classes)],
        text=[f"%{v*100:.1f}" for v in values],
        textposition="auto",
    ))
    fig.update_layout(
        title="Arazi Sınıfı Olasılıkları",
        xaxis_title="Olasılık",
        yaxis_title="",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
    )
    return fig


def sensor_timeseries_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    if "air_temp" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["sol"], y=df["air_temp"],
            name="Hava Sıcaklığı (°C)", mode="lines+markers",
            line=dict(color="#FF6B6B", width=2),
            marker=dict(size=4),
        ))
    if "ground_temp" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["sol"], y=df["ground_temp"],
            name="Zemin Sıcaklığı (°C)", mode="lines+markers",
            line=dict(color="#FFD93D", width=2),
            marker=dict(size=4),
        ))
    if "pressure" in df.columns:
        fig.add_trace(go.Scatter(
            x=df["sol"], y=df["pressure"],
            name="Basınç (Pa)", mode="lines+markers",
            line=dict(color="#6BCB77", width=2),
            marker=dict(size=4),
            yaxis="y2",
        ))

    fig.update_layout(
        title="Mars Sensör Verisi Zaman Serisi",
        xaxis_title="Sol (Mars Günü)",
        yaxis_title="Sıcaklık (°C)",
        yaxis2=dict(title="Basınç (Pa)", overlaying="y", side="right"),
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def risk_gauge_chart(risk_score: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score * 100,
        title={"text": "Risk Skoru", "font": {"color": "#E0E0E0"}},
        number={"suffix": "%", "font": {"color": "#E0E0E0"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#E0E0E0"},
            "bar": {"color": "#FF4444" if risk_score > 0.7 else
                    "#FFD93D" if risk_score > 0.3 else "#6BCB77"},
            "steps": [
                {"range": [0, 30], "color": "rgba(107, 203, 119, 0.2)"},
                {"range": [30, 70], "color": "rgba(255, 217, 61, 0.2)"},
                {"range": [70, 100], "color": "rgba(255, 68, 68, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": risk_score * 100,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
    )
    return fig


def anomaly_highlight_chart(df: pd.DataFrame, anomaly_indices: list) -> go.Figure:
    fig = go.Figure()

    normal_mask = ~df.index.isin(anomaly_indices)

    if "air_temp" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.loc[normal_mask, "sol"],
            y=df.loc[normal_mask, "air_temp"],
            name="Normal", mode="markers",
            marker=dict(color="#6BCB77", size=6),
        ))
        if any(~normal_mask):
            fig.add_trace(go.Scatter(
                x=df.loc[~normal_mask, "sol"],
                y=df.loc[~normal_mask, "air_temp"],
                name="Anomali", mode="markers",
                marker=dict(color="#FF4444", size=10, symbol="x"),
            ))

    fig.update_layout(
        title="Anomali Tespiti — Sıcaklık Değerleri",
        xaxis_title="Sol",
        yaxis_title="Sıcaklık (°C)",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
    )
    return fig
