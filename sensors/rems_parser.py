"""
NASA REMS (Rover Environmental Monitoring Station) veri işleme.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def load_rems_data(filepath=None):
    path = Path(filepath) if filepath else config.DATA_DIR / "sample_rems.csv"
    df = pd.read_csv(path)
    return df


def preprocess(df):
    numeric_cols = ["air_temp", "ground_temp", "pressure", "uv_index", "humidity"]
    existing = [c for c in numeric_cols if c in df.columns]
    df[existing] = df[existing].apply(pd.to_numeric, errors="coerce")
    df[existing] = df[existing].fillna(df[existing].median())
    return df


def get_sensor_summary(df):
    numeric_cols = ["air_temp", "ground_temp", "pressure", "uv_index", "humidity"]
    existing = [c for c in numeric_cols if c in df.columns]
    summary = {}
    for col in existing:
        summary[col] = {
            "min": round(df[col].min(), 2),
            "max": round(df[col].max(), 2),
            "mean": round(df[col].mean(), 2),
            "std": round(df[col].std(), 2),
        }
    return summary
