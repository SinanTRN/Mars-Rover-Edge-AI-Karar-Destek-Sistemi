"""
Sensör anomali tespiti — Isolation Forest + Mars-spesifik threshold.
Çıktı: normal / anormal / kritik durum.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


class SensorAnomalyDetector:
    def __init__(self, contamination=config.ANOMALY_CONTAMINATION):
        self.model = IsolationForest(
            contamination=contamination, random_state=42, n_estimators=100)
        self.is_fitted = False

    def fit(self, df):
        features = self._extract_features(df)
        if len(features) > 0:
            self.model.fit(features)
            self.is_fitted = True

    def detect(self, sensor_values: dict) -> dict:
        """
        sensor_values: {"air_temp": -60, "ground_temp": -45, "pressure": 636, ...}
        Döndürür: {status, anomaly_score, anomalous_sensors, details}
        """
        # 1. Threshold-based kontrol (Mars referanslarına göre)
        threshold_result = self._threshold_check(sensor_values)

        # 2. Isolation Forest (eğitilmişse)
        if_score = 0.0
        if self.is_fitted:
            row = self._values_to_array(sensor_values)
            if row is not None:
                if_score = self.model.decision_function(row.reshape(1, -1))[0]

        # Sonuç birleştirme
        critical_count = sum(1 for d in threshold_result["details"].values()
                             if d["status"] == "kritik")
        abnormal_count = sum(1 for d in threshold_result["details"].values()
                             if d["status"] == "anormal")

        if critical_count > 0 or if_score < -0.3:
            status = "kritik"
        elif abnormal_count > 0 or if_score < -0.1:
            status = "anormal"
        else:
            status = "normal"

        anomalous = [k for k, v in threshold_result["details"].items()
                     if v["status"] != "normal"]

        return {
            "status": status,
            "anomaly_score": round(float(if_score), 4),
            "anomalous_sensors": anomalous,
            "details": threshold_result["details"],
        }

    def _threshold_check(self, values: dict) -> dict:
        ref = config.MARS_REFERENCE
        details = {}

        checks = {
            "air_temp": ("air_temp_min", "air_temp_max"),
            "ground_temp": ("ground_temp_min", "ground_temp_max"),
            "pressure": ("pressure_min", "pressure_max"),
            "uv_index": (None, "uv_index_max"),
            "humidity": (None, "humidity_max"),
        }

        for sensor, (min_key, max_key) in checks.items():
            val = values.get(sensor)
            if val is None:
                continue

            lo = ref.get(min_key, -999) if min_key else -999
            hi = ref.get(max_key, 999)

            # Kritik: referans aralığının çok dışında (%30+ sapma)
            range_size = max(abs(hi - lo), 1)
            if val < lo - 0.3 * range_size or val > hi + 0.3 * range_size:
                details[sensor] = {"value": val, "status": "kritik",
                                   "note": f"Referans aralığı dışı ({lo}–{hi})"}
            elif val < lo or val > hi:
                details[sensor] = {"value": val, "status": "anormal",
                                   "note": f"Sınır değerde ({lo}–{hi})"}
            else:
                details[sensor] = {"value": val, "status": "normal",
                                   "note": "Normal aralıkta"}

        return {"details": details}

    def _extract_features(self, df):
        cols = ["air_temp", "ground_temp", "pressure", "uv_index", "humidity"]
        existing = [c for c in cols if c in df.columns]
        if not existing:
            return np.array([])
        return df[existing].dropna().values

    def _values_to_array(self, values: dict):
        cols = ["air_temp", "ground_temp", "pressure", "uv_index", "humidity"]
        arr = []
        for c in cols:
            if c in values:
                arr.append(float(values[c]))
        return np.array(arr) if arr else None
