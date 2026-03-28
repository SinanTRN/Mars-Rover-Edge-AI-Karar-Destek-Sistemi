"""
Fusion karar motoru — Görüntü + sensör sonuçlarını birleştirir.
Çıktı: zone (güvenli/incelenmeli/kritik), priority (düşük/orta/yüksek), risk_score (0-1)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Risk matrisi: (terrain, sensor_status) → (zone, priority, base_risk)
RISK_MATRIX = {
    ("soil", "normal"):       ("güvenli bölge",  "düşük",  0.1),
    ("soil", "anormal"):      ("incelenmeli",    "orta",   0.4),
    ("soil", "kritik"):       ("kritik bölge",   "yüksek", 0.7),
    ("bedrock", "normal"):    ("güvenli bölge",  "düşük",  0.1),
    ("bedrock", "anormal"):   ("incelenmeli",    "orta",   0.4),
    ("bedrock", "kritik"):    ("kritik bölge",   "yüksek", 0.8),
    ("sand", "normal"):       ("incelenmeli",    "orta",   0.3),
    ("sand", "anormal"):      ("incelenmeli",    "yüksek", 0.6),
    ("sand", "kritik"):       ("kritik bölge",   "yüksek", 0.9),
    ("big_rock", "normal"):   ("incelenmeli",    "orta",   0.3),
    ("big_rock", "anormal"):  ("kritik bölge",   "yüksek", 0.7),
    ("big_rock", "kritik"):   ("kritik bölge",   "yüksek", 0.95),
}

PRIORITY_ORDER = {"düşük": 0, "orta": 1, "yüksek": 2}
PRIORITY_LIST = ["düşük", "orta", "yüksek"]


class FusionEngine:
    def fuse(self, terrain_result: dict, sensor_result: dict) -> dict:
        terrain_class = terrain_result.get("class", "soil")
        sensor_status = sensor_result.get("status", "normal")
        confidence = terrain_result.get("confidence", 1.0)

        key = (terrain_class, sensor_status)
        zone, priority, base_risk = RISK_MATRIX.get(key, ("incelenmeli", "orta", 0.5))

        # Düşük confidence → önceliği bir seviye yükselt
        if confidence < config.CONFIDENCE_THRESHOLD:
            idx = min(PRIORITY_ORDER.get(priority, 1) + 1, 2)
            priority = PRIORITY_LIST[idx]
            base_risk = min(base_risk + 0.15, 1.0)

        reasoning = self._build_reasoning(terrain_class, sensor_status,
                                          confidence, zone, priority)

        return {
            "zone": zone,
            "priority": priority,
            "risk_score": round(base_risk, 2),
            "reasoning": reasoning,
            "terrain_class": terrain_class,
            "sensor_status": sensor_status,
        }

    def _build_reasoning(self, terrain, sensor, confidence, zone, priority):
        parts = []
        parts.append(f"Arazi tipi '{terrain}' olarak sınıflandırıldı "
                     f"(%{confidence*100:.0f} güvenilirlik).")
        parts.append(f"Sensör durumu: {sensor}.")

        if confidence < config.CONFIDENCE_THRESHOLD:
            parts.append("⚠ Model güvenilirliği düşük — öncelik yükseltildi.")

        if zone == "kritik bölge":
            parts.append("🔴 Bu bölge kritik olarak işaretlendi — "
                         "acil inceleme gerekli.")
        elif zone == "incelenmeli":
            parts.append("🟡 Bu bölge incelenmeli — ileri analiz önerilir.")
        else:
            parts.append("🟢 Bu bölge güvenli görünüyor.")

        return " ".join(parts)
