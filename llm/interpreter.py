"""
LLM Yorumlama Modülü — HuggingFace Inference API (Ücretsiz).
Mistral-7B-Instruct ile Mars rover verilerini bilimsel olarak yorumlar.
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from llm.prompts import SYSTEM_PROMPT, build_analysis_prompt, build_quick_summary_prompt

try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


FALLBACK_REPORT = """## 📋 Durum Özeti
LLM bağlantısı kurulamadı. Aşağıdaki sonuçlar rule-based sistemden üretilmiştir.

Arazi tipi: {terrain_class} | Sensör durumu: {sensor_status} | Risk: {zone}

## 🔬 Bilimsel Yorum
Otomatik yorumlama şu an kullanılamıyor. Lütfen HuggingFace API token'ınızı kontrol edin.

## 🧬 Yaşam Potansiyeli
Değerlendirme için LLM bağlantısı gereklidir.

## ⚠️ Risk Değerlendirmesi
Risk seviyesi: {zone} — Öncelik: {priority}

## 🎯 Öneriler
1. HuggingFace API token'ını .env dosyasına ekleyin
2. https://huggingface.co/settings/tokens adresinden ücretsiz token alın
"""


class MarsInterpreter:
    def __init__(self, api_token=None, model=None):
        self.api_token = api_token or config.HF_API_TOKEN
        self.model = model or config.HF_MODEL_PRIMARY
        self.fallback_model = config.HF_MODEL_FALLBACK
        self.client = None

        if HF_AVAILABLE and self.api_token:
            self.client = InferenceClient(token=self.api_token)

    def is_available(self) -> bool:
        return self.client is not None

    def interpret(self, terrain_result: dict, sensor_result: dict,
                  fusion_result: dict, sensor_values: dict) -> dict:
        """Ana yorumlama — 5 bölümlü bilimsel rapor üretir."""
        user_prompt = build_analysis_prompt(
            terrain_result, sensor_result, fusion_result, sensor_values)

        response_text = self._call_llm(user_prompt)

        if response_text is None:
            return {
                "success": False,
                "report": FALLBACK_REPORT.format(
                    terrain_class=terrain_result.get("class", "?"),
                    sensor_status=sensor_result.get("status", "?"),
                    zone=fusion_result.get("zone", "?"),
                    priority=fusion_result.get("priority", "?"),
                ),
                "model_used": "fallback",
            }

        return {
            "success": True,
            "report": response_text,
            "model_used": self.model,
        }

    def quick_summary(self, terrain_result, sensor_result, fusion_result) -> str:
        """Kısa 2-3 cümlelik özet."""
        prompt = build_quick_summary_prompt(
            terrain_result, sensor_result, fusion_result)
        result = self._call_llm(prompt)
        if result is None:
            return (f"Arazi: {terrain_result.get('class', '?')} | "
                    f"Sensör: {sensor_result.get('status', '?')} | "
                    f"Risk: {fusion_result.get('zone', '?')}")
        return result

    def _call_llm(self, user_prompt: str, retries=2) -> str | None:
        if not self.is_available():
            return None

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        models_to_try = [self.model, self.fallback_model]

        for model_id in models_to_try:
            for attempt in range(retries):
                try:
                    response = self.client.chat_completion(
                        model=model_id,
                        messages=messages,
                        max_tokens=config.LLM_MAX_TOKENS,
                        temperature=config.LLM_TEMPERATURE,
                    )
                    text = response.choices[0].message.content
                    if text and len(text.strip()) > 20:
                        self.model = model_id  # başarılı modeli hatırla
                        return text.strip()
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate" in error_str or "limit" in error_str:
                        wait = 2 ** (attempt + 1)
                        time.sleep(wait)
                        continue
                    if "model" in error_str and "not" in error_str:
                        break  # bu modelle olmayacak, sonrakine geç
                    if attempt < retries - 1:
                        time.sleep(1)
                        continue
                    break  # son deneme de başarısız
        return None
