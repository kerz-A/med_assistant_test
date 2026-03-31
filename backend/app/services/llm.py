"""LLM service v3: medical correction, calibration extraction, protocol extraction, finalization."""

import json
import logging
import time

import httpx

from ..config import settings
from ..models.messages import Utterance
from ..models.protocol import MedicalProtocol

logger = logging.getLogger(__name__)

# ============================================================
# PROMPTS
# ============================================================

MEDICAL_CORRECTION_PROMPT = """\
Ты — медицинский корректор. Исправь ошибки распознавания речи в тексте.
Исправляй ТОЛЬКО ошибки в мед. терминах и лекарствах: нурофон→нурофен, парацетомол→парацетамол, \
гастрид→гастрит, гипертания→гипертония, диобед→диабет, тохикардия→тахикардия и т.д.
НЕ меняй смысл, НЕ перефразируй. Числа оставь как есть.
Ответь ТОЛЬКО исправленным текстом."""

CALIBRATION_PROMPT = """\
Из текста пациента извлеки ФИО, возраст и пол.
Ответь СТРОГО в JSON:
{"full_name": "Фамилия Имя Отчество" или null, "age": число или null, "gender": "М" или "Ж" или null}
Если данных нет — null. Без пояснений."""

EXTRACTION_SYSTEM_PROMPT = """\
Ты — квалифицированный врач с широкой клинической практикой, заполняющий медицинскую карту \
амбулаторного пациента на основе разговора врача и пациента.

═══ МЕДИЦИНСКИЙ СТИЛЬ ═══
Заполняй в стиле медицинской документации:
- Мед. термин + расшифровка: "Цефалгия (головная боль) в височно-затылочной области"
- Лекарства: МНН (торговое), дозировка. Пример: "Ибупрофен (Нурофен) 400 мг по потребности"
- Числовые данные точно: "интенсивность 6-7/10 по ВАШ"

═══ СТРОГОЕ РАЗДЕЛЕНИЕ ═══
- complaints / complaints_details — ТОЛЬКО жалобы САМОГО пациента
- anamnesis — история ТЕКУЩЕГО заболевания самого пациента
- life_anamnesis — хронические болезни пациента + операции + привычки \
  + "Наследственный анамнез: мать — ..., отец — ..."
- allergies — аллергии САМОГО пациента
- medications — лекарства САМОГО пациента
НИКОГДА не путай данные пациента и его родственников!

═══ ПРАВИЛА ═══
1. Извлекай данные ТОЛЬКО из реплик ПАЦИЕНТА. Реплики врача — контекст для понимания.
2. НЕ выдумывай. Заполняй только то, что пациент явно сказал.
3. Сохраняй ранее заполненные поля. НЕ удаляй и НЕ обнуляй.
4. Все текстовые поля — строки. Все vitals кроме systolic_bp — числа.

Ответь СТРОГО в JSON:
{{
  "exam_data": {{
    "complaints": "строка или null",
    "complaints_details": "строка или null",
    "anamnesis": "строка или null",
    "life_anamnesis": "строка или null",
    "allergies": "строка или null",
    "medications": "строка или null"
  }},
  "vitals": {{
    "height_cm": null, "weight_kg": null, "pulse": null, "spo2": null, "systolic_bp": null
  }}
}}\
"""

FINALIZATION_PROMPT = """\
Ты — квалифицированный врач с широкой клинической практикой. \
На основе собранного анамнеза сформируй заключение.

═══ ДАННЫЕ ПАЦИЕНТА ═══
{protocol_json}

═══ ПОЛНАЯ ТРАНСКРИПЦИЯ ═══
{transcript}

═══ ТРЕБОВАНИЯ ═══
1. ДИАГНОЗ: по МКБ-10, пометка "Предварительный", дифф. ряд если неоднозначно
2. ПЛАН ЛЕЧЕНИЯ: обследования (ОАК, БАК, ЭКГ...), препараты с дозировками, консультации
3. РЕКОМЕНДАЦИИ: простым языком для пациента, режим, тревожные симптомы

Ранее заполненные поля — перепиши в корректном медицинском стиле.

Ответь СТРОГО в JSON:
{{
  "exam_data": {{
    "complaints": "мед. стиль",
    "complaints_details": "мед. стиль",
    "anamnesis": "мед. стиль",
    "life_anamnesis": "мед. стиль",
    "allergies": "мед. стиль",
    "medications": "МНН (торг.), дозировка",
    "diagnosis": "Предварительный: код МКБ-10 Название",
    "treatment_plan": "конкретный план",
    "patient_recommendations": "простым языком"
  }},
  "vitals": {{ ... }}
}}\
"""


class LLMService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    def initialize(self) -> None:
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url, headers=headers, timeout=60.0,
        )
        logger.info("[LLM] Initialized: provider=%s model=%s", settings.llm_provider, settings.llm_model)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def _chat_json(self, system: str, user: str, temperature: float = 0.2) -> str:
        if not self._client:
            raise RuntimeError("LLM not initialized")
        payload = {
            "model": settings.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        }
        t0 = time.monotonic()
        resp = await self._client.post("/chat/completions", json=payload)
        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[LLM] Response: %d in %dms", resp.status_code, ms)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        logger.info("[LLM] Tokens: %s/%s/%s", usage.get("prompt_tokens","?"), usage.get("completion_tokens","?"), usage.get("total_tokens","?"))
        return data["choices"][0]["message"]["content"]

    async def _chat_plain(self, system: str, user: str) -> str:
        if not self._client:
            raise RuntimeError("LLM not initialized")
        payload = {
            "model": settings.llm_model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        t0 = time.monotonic()
        resp = await self._client.post("/chat/completions", json=payload)
        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[LLM] Correction: %d in %dms", resp.status_code, ms)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # ---- Medical correction ----
    async def correct_medical_terms(self, text: str) -> str:
        try:
            corrected = await self._chat_plain(MEDICAL_CORRECTION_PROMPT, text)
            if corrected and len(corrected) > 3:
                if corrected != text:
                    logger.info("[LLM] Corrected: '%s' → '%s'", text[:50], corrected[:50])
                return corrected
            return text
        except Exception as e:
            logger.warning("[LLM] Correction failed: %s", e)
            return text

    # ---- Calibration: extract patient info ----
    async def extract_patient_info(self, text: str) -> dict:
        try:
            raw = await self._chat_json(CALIBRATION_PROMPT, text)
            parsed = json.loads(raw)
            if parsed.get("age") and isinstance(parsed["age"], str):
                parsed["age"] = int("".join(c for c in parsed["age"] if c.isdigit()) or "0") or None
            logger.info("[LLM] Calibration: name=%s age=%s gender=%s",
                        parsed.get("full_name"), parsed.get("age"), parsed.get("gender"))
            return parsed
        except Exception as e:
            logger.error("[LLM] Calibration extract error: %s", e)
            return {}

    # ---- Protocol extraction (every cycle) ----
    async def extract_protocol_data(
        self, current: MedicalProtocol, utterances: list[Utterance], full_transcript: str,
    ) -> MedicalProtocol:
        new_text = "\n".join(
            f"[{'Врач' if u.speaker == 'doctor' else 'Пациент'}]: {u.text}"
            for u in utterances
        )
        user_prompt = (
            f"ТЕКУЩИЙ ПРОТОКОЛ:\n{current.model_dump_json(indent=2)}\n\n"
            f"НОВЫЙ ФРАГМЕНТ РАЗГОВОРА:\n{new_text}\n\n"
            "Извлеки данные из реплик ПАЦИЕНТА в медицинском стиле."
        )
        try:
            raw = await self._chat_json(EXTRACTION_SYSTEM_PROMPT, user_prompt)
            return self._merge_protocol(raw, current)
        except Exception as e:
            logger.error("[LLM] Extraction error: %s", e)
            return current

    # ---- Finalization ----
    async def finalize_protocol(self, current: MedicalProtocol, full_transcript: str) -> MedicalProtocol:
        system = FINALIZATION_PROMPT.format(
            protocol_json=current.model_dump_json(indent=2),
            transcript=full_transcript,
        )
        try:
            raw = await self._chat_json(system, "Сформируй заключение.", temperature=0.3)
            return self._merge_protocol(raw, current)
        except Exception as e:
            logger.error("[LLM] Finalization error: %s", e)
            return current

    # ---- Helpers ----
    def _merge_protocol(self, raw_json: str, fallback: MedicalProtocol) -> MedicalProtocol:
        try:
            parsed = json.loads(raw_json)
            updated = fallback.model_copy(deep=True)

            if "exam_data" in parsed:
                for field, value in parsed["exam_data"].items():
                    if value and hasattr(updated.exam_data, field):
                        if isinstance(value, (dict, list)):
                            value = str(value) if isinstance(value, list) else "; ".join(f"{k}: {v}" for k,v in value.items() if v)
                        setattr(updated.exam_data, field, str(value))

            if "vitals" in parsed:
                for field, value in parsed["vitals"].items():
                    if value is not None and hasattr(updated.vitals, field):
                        if field in ("height_cm", "weight_kg", "bmi", "pulse", "spo2"):
                            try:
                                value = float(str(value).replace(",", "."))
                            except (ValueError, TypeError):
                                continue
                        setattr(updated.vitals, field, value)

            # Auto BMI
            if updated.vitals.height_cm and updated.vitals.weight_kg:
                h = updated.vitals.height_cm / 100
                updated.vitals.bmi = round(updated.vitals.weight_kg / (h * h), 1)

            return updated
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("[LLM] Parse error: %s | raw: %s", e, raw_json[:200])
            return fallback
