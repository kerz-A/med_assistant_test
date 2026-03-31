"""LLM service v3: Groq (default) / Ollama, Russian prompts, retry with backoff."""

import asyncio
import json
import logging
import re
import time

import httpx

from ..config import settings
from ..models.messages import Utterance
from ..models.protocol import MedicalProtocol

logger = logging.getLogger(__name__)

# ============================================================
# ПРОМПТЫ (русские)
# ============================================================

MEDICAL_CORRECTION_PROMPT = """\
Исправь ошибки распознавания речи в медицинских терминах и названиях лекарств.
Примеры: нурофон→нурофен, парацетомол→парацетамол, ибопрофен→ибупрофен, \
гастрид→гастрит, гипертания→гипертония, диобед→диабет, тохикардия→тахикардия.
НЕ меняй смысл. НЕ перефразируй. Числа оставь как есть.
Если текст содержит несколько фрагментов через " ||| " — исправь каждый отдельно, сохраняя разделитель.
Ответь ТОЛЬКО исправленным текстом, без пояснений."""

CALIBRATION_PROMPT = """\
Из текста извлеки данные пациента: ФИО, возраст (число) и пол.
Ответь СТРОГО в формате JSON, без другого текста:
{{"full_name": "Фамилия Имя Отчество", "age": 45, "gender": "Ж"}}
Если данных нет — используй null. Пол: "М" для мужского, "Ж" для женского."""

EXTRACTION_PROMPT = """\
Ты — квалифицированный врач, заполняющий медицинскую карту пациента \
на основе разговора врача и пациента.

═══ ПРАВИЛА ═══
1. Извлекай данные ТОЛЬКО из реплик ПАЦИЕНТА. Реплики врача — контекст.
2. НЕ выдумывай данные. Записывай только то, что пациент явно сказал.
3. Сохраняй ранее заполненные поля — НЕ удаляй и НЕ обнуляй.
4. Исправляй ошибки ASR: нурофон→нурофен, ибопрофен→ибупрофен.

═══ МЕДИЦИНСКИЙ СТИЛЬ ═══
- Используй мед. терминологию: "Цефалгия (головная боль) в височной области, 6/10 по ВАШ"
- Лекарства: МНН (торговое), дозировка. Пример: "Ибупрофен (Нурофен) 400 мг"
- Давление: строка "130/85", НЕ число.

═══ СТРОГОЕ РАЗДЕЛЕНИЕ ═══
- complaints — ТОЛЬКО жалобы САМОГО пациента (основная причина обращения)
- complaints_details — детали жалоб: локализация, характер, интенсивность
- anamnesis — история ТЕКУЩЕГО заболевания пациента
- life_anamnesis — хронические болезни пациента + привычки + "Наследственный анамнез: мать — ..., отец — ..."
- allergies — аллергии САМОГО пациента
- medications — лекарства САМОГО пациента
НИКОГДА не записывай болезни родственников в complaints или anamnesis!

Ответь СТРОГО в JSON (ТОЛЬКО на русском языке):
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
    "height_cm": null,
    "weight_kg": null,
    "pulse": null,
    "spo2": null,
    "systolic_bp": null
  }}
}}"""

FINALIZATION_PROMPT = """\
Ты — квалифицированный врач. На основе собранного анамнеза сформируй заключение.
Отвечай ТОЛЬКО на русском языке.

═══ ДАННЫЕ ПАЦИЕНТА ═══
{protocol_json}

═══ ТРАНСКРИПЦИЯ РАЗГОВОРА ═══
{transcript}

═══ ТРЕБОВАНИЯ ═══
1. ДИАГНОЗ: предварительный, с кодом МКБ-10. Дифференциальный ряд если неоднозначно.
2. ПЛАН ЛЕЧЕНИЯ: конкретные обследования (ОАК, БАК, ЭКГ), препараты с дозировками, консультации специалистов.
3. РЕКОМЕНДАЦИИ: простым языком для пациента, тревожные симптомы (red flags).

Ранее заполненные поля перепиши в корректном медицинском стиле, сохраняя все данные.

Ответь СТРОГО в JSON (на русском):
{{
  "exam_data": {{
    "complaints": "мед. стиль",
    "complaints_details": "мед. стиль, подробно",
    "anamnesis": "мед. стиль",
    "life_anamnesis": "мед. стиль",
    "allergies": "строка",
    "medications": "МНН (торговое), дозировка",
    "diagnosis": "Предварительный: код МКБ-10 Название",
    "treatment_plan": "конкретный план",
    "patient_recommendations": "простым языком"
  }},
  "vitals": {{
    "height_cm": число, "weight_kg": число, "pulse": число, "spo2": число, "systolic_bp": "строка 120/80"
  }}
}}"""


class LLMService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._is_ollama: bool = False

    def initialize(self) -> None:
        self._is_ollama = settings.llm_provider == "ollama"
        headers = {"Content-Type": "application/json"}
        if not self._is_ollama and settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url, headers=headers,
            timeout=120.0 if self._is_ollama else 60.0,
        )
        logger.info("[LLM] Initialized: provider=%s model=%s", settings.llm_provider, settings.llm_model)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()

    async def pull_ollama_model(self) -> None:
        if not self._is_ollama:
            return
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/pull",
                    json={"name": settings.llm_model, "stream": False},
                )
                if resp.status_code == 200:
                    logger.info("[LLM] Ollama model '%s' ready", settings.llm_model)
                else:
                    logger.warning("[LLM] Ollama pull: %d %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.error("[LLM] Ollama pull failed: %s", e)

    # ---- Core chat with retry ----

    async def _chat(self, system: str, user: str, temperature: float = 0.2, json_mode: bool = True) -> str:
        if not self._client:
            raise RuntimeError("LLM not initialized")

        payload: dict = {
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if json_mode and not self._is_ollama:
            payload["response_format"] = {"type": "json_object"}
        if not json_mode:
            payload["max_tokens"] = 500

        for attempt in range(3):
            t0 = time.monotonic()
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                ms = int((time.monotonic() - t0) * 1000)

                if resp.status_code in (403, 429):
                    wait = 2 ** attempt
                    logger.warning("[LLM] Rate limited (%d), retry %d/3 in %ds", resp.status_code, attempt + 1, wait)
                    await asyncio.sleep(wait)
                    continue

                logger.info("[LLM] Response: %d in %dms", resp.status_code, ms)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                if usage:
                    logger.info("[LLM] Tokens: %s/%s/%s",
                                usage.get("prompt_tokens", "?"),
                                usage.get("completion_tokens", "?"),
                                usage.get("total_tokens", "?"))
                return data["choices"][0]["message"]["content"].strip()

            except httpx.TimeoutException:
                logger.warning("[LLM] Timeout attempt %d/3", attempt + 1)
                if attempt < 2:
                    await asyncio.sleep(1)
                    continue
                raise

        resp.raise_for_status()
        return ""

    # ---- Medical correction ----

    async def correct_medical_terms(self, text: str) -> str:
        try:
            corrected = await self._chat(MEDICAL_CORRECTION_PROMPT, text, json_mode=False)
            if corrected and len(corrected) > 3:
                corrected = corrected.strip('"\'')
                if corrected != text:
                    logger.info("[LLM] Corrected: '%s' → '%s'", text[:50], corrected[:50])
                return corrected
            return text
        except Exception as e:
            logger.warning("[LLM] Correction failed: %s", e)
            return text

    # ---- Calibration ----

    async def extract_patient_info(self, text: str) -> dict:
        try:
            raw = await self._chat(CALIBRATION_PROMPT, text)
            parsed = self._parse_json(raw)
            if not parsed:
                return {}
            if parsed.get("age") and isinstance(parsed["age"], str):
                digits = "".join(c for c in parsed["age"] if c.isdigit())
                parsed["age"] = int(digits) if digits else None
            logger.info("[LLM] Calibration: name=%s age=%s gender=%s",
                        parsed.get("full_name"), parsed.get("age"), parsed.get("gender"))
            return parsed
        except Exception as e:
            logger.error("[LLM] Calibration error: %s", e)
            return {}

    # ---- Protocol extraction (every cycle during recording) ----

    async def extract_protocol_data(
        self, current: MedicalProtocol, utterances: list[Utterance], full_transcript: str,
    ) -> MedicalProtocol:
        new_text = "\n".join(
            f"[{'Врач' if u.speaker == 'doctor' else 'Пациент'}]: {u.text}"
            for u in utterances
        )
        user_prompt = (
            f"ТЕКУЩИЙ ПРОТОКОЛ:\n{current.model_dump_json(indent=2)}\n\n"
            f"ПОЛНАЯ ИСТОРИЯ РАЗГОВОРА (для контекста):\n{full_transcript}\n\n"
            f"НОВЫЙ ФРАГМЕНТ (извлеки данные отсюда):\n{new_text}\n\n"
            "Извлеки данные из реплик ПАЦИЕНТА. Сохраняй ранее заполненные поля."
        )
        try:
            raw = await self._chat(EXTRACTION_PROMPT, user_prompt)
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
            raw = await self._chat(system, "Сформируй заключение на русском языке.", temperature=0.3)
            return self._merge_protocol(raw, current)
        except Exception as e:
            logger.error("[LLM] Finalization error: %s", e)
            return current

    # ---- JSON parser with fallbacks ----

    def _parse_json(self, raw: str) -> dict | None:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = raw[start:end]
            json_str = re.sub(r",\s*}", "}", json_str)
            json_str = re.sub(r",\s*]", "]", json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        logger.error("[LLM] JSON parse failed: %s", raw[:300])
        return None

    def _merge_protocol(self, raw: str, fallback: MedicalProtocol) -> MedicalProtocol:
        parsed = self._parse_json(raw)
        if not parsed:
            return fallback
        try:
            updated = fallback.model_copy(deep=True)
            if "exam_data" in parsed:
                for field, value in parsed["exam_data"].items():
                    if value and hasattr(updated.exam_data, field):
                        if isinstance(value, dict):
                            value = "; ".join(f"{k}: {v}" for k, v in value.items() if v)
                        elif isinstance(value, list):
                            value = "; ".join(str(v) for v in value)
                        setattr(updated.exam_data, field, str(value))
            if "vitals" in parsed:
                for field, value in parsed["vitals"].items():
                    if value is not None and hasattr(updated.vitals, field):
                        if field in ("height_cm", "weight_kg", "bmi", "pulse", "spo2"):
                            try:
                                value = float(str(value).replace(",", "."))
                            except (ValueError, TypeError):
                                continue
                        elif field == "systolic_bp":
                            value = str(value)  # Always string for BP
                        setattr(updated.vitals, field, value)
            # Auto BMI
            if updated.vitals.height_cm and updated.vitals.weight_kg:
                h = updated.vitals.height_cm / 100
                updated.vitals.bmi = round(updated.vitals.weight_kg / (h * h), 1)
            return updated
        except Exception as e:
            logger.error("[LLM] Merge error: %s", e)
            return fallback
