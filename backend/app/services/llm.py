"""LLM service v3: Groq (default) / Ollama, Russian prompts, retry with backoff."""

import asyncio
import json
import logging
import random
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
Исправь ошибки распознавания речи (ASR) в медицинских терминах и названиях лекарств.

Примеры: нурофон→нурофен, парацетомол→парацетамол, ибопрофен→ибупрофен, \
гастрид→гастрит, гипертания→гипертония, диобед→диабет, тохикардия→тахикардия, \
амоксицилин→амоксициллин, холицистит→холецистит, панкратит→панкреатит, \
кординол→кардиология, мезентерий→мезентерий, апатия→апатия.

НЕ меняй смысл. НЕ перефразируй. Числа оставь как есть.

Вход: {"items": ["текст 1", "текст 2"]}
Выход: {"items": ["исправленный 1", "исправленный 2"]}

Ответь ТОЛЬКО валидным JSON."""

CALIBRATION_PROMPT = """\
Из текста пациента извлеки: ФИО, возраст (число), пол.

═══ ПРИМЕРЫ ═══
Текст: "Меня зовут Иванова Мария Петровна. Мне сорок пять лет. Женский."
→ {{"full_name": "Иванова Мария Петровна", "age": 45, "gender": "Ж"}}

Текст: "Петров Алексей, тридцать восемь, мужчина"
→ {{"full_name": "Петров Алексей", "age": 38, "gender": "М"}}

Текст: "Зовут Ольга Сергеевна"
→ {{"full_name": "Ольга Сергеевна", "age": null, "gender": null}}

═══ ПРАВИЛА ═══
1. age — ТОЛЬКО целое число. "Сорок пять" → 45, "тридцать два" → 32
2. gender: мужской/мужчина → "М", женский/женщина → "Ж", неизвестно → null
3. full_name — как пациент назвал (ФИО, ИО, Ф — что дали)
4. Если данных нет — null

Ответь СТРОГО JSON: {{"full_name": "...", "age": число|null, "gender": "М"|"Ж"|null}}"""

EXTRACTION_PROMPT = """\
Ты — опытный врач, заполняющий медкарту по разговору с пациентом.

═══ ПРАВИЛА ═══
1. Извлекай ТОЛЬКО из реплик ПАЦИЕНТА (реплики врача — контекст)
2. НЕ выдумывай данных
3. Сохраняй ранее заполненные поля — НЕ удаляй и НЕ обнуляй
4. Исправляй ASR: нурофон→нурофен, гастрид→гастрит, диобед→диабет

═══ ЧИСЛА ═══
Пациент говорит словами — преобразуй в цифры:
- "сто тридцать на восемьдесят пять" → systolic_bp: "130/85"
- "сто семьдесят восемь сантиметров" → height_cm: 178
- "восемьдесят два килограмма" → weight_kg: 82
- "семьдесят два удара" → pulse: 72

═══ РАЗДЕЛЕНИЕ ПОЛЕЙ (СТРОГО) ═══
- complaints: ГЛАВНАЯ жалоба ("головная боль")
- complaints_details: детали — локализация, ВАШ, характер, когда усиливается
  Стиль: "Цефалгия (головная боль) в височной области, 6-7/10 по ВАШ, усиливается к вечеру и при длительном сидении"
- anamnesis: история ТЕКУЩЕГО заболевания ("началось 2 недели назад после простуды")
- life_anamnesis: хронические болезни + привычки + семейный анамнез
  Пример: "стоит на учёте лет 5, аллергия на амоксициллин, не курю, сплю плохо, стресс на работе"
  ВКЛЮЧИТЬ сюда семью: "Наследственный анамнез: мать — гипертония, диабет; отец — инфаркт в 55 лет"
- allergies: аллергии ТОЛЬКО пациента ("Амоксициллин — сыпь")
- medications: лекарства ТОЛЬКО пациента ("Ибупрофен (Нурофен) 400 мг")

ЗАПРЕЩЕНО: болезни родственников в complaints / anamnesis!

═══ ОТРИЦАНИЯ ═══
- "не аллергик" → allergies: "Отрицает"
- "лекарств не принимаю" → medications: "Не принимает"

Ответь СТРОГО JSON:
{{
  "exam_data": {{
    "complaints": "строка|null",
    "complaints_details": "строка|null",
    "anamnesis": "строка|null",
    "life_anamnesis": "строка|null",
    "allergies": "строка|null",
    "medications": "строка|null"
  }},
  "vitals": {{
    "height_cm": число|null, "weight_kg": число|null,
    "pulse": число|null, "spo2": число|null,
    "systolic_bp": "строка NNN/NN|null"
  }}
}}"""

FINALIZATION_PROMPT = """\
Ты — опытный врач. Сформируй заключение на основе данных пациента и разговора.
Отвечай ТОЛЬКО на русском языке.

═══ ДАННЫЕ ПАЦИЕНТА ═══
{protocol_json}

═══ ТРАНСКРИПЦИЯ РАЗГОВОРА ═══
{transcript}

═══ ТРЕБОВАНИЯ ═══
1. ДИАГНОЗ: "Предварительный диагноз: [МКБ-10 код] Название"
   Пример: "R51 Цефалгия напряжённого типа. Дифф. диагноз: G43 Мигрень"

2. ПЛАН ЛЕЧЕНИЯ:
   - Обследования: ОАК, БАК, ЭКГ (с обоснованием)
   - Лекарства: МНН (торговое) доза × кратность, длительность
   - Консультации: какие специалисты, срочность

3. РЕКОМЕНДАЦИИ ПАЦИЕНТУ (простым языком):
   - Что делать и что не делать
   - RED FLAGS: "Немедленно к врачу если: [конкретные симптомы]"
   - Когда повторный осмотр

═══ ПРАВИЛА ═══
- Перепиши ранее заполненные поля в корректном медицинском стиле, НЕ теряя данных
- МКБ-10: буква + 2 цифры (R51, I10, G43 и т.д.)

Ответь СТРОГО в JSON:
{{
  "exam_data": {{
    "complaints": "мед. стиль",
    "complaints_details": "мед. стиль, подробно",
    "anamnesis": "мед. стиль",
    "life_anamnesis": "мед. стиль",
    "allergies": "как есть",
    "medications": "МНН (торговое), дозы",
    "diagnosis": "Предварительный: МКБ-10 Название",
    "treatment_plan": "Обследования: ...\\nЛекарства: ...\\nКонсультации: ...",
    "patient_recommendations": "Простым языком. RED FLAGS: ..."
  }},
  "vitals": {{
    "height_cm": число, "weight_kg": число, "pulse": число,
    "spo2": число, "systolic_bp": "строка"
  }}
}}"""


class LLMService:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None
        self._is_ollama: bool = False
        self._call_lock: asyncio.Lock = asyncio.Lock()
        self._last_call_time: float = 0.0
        self._rate_limited_until: float = 0.0

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

        max_retries = 5
        retry_start = time.monotonic()
        resp = None

        for attempt in range(max_retries):
            # Hard cap: don't retry longer than 30s total
            if time.monotonic() - retry_start > 30.0:
                logger.warning("[LLM] Retry time limit exceeded (30s), giving up")
                break

            # Lock for HTTP request + rate limit cooldown (prevents concurrent 429 races)
            async with self._call_lock:
                # Wait for rate limit cooldown if another call got 429
                now = time.monotonic()
                if now < self._rate_limited_until:
                    cooldown = self._rate_limited_until - now
                    if (now - retry_start) + cooldown > 30.0:
                        logger.warning("[LLM] Cooldown %.1fs would exceed 30s budget", cooldown)
                        break
                    logger.info("[LLM] Waiting %.1fs for rate-limit cooldown", cooldown)
                    await asyncio.sleep(cooldown)

                # Min 2s gap between calls
                elapsed = time.monotonic() - self._last_call_time
                if elapsed < 2.0:
                    await asyncio.sleep(2.0 - elapsed)

                t0 = time.monotonic()
                try:
                    resp = await self._client.post("/chat/completions", json=payload)
                except httpx.TimeoutException:
                    self._last_call_time = time.monotonic()
                    logger.warning("[LLM] Timeout attempt %d/%d", attempt + 1, max_retries)
                    if attempt < max_retries - 1:
                        continue
                    raise

                self._last_call_time = time.monotonic()
                ms = int((time.monotonic() - t0) * 1000)

                # Rate limit: set cooldown INSIDE lock so other callers see it
                if resp.status_code in (403, 429):
                    retry_after = resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = min(float(retry_after), 8.0)
                        except ValueError:
                            wait = min(2 ** attempt, 8)
                    else:
                        wait = min(2 ** attempt, 8)
                    wait += random.uniform(0, 0.5)
                    self._rate_limited_until = time.monotonic() + wait
                    logger.warning("[LLM] Rate limited (%d), cooldown %.1fs (attempt %d/%d)",
                                   resp.status_code, wait, attempt + 1, max_retries)
                    continue  # Next iteration sleeps inside lock via cooldown check

            # Success (outside lock)
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

        # All retries exhausted
        if resp is not None:
            resp.raise_for_status()
        raise RuntimeError("LLM retries exhausted")

    # ---- Medical correction (JSON-based, not separator-based) ----

    async def correct_medical_terms_batch(self, texts: list[str]) -> list[str] | None:
        """Correct medical terms in a batch of texts. Returns corrected list or None on failure."""
        try:
            payload = json.dumps({"items": texts}, ensure_ascii=False)
            raw = await self._chat(MEDICAL_CORRECTION_PROMPT, payload, json_mode=True)
            parsed = self._parse_json(raw)
            if parsed and "items" in parsed and isinstance(parsed["items"], list):
                corrected = parsed["items"]
                if len(corrected) == len(texts):
                    for i, (orig, corr) in enumerate(zip(texts, corrected)):
                        if orig != corr:
                            logger.info("[LLM] Corrected[%d]: '%s' → '%s'", i, orig[:50], corr[:50])
                    return corrected
                logger.warning("[LLM] Correction count mismatch: got %d, expected %d", len(corrected), len(texts))
            return None
        except Exception as e:
            logger.warning("[LLM] Correction failed: %s", e)
            return None

    # Keep old method for backward compatibility but deprecated
    async def correct_medical_terms(self, text: str) -> str:
        result = await self.correct_medical_terms_batch([text])
        return result[0] if result else text

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
        self, current: MedicalProtocol, utterances: list[Utterance], context: str,
    ) -> MedicalProtocol:
        new_text = "\n".join(
            f"[{'Врач' if u.speaker == 'doctor' else 'Пациент'}]: {u.text}"
            for u in utterances
        )
        user_prompt = (
            f"ТЕКУЩИЙ ПРОТОКОЛ:\n{current.model_dump_json(indent=2)}\n\n"
            f"КОНТЕКСТ (последние реплики):\n{context}\n\n"
            f"НОВЫЙ ФРАГМЕНТ (извлеки данные отсюда):\n{new_text}\n\n"
            "Извлеки данные из реплик ПАЦИЕНТА. Сохрани заполненные поля."
        )
        try:
            raw = await self._chat(EXTRACTION_PROMPT, user_prompt)
            return self._merge_protocol(raw, current)
        except Exception as e:
            logger.error("[LLM] Extraction error: %s", e)
            raise

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
                    if not hasattr(updated.exam_data, field):
                        continue
                    current_val = getattr(updated.exam_data, field, None)
                    new_val = None
                    if value:
                        if isinstance(value, dict):
                            new_val = "; ".join(f"{k}: {v}" for k, v in value.items() if v)
                        elif isinstance(value, list):
                            new_val = "; ".join(str(v) for v in value)
                        else:
                            new_val = str(value).strip()

                    if not new_val:
                        continue  # Don't overwrite existing data with empty

                    # For medications/allergies: append if new info, don't replace
                    if current_val and field in ("medications", "allergies"):
                        existing_norm = current_val.lower().replace("—", "-").replace("–", "-")
                        new_norm = new_val.lower().replace("—", "-").replace("–", "-")
                        if new_norm not in existing_norm:
                            setattr(updated.exam_data, field, f"{current_val}; {new_val}")
                    else:
                        setattr(updated.exam_data, field, new_val)

            if "vitals" in parsed:
                for field, value in parsed["vitals"].items():
                    if value is None or not hasattr(updated.vitals, field):
                        continue
                    if field in ("height_cm", "weight_kg", "bmi", "pulse", "spo2"):
                        try:
                            num = float(str(value).replace(",", "."))
                            # Sanity checks
                            if field == "height_cm" and not (50 < num < 250):
                                continue
                            if field == "weight_kg" and not (2 < num < 300):
                                continue
                            if field == "pulse" and not (30 < num < 250):
                                continue
                            if field == "spo2" and not (50 < num < 101):
                                continue
                            setattr(updated.vitals, field, num)
                        except (ValueError, TypeError):
                            logger.warning("[LLM] Failed to parse %s: %s", field, value)
                            continue
                    elif field == "systolic_bp":
                        bp = str(value).strip()
                        if re.match(r"^\d{2,3}/\d{2,3}$", bp):
                            setattr(updated.vitals, field, bp)
                        else:
                            logger.warning("[LLM] Invalid BP format: %s", value)
            # Auto BMI
            if updated.vitals.height_cm and updated.vitals.weight_kg:
                h = updated.vitals.height_cm / 100
                updated.vitals.bmi = round(updated.vitals.weight_kg / (h * h), 1)
            return updated
        except Exception as e:
            logger.error("[LLM] Merge error: %s", e)
            return fallback
