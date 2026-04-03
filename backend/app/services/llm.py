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

═══ ПРАВИЛА КОРРЕКЦИИ ═══
1. Исправляй фонетически похожие слова на корректные медицинские термины
2. Учитывай КОНТЕКСТ (симптомы, жалобы, диагноз) для выбора наиболее подходящего термина
3. Если слово созвучно с названием препарата И контекст подтверждает — исправляй на препарат
4. НЕ меняй смысл. НЕ перефразируй. Числа оставь как есть.

═══ ПРИМЕРЫ КОРРЕКЦИИ ═══
Простые ASR-ошибки:
нурофон→нурофен, парацетомол→парацетамол, ибопрофен→ибупрофен, \
гастрид→гастрит, гипертания→гипертония, диобед→диабет, тохикардия→тахикардия, \
амоксицилин→амоксициллин, холицистит→холецистит, панкратит→панкреатит, \
кординол→кардиология, мезентерий→мезентерий.

Контекстно-зависимые (распознавание далеко от оригинала):
"те резин" при дерматите/аллергии → "цетиризин"
"на прокс" при боли → "напроксен"
"а мок лав" при инфекции → "амоксиклав"
"лоза ртан" при гипертонии → "лозартан"
"мета формин" при диабете → "метформин"
"о мепра зол" при гастрите → "омепразол"
"карба мазе пин" при эпилепсии/невралгии → "карбамазепин"
"тора семид" при отёках → "торасемид"

═══ ПРИНЦИП ═══
Если слово не является известным словом русского языка, но СОЗВУЧНО с медицинским \
термином или препаратом — исправь его. Используй жалобы и симптомы из контекста \
для определения наиболее вероятного препарата/термина.

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
4. Исправляй ASR-ошибки: нурофон→нурофен, гастрид→гастрит, диобед→диабет
5. Если слово не распознаётся, но созвучно с препаратом/термином — подбирай \
наиболее подходящий вариант по контексту жалоб и симптомов \
(пример: "те резин" при кожных симптомах → "цетиризин")

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

═══ КРИТИЧЕСКИ ВАЖНО: НЕ ТЕРЯЙ ИНФОРМАЦИЮ ═══
1. АЛЛЕРГИИ: Если пациент ЛЮБЫМ образом упоминает аллергию (даже вскользь, даже позже \
в разговоре) — это ОБЯЗАТЕЛЬНО должно попасть в поле allergies. \
НЕЛЬЗЯ писать "Отрицает" если пациент хоть раз упомянул аллергию! \
Пример ОШИБКИ: пациент говорит "у меня аллергия на цитрусовые" → allergies: "Отрицает" — ЭТО НЕДОПУСТИМО. \
Правильно: allergies: "Цитрусовые"
2. ПРЕПАРАТЫ: ВСЕ упомянутые пациентом препараты, витамины, БАДы, травы — записывай в medications. \
"Пью витамины группы Б" → medications: "Витамины группы B". \
"Принимаю омегу" → medications: "Омега-3". \
НЕ ИГНОРИРУЙ витамины, БАДы, гомеопатию — это тоже medications!
3. ОТРИЦАНИЯ: Используй "Отрицает" / "Не принимает" ТОЛЬКО если пациент ЯВНО и ОДНОЗНАЧНО \
сказал что аллергии нет / лекарств не принимает, И при этом НИГДЕ в разговоре не упоминал обратное.
4. ПРОТИВОРЕЧИЯ: Если пациент сначала отрицает, а потом упоминает аллергию/препарат — \
верь ПОЗДНЕМУ высказыванию. Факт важнее отрицания.

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

PATIENT_SUMMARY_PROMPT = """\
Ты — медицинский ассистент. Суммаризируй слова пациента из приёма у врача.

ОБЯЗАТЕЛЬНО СОХРАНИ:
- Точные формулировки жалоб (локализация, характер, иррадиация, шкала боли)
- Хронологию симптомов (когда началось, как менялось)
- Названия лекарств и дозировки
- Аллергии и их проявления
- Семейную историю болезней
- Показатели, которые пациент называл (рост, вес, давление, пульс, сатурация)

Ответь на русском, 15-20 предложений. Пиши от третьего лица ("Пациент жалуется на...")."""

QUALITY_ANALYSIS_PROMPT = """\
Ты — строгий независимый эксперт по оценке качества врачебных приёмов. \
Проанализируй транскрипт диалога врача с пациентом.

═══ КРИТЕРИИ КАЧЕСТВА (0-2 балла каждый) ═══

ВАЖНО: Используй ВСЕ ТРИ оценки (0, 1, 2) в зависимости от реального качества осмотра!
- 0 = тема НЕ затронута вообще, врач полностью пропустил этот аспект
- 1 = тема затронута ПОВЕРХНОСТНО или ЧАСТИЧНО (врач спросил, но не углубился; \
упомянул, но не раскрыл; начал, но не завершил)
- 2 = тема раскрыта ПОЛНОСТЬЮ и ПОДРОБНО (врач задал уточняющие вопросы, получил \
развёрнутые ответы, систематически собрал информацию)

НЕ ЗАВЫШАЙ оценки! Оценка "2" — это ОТЛИЧНАЯ работа, а не просто "упомянул тему". \
Если врач задал один вопрос без уточнений — это "1", а не "2". \
Оценка "1" — самая частая для среднего приёма.

Критерии:
- greeting_and_contact: Приветствие, представление по имени, установление контакта
  0=не поздоровался / не представился; 1=поздоровался но без представления или формально; 2=полноценное приветствие, представление, установление контакта
- conversation_structure: Логичная структура беседы
  0=хаотичный разговор без структуры; 1=есть базовая структура но с перескоками между темами; 2=чёткая последовательная структура (жалобы→анамнез→осмотр→план)
- needs_identification: Выявление потребностей/ожиданий пациента
  0=не спрашивал чего ожидает пациент; 1=формально спросил причину визита; 2=выяснил ожидания, опасения, цели визита
- current_complaints_identification: Выявление текущих жалоб
  0=не расспросил о жалобах; 1=узнал основную жалобу без деталей (локализация, характер, интенсивность); 2=подробно расспросил: локализация, характер, интенсивность (ВАШ), иррадиация, провоцирующие факторы
- disease_history: Анамнез текущего заболевания
  0=не спрашивал когда и как началось; 1=узнал когда началось, но без деталей динамики; 2=подробно: начало, динамика, что уже предпринимал, эффект от лечения
- general_medical_history: Хронические болезни, операции, госпитализации
  0=не спросил о хронических заболеваниях; 1=спросил "чем болеете?" без уточнений; 2=систематически расспросил о хронических болезнях, операциях, госпитализациях
- medication_history: Текущие лекарства, дозировки, регулярность приёма
  0=не спросил о лекарствах; 1=спросил "что принимаете?" без уточнения доз и схемы; 2=узнал конкретные препараты, дозы, кратность, длительность приёма
- family_history: Семейный/наследственный анамнез
  0=не спросил о болезнях в семье; 1=задал общий вопрос о наследственности; 2=расспросил о конкретных заболеваниях у родителей, братьев/сестёр
- prevention_and_risk_control: Образ жизни, курение, алкоголь, физ. активность
  0=не обсуждал образ жизни; 1=спросил об 1-2 факторах; 2=системно обсудил: курение, алкоголь, питание, сон, физ. активность, стресс
- treatment_planning: План лечения/обследований
  0=не озвучил плана; 1=назначил лечение без обоснования или неполный план; 2=подробный план: обследования с обоснованием, лекарства с дозами, консультации
- visit_closure: Подведение итогов, рекомендации, контрольный визит
  0=приём закончился без подведения итогов; 1=кратко резюмировал или назначил повторный визит; 2=подвёл итоги, дал рекомендации, обсудил red flags, назначил контрольный визит

═══ АНАЛИТИКА ДИАЛОГА ═══

Все метрики 0 или 1, КРОМЕ doctor_interrupted_patient (0-2):

- doctor_showed_empathy: Врач проявил эмпатию/сочувствие к пациенту (0/1)
- doctor_interrupted_patient: Врач перебивал пациента (ШКАЛА 0-2):
  2 = врач НЕ перебивал, давал пациенту договорить, выдерживал паузы
  1 = врач иногда перебивал, но в целом давал говорить
  0 = врач систематически перебивал пациента, не давал закончить мысль
- patient_asked_questions: Пациент задавал вопросы (0/1)
- doctor_used_medical_jargon: Врач использовал медицинский жаргон без пояснений (0/1)
- doctor_confirmed_understanding: Врач уточнял, правильно ли понял пациента (0/1)
- lifestyle_discussed: Обсуждался образ жизни (питание, сон, стресс, физ. активность) (0/1)
- allergies_discussed: Обсуждались аллергии (0/1)
- shared_decision_making: Совместное принятие решений о лечении (0/1)
- patient_compliance_assessment: Оценка приверженности пациента лечению (0/1)
- doctor_pacing: Врач соблюдал комфортный темп разговора (0/1)

Ответь СТРОГО JSON:
{{
  "quality_criteria": {{
    "greeting_and_contact": 0, "conversation_structure": 0, "needs_identification": 0,
    "current_complaints_identification": 0, "disease_history": 0, "general_medical_history": 0,
    "medication_history": 0, "family_history": 0, "prevention_and_risk_control": 0,
    "treatment_planning": 0, "visit_closure": 0
  }},
  "dialogue_analytics": {{
    "doctor_showed_empathy": 0, "doctor_interrupted_patient": 0, "patient_asked_questions": 0,
    "doctor_used_medical_jargon": 0, "doctor_confirmed_understanding": 0, "lifestyle_discussed": 0,
    "allergies_discussed": 0, "shared_decision_making": 0, "patient_compliance_assessment": 0,
    "doctor_pacing": 0
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
        self._fallback_client: httpx.AsyncClient | None = None
        self._is_ollama: bool = False
        self._is_gigachat: bool = False
        self._call_lock: asyncio.Lock = asyncio.Lock()
        self._http_sem: asyncio.Semaphore = asyncio.Semaphore(settings.llm_max_concurrent)
        self._last_call_time: float = 0.0
        self._rate_limited_until: float = 0.0
        self._gigachat_token: str | None = None
        self._gigachat_token_expires: float = 0.0

    def initialize(self) -> None:
        self._is_ollama = settings.llm_provider == "ollama"
        self._is_gigachat = settings.llm_provider == "gigachat"
        headers = {"Content-Type": "application/json"}
        if not self._is_ollama and not self._is_gigachat and settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        self._client = httpx.AsyncClient(
            base_url=settings.llm_base_url, headers=headers,
            timeout=120.0 if self._is_ollama else 60.0,
            verify=not self._is_gigachat,
        )
        logger.info("[LLM] Initialized: provider=%s model=%s", settings.llm_provider, settings.llm_model)

        # Ollama fallback (when primary is not Ollama)
        if not self._is_ollama:
            self._fallback_client = httpx.AsyncClient(
                base_url=f"{settings.ollama_base_url}/v1",
                headers={"Content-Type": "application/json"},
                timeout=120.0,
            )
            logger.info("[LLM] Fallback initialized: ollama/%s at %s",
                        settings.ollama_model, settings.ollama_base_url)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
        if self._fallback_client:
            await self._fallback_client.aclose()

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

    async def pull_fallback_model(self) -> None:
        """Pull Ollama fallback model (called at startup when primary is Groq)."""
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                resp = await client.post(
                    f"{settings.ollama_base_url}/api/pull",
                    json={"name": settings.ollama_model, "stream": False},
                )
                if resp.status_code == 200:
                    logger.info("[LLM-FALLBACK] Model '%s' ready", settings.ollama_model)
                else:
                    logger.warning("[LLM-FALLBACK] Pull: %d %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logger.warning("[LLM-FALLBACK] Pull failed (non-critical): %s", e)

    async def _refresh_gigachat_token(self) -> None:
        """Refresh GigaChat access token via OAuth (30 min TTL)."""
        try:
            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                resp = await client.post(
                    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                    headers={
                        "Authorization": f"Basic {settings.gigachat_auth_key}",
                        "RqUID": str(__import__('uuid').uuid4()),
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    data={"scope": "GIGACHAT_API_PERS"},
                )
                resp.raise_for_status()
                data = resp.json()
                if "access_token" not in data:
                    raise RuntimeError(f"GigaChat OAuth: missing access_token in response: {str(data)[:200]}")
                self._gigachat_token = data["access_token"]
                # Token TTL ~30 min; refresh at 90% to avoid expiry mid-request
                expires_at_ms = data.get("expires_at", 0)
                if expires_at_ms:
                    ttl_s = (expires_at_ms / 1000) - time.time()
                    self._gigachat_token_expires = time.monotonic() + (ttl_s * 0.9)
                else:
                    self._gigachat_token_expires = time.monotonic() + 1620  # 27 min fallback
                logger.info("[LLM] GigaChat token refreshed, expires in %.0fs",
                            self._gigachat_token_expires - time.monotonic())
        except Exception as e:
            logger.error("[LLM] GigaChat token refresh failed: %s", e)
            raise

    async def _chat_fallback(self, system: str, user: str, temperature: float = 0.2) -> str:
        """Fallback to local Ollama when primary LLM fails. No retry needed (local)."""
        if not self._fallback_client:
            raise RuntimeError("Fallback not available")
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        t0 = time.monotonic()
        resp = await self._fallback_client.post("/chat/completions", json=payload)
        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[LLM-FALLBACK] Response: %d in %dms", resp.status_code, ms)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        if usage:
            logger.info("[LLM-FALLBACK] Tokens: %s/%s/%s",
                        usage.get("prompt_tokens", "?"),
                        usage.get("completion_tokens", "?"),
                        usage.get("total_tokens", "?"))
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as e:
            logger.error("[LLM-FALLBACK] Unexpected response structure: %s", str(data)[:300])
            raise RuntimeError(f"LLM fallback response missing expected fields: {e}")

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
        retried_auth = False

        for attempt in range(max_retries):
            # Hard cap: don't retry longer than 30s total
            if time.monotonic() - retry_start > 30.0:
                logger.warning("[LLM] Retry time limit exceeded (30s), giving up")
                break

            # SHORT lock: rate-limit check, token refresh, gap enforcement
            should_break = False
            async with self._call_lock:
                # Wait for rate limit cooldown if another call got 429
                now = time.monotonic()
                if now < self._rate_limited_until:
                    cooldown = self._rate_limited_until - now
                    if (now - retry_start) + cooldown > 30.0:
                        logger.warning("[LLM] Cooldown %.1fs would exceed 30s budget", cooldown)
                        should_break = True
                    else:
                        logger.info("[LLM] Waiting %.1fs for rate-limit cooldown", cooldown)
                        await asyncio.sleep(cooldown)

                if not should_break:
                    # GigaChat: refresh OAuth token if expired
                    if self._is_gigachat and time.monotonic() >= self._gigachat_token_expires:
                        await self._refresh_gigachat_token()
                        self._client.headers["Authorization"] = f"Bearer {self._gigachat_token}"

                    # Configurable gap between calls (0 for GigaChat, 2.0 for Groq)
                    if settings.llm_min_gap_seconds > 0:
                        elapsed = time.monotonic() - self._last_call_time
                        if elapsed < settings.llm_min_gap_seconds:
                            await asyncio.sleep(settings.llm_min_gap_seconds - elapsed)

                    self._last_call_time = time.monotonic()

            if should_break:
                break

            # HTTP request with concurrency limit (no gap, just max parallel requests)
            t0 = time.monotonic()
            async with self._http_sem:
                try:
                    resp = await self._client.post("/chat/completions", json=payload)
                except httpx.TimeoutException:
                    logger.warning("[LLM] Timeout attempt %d/%d", attempt + 1, max_retries)
                    if attempt < max_retries - 1:
                        continue
                    raise

            ms = int((time.monotonic() - t0) * 1000)

            # Rate limit: SHORT lock to update shared cooldown state
            if resp.status_code in (403, 429):
                async with self._call_lock:
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
                continue

            # GigaChat 401: token expired mid-session — refresh and retry once
            if resp.status_code == 401 and self._is_gigachat and not retried_auth:
                async with self._call_lock:
                    logger.info("[LLM] GigaChat 401, refreshing token...")
                    await self._refresh_gigachat_token()
                    self._client.headers["Authorization"] = f"Bearer {self._gigachat_token}"
                retried_auth = True
                continue

            # Success
            logger.info("[LLM] Response: %d in %dms", resp.status_code, ms)
            resp.raise_for_status()
            data = resp.json()
            usage = data.get("usage", {})
            if usage:
                logger.info("[LLM] Tokens: %s/%s/%s",
                            usage.get("prompt_tokens", "?"),
                            usage.get("completion_tokens", "?"),
                            usage.get("total_tokens", "?"))
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError, TypeError) as e:
                logger.error("[LLM] Unexpected response structure: %s", str(data)[:300])
                raise RuntimeError(f"LLM response missing expected fields: {e}")

        # All retries exhausted
        if resp is not None:
            logger.error("[LLM] Final response: %d %s", resp.status_code, resp.text[:200])
        raise RuntimeError(f"LLM retries exhausted (last status: {resp.status_code if resp else 'timeout'})")

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
            if self._fallback_client:
                try:
                    logger.info("[LLM-FALLBACK] Trying Ollama for extraction...")
                    raw = await self._chat_fallback(EXTRACTION_PROMPT, user_prompt)
                    return self._merge_protocol(raw, current)
                except Exception as e2:
                    logger.error("[LLM-FALLBACK] Also failed: %s", e2)
            raise

    # ---- Patient speech summarization (for long sessions) ----

    async def summarize_patient_speech(self, patient_text: str) -> str:
        """Summarize all patient utterances into a compact medical narrative."""
        logger.info("[LLM] Summarizing patient speech (%d chars)...", len(patient_text))
        return await self._chat(PATIENT_SUMMARY_PROMPT, patient_text, json_mode=False)

    # ---- Quality analysis ----

    async def analyze_quality(self, transcript: str) -> dict:
        """Analyze transcript for consultation quality criteria and dialogue analytics."""
        logger.info("[LLM] Analyzing consultation quality...")
        try:
            raw = await self._chat(QUALITY_ANALYSIS_PROMPT, transcript, temperature=0.1)
            return self._parse_json(raw) or {}
        except Exception as e:
            logger.error("[LLM] Quality analysis failed: %s", e)
            return {}

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
            if self._fallback_client:
                try:
                    logger.info("[LLM-FALLBACK] Trying Ollama for finalization...")
                    raw = await self._chat_fallback(system, "Сформируй заключение на русском языке.", temperature=0.3)
                    return self._merge_protocol(raw, current)
                except Exception as e2:
                    logger.error("[LLM-FALLBACK] Also failed: %s", e2)
            return current

    # ---- JSON parser with fallbacks ----

    def _parse_json(self, raw: str) -> dict | None:
        if not raw:
            return None
        # GigaChat wraps JSON in double braces: {{ "key": {{ ... }} }}
        cleaned = raw.strip()
        if "{{" in cleaned:
            cleaned = cleaned.replace("{{", "{").replace("}}", "}")
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
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
                    if field in ("medications", "allergies"):
                        new_norm = new_val.lower().replace("—", "-").replace("–", "-").strip()
                        negation_phrases = ("отрицает", "не принимает", "нет", "отсутствуют")
                        new_is_negation = any(neg in new_norm for neg in negation_phrases)

                        if current_val:
                            current_norm = current_val.lower().replace("—", "-").replace("–", "-").strip()
                            current_is_negation = any(neg in current_norm for neg in negation_phrases)

                            if new_is_negation and not current_is_negation:
                                # Don't overwrite real data with negation
                                continue
                            elif not new_is_negation and current_is_negation:
                                # Replace negation with real data (patient corrected themselves)
                                setattr(updated.exam_data, field, new_val)
                            elif new_norm not in current_norm:
                                setattr(updated.exam_data, field, f"{current_val}; {new_val}")
                        else:
                            setattr(updated.exam_data, field, new_val)
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
                        # Extract "120/80" from strings like "150/95 мм рт. ст." or "130/85|null"
                        bp_match = re.search(r"(\d{2,3}/\d{2,3})", bp)
                        if bp_match:
                            setattr(updated.vitals, field, bp_match.group(1))
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
