# Aiston TT v3 — План разработки

## Концепция

Realtime-система заполнения медицинского протокола осмотра.
Врач ведёт приём пациента — система слушает, транскрибирует, разделяет
роли (врач/пациент), извлекает данные и заполняет протокол на лету.
После окончания записи врач правит поля вручную, затем система
генерирует диагноз, план лечения и рекомендации.

---

## User Flow

```
1. Врач нажимает [Начать приём]
2. КАЛИБРОВКА: врач спрашивает "Как вас зовут? Сколько вам лет?"
   → система определяет голос врача (speaker ID)
   → распознаёт ФИО и возраст пациента
   → заполняет поля: patient_name, patient_age
3. ЗАПИСЬ ПРИЁМА: непрерывная запись разговора
   → каждые 8 сек: ASR + диаризация + коррекция + извлечение
   → транскрипция в реальном времени (врач синим, пациент зелёным)
   → поля протокола заполняются автоматически (read-only)
4. Врач нажимает [Стоп]
   → запись останавливается
   → поля становятся редактируемыми (textarea)
   → врач правит если нужно
5. Врач нажимает [Сформировать заключение]
   → LLM генерирует: диагноз, план лечения, рекомендации
   → поля заключения заполняются (тоже редактируемые)
6. Врач нажимает [Экспорт] (будущее)
   → PDF / печать
```

---

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│  Frontend                                            │
│                                                      │
│  ┌──────────────────┐  ┌──────────────────────────┐  │
│  │ Транскрипция      │  │ Протокол                  │  │
│  │ (scroll, realtime)│  │                           │  │
│  │                   │  │ ФИО: [Иванов И.И.]       │  │
│  │ 👨‍⚕️ Как вас зовут? │  │ Возраст: [45]            │  │
│  │ 🤒 Иванов Иван.. │  │ Жалобы: [auto-fill...]   │  │
│  │ 👨‍⚕️ Что беспокоит?│  │ Анамнез: [auto-fill...]  │  │
│  │ 🤒 Голова болит.. │  │ Витальные: [___]          │  │
│  │                   │  │ ...                       │  │
│  └──────────────────┘  │ Диагноз: [после финализ.] │  │
│                         │ Лечение: [после финализ.] │  │
│  ● Запись  ■ Стоп       │ Рекоменд: [после финализ.]│  │
│  ✓ Заключение           └──────────────────────────┘  │
│  00:03:45                                             │
└──────────────────┬────────────────────────────────────┘
                   │ WebSocket (PCM16 + JSON)
┌──────────────────▼────────────────────────────────────┐
│  Backend (FastAPI + GPU)                               │
│                                                        │
│  AudioRingBuffer (120с)                                │
│       ↓ каждые 8с                                      │
│  ┌─────────────────────────────────────┐               │
│  │ ASR (Whisper medium, GPU, float16)  │ new audio     │
│  │ Diarization (pyannote 3.1, GPU)     │ full buffer   │
│  └──────────┬──────────────────────────┘               │
│             ↓ asyncio.gather (параллельно)             │
│  ┌─────────────────────────────────────┐               │
│  │ Alignment + Speaker Mapping          │               │
│  │ (speaker_map хранится в сессии)      │               │
│  └──────────┬──────────────────────────┘               │
│             ↓                                          │
│  ┌─────────────────────────────────────┐               │
│  │ Medical Correction (Groq, ~0.5с)    │               │
│  └──────────┬──────────────────────────┘               │
│             ↓                                          │
│  ┌─────────────────────────────────────┐               │
│  │ Protocol Extraction (Groq, ~1.5с)   │               │
│  │ - извлечение данных ТОЛЬКО пациента  │               │
│  │ - мед. стиль записи                  │               │
│  │ - разделение: жалобы ≠ наследств.    │               │
│  └──────────┬──────────────────────────┘               │
│             ↓                                          │
│  transcript_update + protocol_update → Frontend        │
│                                                        │
│  [Финализация]: полный транскрипт → Groq →             │
│  → диагноз МКБ-10 + лечение + рекомендации            │
└────────────────────────────────────────────────────────┘
```

---

## Калибровка (первые ~15 секунд)

Фаза калибровки — первый processing cycle.

1. Врач нажимает «Начать приём»
2. На экране подсказка: "Спросите у пациента: ФИО и полный возраст"
3. Врач говорит: "Здравствуйте, как вас зовут? Сколько вам полных лет?"
4. Пациент отвечает: "Иванов Иван Иванович, 45 лет"
5. Первый cycle обрабатывает:
   - ASR: распознаёт обоих
   - Diarization: определяет 2 спикера
   - Первый спикер → ВРАЧ (speaker_map сохраняется в сессии навсегда)
   - LLM извлекает ФИО + возраст из ответа пациента
6. Калибровка завершена → статус "calibrated"
7. Далее — обычная запись приёма

---

## VRAM-бюджет (6GB GTX 1660)

| Модель | VRAM (float16) |
|--------|---------------|
| Whisper medium | 1.5 GB |
| Pyannote 3.1 | 0.7 GB |
| PyTorch overhead | ~0.3 GB |
| **Итого** | **~2.5 GB** |
| **Свободно** | **~3.5 GB** |

Опция large-v3: +1.5 GB = ~4.0 GB, свободно ~2.0 GB. Настраивается через .env.

---

## Скорость (GTX 1660, 8с аудио, medium)

| Этап | Время | Параллельно |
|------|-------|-------------|
| ASR (medium, float16) | ~1.0-1.5с | ← gather |
| Diarization (pyannote) | ~0.5-1.0с | ← gather |
| Alignment | ~0.01с | |
| Medical Correction (Groq) | ~0.5-1.0с | |
| Protocol Extraction (Groq) | ~1.5-2.0с | |
| **Итого** | **~3.5-5.5с** | < 8с ✅ |

---

## Протокол (расширенный)

```python
class PatientInfo(BaseModel):
    full_name: str | None = None     # ФИО (из калибровки)
    age: int | None = None           # Возраст (из калибровки)

class ExamData(BaseModel):
    complaints: str | None = None
    complaints_details: str | None = None
    anamnesis: str | None = None
    life_anamnesis: str | None = None
    allergies: str | None = None
    medications: str | None = None
    diagnosis: str | None = None
    treatment_plan: str | None = None
    patient_recommendations: str | None = None

class Vitals(BaseModel):
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    pulse: float | None = None
    spo2: float | None = None
    systolic_bp: str | None = None

class MedicalProtocol(BaseModel):
    patient_info: PatientInfo = PatientInfo()
    exam_data: ExamData = ExamData()
    vitals: Vitals = Vitals()
```

---

## WebSocket API v3

### Клиент → Сервер

```jsonc
{ "type": "start_session", "config": { "num_speakers": 2 } }
// Бинарные PCM16 фреймы (непрерывно во время записи)
{ "type": "stop_session" }
{ "type": "finalize" }
{ "type": "edit_field", "field": "complaints", "value": "Цефалгия..." }
```

### Сервер → Клиент

```jsonc
// Калибровка завершена
{
  "type": "calibration_complete",
  "patient_info": { "full_name": "Иванов Иван Иванович", "age": 45 }
}

// Транскрипция (каждые 8с)
{
  "type": "transcript_update",
  "utterances": [
    { "speaker": "doctor", "text": "...", "start": 0.0, "end": 2.5 },
    { "speaker": "patient", "text": "...", "start": 3.0, "end": 5.0 }
  ],
  "corrected_utterances": [...]  // после мед. коррекции
}

// Обновление полей протокола
{
  "type": "protocol_update",
  "protocol": { ... },
  "filled_fields": [...]
}

// Статус
{
  "type": "status",
  "status": "calibrating|recording|processing|stopped|finalizing|done"
}
```

---

## Файловая структура v3

```
backend/
├── app/
│   ├── config.py              # .env: whisper model/device, intervals
│   ├── main.py                # FastAPI + lifespan (ASR + Diar + LLM)
│   ├── models/
│   │   ├── protocol.py        # MedicalProtocol + PatientInfo
│   │   └── messages.py        # WS messages
│   ├── core/
│   │   ├── audio_buffer.py    # RingBuffer из v1
│   │   ├── session.py         # Session + speaker_map + calibration state
│   │   └── pipeline.py        # ASR || Diar → Align → Correct → Extract
│   ├── services/
│   │   ├── transcription.py   # Whisper (из v1)
│   │   ├── diarization.py     # Pyannote (из v1)
│   │   ├── alignment.py       # Speaker mapping (фикс из v1: стабильный map)
│   │   └── llm.py             # Correction + Extraction + Finalization
│   └── api/
│       ├── rest.py
│       └── websocket.py       # Streaming flow + calibration + edit
├── tests/
├── requirements.txt           # С pyannote
└── Dockerfile                 # GPU: CUDA torch + pyannote

frontend/
├── src/
│   ├── audio-capture.ts       # Без изменений
│   ├── websocket-client.ts    # Новые типы сообщений
│   ├── main.ts                # UI logic
│   ├── transcript-panel.ts    # Realtime транскрипция
│   ├── protocol-display.ts    # Editable fields
│   └── styles.css
├── index.html
└── ...
```

---

## .env переменные

```env
# LLM
LLM_PROVIDER=groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile

# Whisper ASR
WHISPER_MODEL=medium          # medium | large-v3
WHISPER_DEVICE=auto           # auto | cpu
WHISPER_COMPUTE_TYPE=float16  # float16 (GPU) | int8 (CPU)

# Pyannote
HF_TOKEN=...

# Processing
PROCESSING_INTERVAL_SECONDS=8
AUDIO_BUFFER_DURATION_SECONDS=120
NUM_SPEAKERS=2
```

---

## Что берём из предыдущих версий

| Компонент | Источник | Изменения |
|-----------|----------|-----------|
| AudioRingBuffer | v1 | Без изменений |
| ASR | v1 | medium по умолчанию, large-v3 опция |
| Diarization | v1 | Без изменений |
| Alignment | v1 | speaker_map хранится в сессии (фикс) |
| Medical Correction | v2 | Без изменений |
| LLM промпты | v2 | Адаптация: два спикера, мед. стиль |
| Parallel pipeline | v2 | asyncio.gather (ASR + Diar) |
| Editable fields | НОВОЕ | textarea после стоп |
| Калибровка | НОВОЕ | Первый cycle = определение врача + ФИО |
| edit_field WS message | НОВОЕ | Ручная правка → сервер |

---

## Порядок реализации

### Фаза 1 — Backend core
1. Session + калибровка + speaker_map persistence
2. Pipeline: ASR || Diar → Align → Correct → Extract
3. LLM промпты: extraction из разговора двоих
4. WebSocket: streaming + calibration + stop + finalize + edit
5. Тесты

### Фаза 2 — Frontend
1. Адаптация UI v2: две панели (транскрипция + протокол)
2. Realtime transcript (врач/пациент цветом)
3. Read-only поля во время записи → editable после стоп
4. Кнопка «Сформировать заключение»
5. Таймер записи

### Фаза 3 — GPU Dockerfile
1. CUDA torch + pyannote
2. docker-compose с GPU passthrough
3. .env переключаемый CPU/GPU

### Фаза 4 — Тестирование
1. Тестовый сценарий (врач + пациент)
2. Тюнинг промптов
3. Оптимизация скорости
