# Aiston TT v4 — Realtime заполнение протокола осмотра

Система реального времени для заполнения медицинского протокола во время приёма.
Врач ведёт приём — система слушает, транскрибирует, разделяет роли (врач/пациент),
извлекает данные и заполняет протокол на лету.

## 4 стадии приёма

| Стадия | Действие | Что происходит |
|--------|---------|----------------|
| Калибровка | Врач спрашивает ФИО/возраст/пол | VAD-сегментация + ECAPA-TDNN эмбеддинги спикеров + данные пациента |
| Запись | Врач ведёт приём | Realtime транскрипция + батчевое извлечение данных в протокол |
| Остановка | Врач завершает приём | Flush pending utterances + финальная extraction |
| Заключение | Генерация диагноза | LLM формирует диагноз по МКБ-10, план лечения, рекомендации |

## Быстрый старт

```bash
cp .env.example .env
# Настроить LLM_PROVIDER и API ключ (см. ниже)

# Запуск с GPU:
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Запуск без GPU (CPU only):
docker compose up --build
```

Frontend: http://localhost:3000 | Backend: http://localhost:8000/api/health

## LLM-провайдеры

| Провайдер | Модель | Оплата из РФ | Rate Limit | Рекомендация |
|-----------|--------|-------------|------------|--------------|
| **GigaChat (Сбер)** | GigaChat-2-Max/Pro | Да (Mir, SberPay) | По concurrency | Рекомендован для РФ |
| **DeepSeek** | deepseek-chat (V3) | Через посредников | Нет лимитов | Лучшее соотношение цена/качество |
| **Groq** | Llama 3.3 70B | Нет | 6K TPM (жёсткий) | Быстрый, но частые 429 |
| **OpenRouter** | Llama 3.3 70B :free | Нет | 50 RPD | Бесплатный, но лимитированный |
| **OpenAI** | gpt-4o-mini | Нет | 500+ RPM | Лучшее качество |
| **Ollama** (fallback) | qwen2.5:3b | Локально | Нет | Автоматический fallback при отказе облака |

Настройка в `.env`:

```bash
# GigaChat (рекомендовано)
LLM_PROVIDER=gigachat
GIGACHAT_AUTH_KEY=ваш_authorization_key
GIGACHAT_MODEL=GigaChat-2-Max

# DeepSeek
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...

# Groq
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

## Архитектура

| Компонент | Технология |
|-----------|-----------|
| VAD | Silero VAD (GPU) — сегментация речи |
| ASR | faster-whisper medium (GPU) — транскрипция |
| Speaker ID | ECAPA-TDNN (GPU) — идентификация врач/пациент |
| LLM | Мульти-провайдер + Ollama fallback |
| Backend | Python 3.11, FastAPI, WebSocket |
| Frontend | TypeScript, Vite |

### Отказоустойчивость LLM

Система гарантирует 100% обработку данных:
- При ошибке LLM — utterances остаются в pending очереди и повторно обрабатываются
- При rate limit (429) — cooldown внутри asyncio.Lock, без гонки между concurrent вызовами
- При отказе облачного провайдера — автоматический fallback на локальный Ollama (qwen2.5:3b)
- Финализация использует полный transcript — даже если extraction пропустила utterances

## Тестирование

```bash
# Unit-тесты
cd backend && python -m pytest tests/ -v

# Автоматические сценарии (10 медицинских специализаций)
cd tools
python run_all_scenarios.py

# Выборочные сценарии
python run_all_scenarios.py --scenarios 01_cardiology 10_emergency

# Оценка результатов
python evaluate_test.py
```

### Метрики качества (GigaChat-2-Pro, 8 завершённых сценариев)

| Метрика | Значение | Описание |
|---------|----------|----------|
| TC | 100% | Полнота транскрипции |
| SA | 81.2% | Точность определения спикера |
| FER | 100% | Доля заполненных полей протокола |
| FVA | 95.9% | Точность значений полей |
| DA | 62.5% | Правильность диагноза по МКБ-10 |
| **OQS** | **91.5%** | **Общая оценка качества** |

## Docker-сервисы

| Сервис | Описание |
|--------|----------|
| `backend` | FastAPI + ML модели (Whisper, VAD, ECAPA-TDNN) |
| `frontend` | React/Vite UI |
| `ollama` | Локальный LLM для fallback (qwen2.5:3b) |
