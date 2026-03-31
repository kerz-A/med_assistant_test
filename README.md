# Aiston TT v3 — Realtime заполнение протокола осмотра

Система реального времени для заполнения медицинского протокола во время приёма.
Врач ведёт приём — система слушает, транскрибирует, разделяет роли (врач/пациент),
извлекает данные и заполняет протокол на лету.

## 4 стадии приёма

| Стадия | Действие | Что происходит |
|--------|---------|----------------|
| 🎯 Калибровка | Врач спрашивает ФИО/возраст/пол | Определение голосов + данные пациента |
| 🔴 Запись | Врач ведёт приём | Realtime транскрипция + заполнение полей |
| ⏸ Редактирование | Врач правит поля | Поля становятся редактируемыми |
| ✅ Заключение | Генерация диагноза | LLM формирует диагноз + лечение |

## Быстрый старт

```bash
cp .env.example .env   # выбрать LLM_PROVIDER: ollama или groq

# Вариант A — полностью локально (Ollama + Qwen2.5-3B):
# В .env: LLM_PROVIDER=ollama (по умолчанию)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# Вариант B — облачный LLM (Groq):
# В .env: LLM_PROVIDER=groq, GROQ_API_KEY=...
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Frontend: http://localhost:3000 | Backend: http://localhost:8000/api/health

## Стек

| Компонент | Технология |
|-----------|-----------|
| ASR | faster-whisper medium/large-v3 (GPU) |
| Диаризация | pyannote.audio 3.1 (GPU) |
| LLM (локально) | Ollama + Qwen2.5-3B (GPU, ~2 GB VRAM) |
| LLM (облако) | Groq Llama 3.3 70B (переключаемый) |
| Backend | Python 3.11, FastAPI, WebSocket |
| Frontend | TypeScript, Vite |

## Тесты

```bash
cd backend && python -m pytest tests/ -v
```
