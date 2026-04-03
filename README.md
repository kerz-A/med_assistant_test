# MedScribe — Realtime Medical Protocol Assistant

Test implementation of a real-time system that fills in a medical examination protocol during a doctor-patient visit. The system listens, transcribes, identifies speakers (doctor/patient), extracts medical data, and populates the protocol on the fly.

> **Note:** This is a test/prototype implementation. In production with adequate GPU resources, all ML models (ASR, VAD, Speaker ID) should run on GPU for real-time field population effect. Current setup supports both CPU and GPU modes.

## 4 Stages of Examination

| Stage | Action | What happens |
|-------|--------|--------------|
| Calibration | Doctor asks patient's name/age/gender | Voice identification + patient data extraction |
| Recording | Doctor conducts examination | Real-time transcription + protocol field population |
| Editing | Doctor reviews/edits fields | Fields become editable |
| Finalization | Diagnosis generation | LLM generates diagnosis + treatment plan + recommendations |

## Quick Start

```bash
cp .env.example .env   # configure LLM provider and model settings
```

### GPU deployment (recommended for real-time experience)

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build --no-deps backend frontend
```

Requires NVIDIA driver >= 525.60 (CUDA 12.1). All ML models run on GPU — ASR processes segments in <1s, giving real-time field updates.

### CPU deployment (sufficient for testing)

```bash
docker compose up --build --no-deps backend frontend
```

ASR runs ~3-5x slower than real-time. Set `WHISPER_MODEL=small` in `.env` for faster processing at slight quality cost.

Frontend: http://localhost:3000 | Health check: http://localhost:8000/api/health

## LLM Provider

The system uses a cloud LLM for medical data extraction and diagnosis generation. Configure in `.env`:

| Provider | Config | Notes |
|----------|--------|-------|
| **GigaChat** (Sber) | `LLM_PROVIDER=gigachat` | Russian-optimized, recommended for Russian medical text |
| DeepSeek | `LLM_PROVIDER=deepseek` | Good quality, affordable |
| Groq | `LLM_PROVIDER=groq` | Fast inference, free tier available |
| OpenAI | `LLM_PROVIDER=openai` | GPT-4o-mini, highest quality |
| Ollama (local) | `LLM_PROVIDER=ollama` | Requires `ollama` service + GPU with >= 2 GB VRAM |

LLM runs via API — no local GPU memory needed (except Ollama). Add `--no-deps backend frontend` to skip the Ollama container when using cloud providers.

## Tech Stack

| Component | Technology | GPU | CPU |
|-----------|-----------|-----|-----|
| VAD | Silero VAD (~5 MB) | ~2ms/chunk | ~5ms/chunk |
| ASR | faster-whisper medium (~1.5 GB) | ~0.5-1s/segment | ~3-5s/segment |
| Speaker ID | ECAPA-TDNN SpeechBrain (~120 MB) | ~20ms | ~100-200ms |
| LLM | Cloud API (GigaChat/DeepSeek/Groq/OpenAI) | — | — |
| Backend | Python 3.11, FastAPI, WebSocket | | |
| Frontend | TypeScript, Vite, Nginx | | |

## Long Session Support

Sessions over ~50 utterances (~15+ minutes) use automatic patient speech summarization before finalization. This compresses the transcript from ~27 KB to ~5-7 KB while preserving all medically significant details, preventing LLM context overflow.

## Testing

### Unit tests

```bash
cd backend && python -m pytest tests/ -v
```

### End-to-end scenario tests

```bash
cd tools
pip install -r requirements.txt   # edge-tts, websockets, imageio-ffmpeg

# Generate test audio (requires internet for edge-tts)
python generate_test_scenarios.py --scenario 01_cardiology

# Run scenario
python run_all_scenarios.py --scenarios 01_cardiology

# Run all 11 scenarios including long session test
python run_all_scenarios.py
```

10 medical specialties + 1 long session scenario (70 utterances, ~14 min audio) for testing long session handling and GigaChat token refresh.

## Environment Variables

See `.env.example` for full list. Key settings:

```env
# LLM
LLM_PROVIDER=gigachat          # gigachat | deepseek | groq | openai | ollama
GIGACHAT_AUTH_KEY=...           # required for gigachat

# Whisper ASR
WHISPER_MODEL=medium            # tiny | base | small | medium | large-v3
WHISPER_DEVICE=auto             # auto | cpu | cuda
WHISPER_COMPUTE_TYPE=float16    # float16 (GPU) | int8 (CPU) | int8_float32
```
