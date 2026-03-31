import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.rest import router as rest_router
from .api.websocket import create_websocket_router
from .config import settings
from .core.pipeline import ProcessingPipeline
from .services.diarization import DiarizationService
from .services.llm import LLMService
from .services.transcription import TranscriptionService

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
for lib in ("httpx", "httpcore", "faster_whisper", "pyannote", "speechbrain", "uvicorn.access"):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

transcription_service = TranscriptionService()
diarization_service = DiarizationService()
llm_service = LLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AISTON TT v3 — Realtime Medical Protocol")
    logger.info("=" * 60)
    logger.info("[CONFIG] llm=%s/%s", settings.llm_provider, settings.llm_model)
    logger.info("[CONFIG] whisper=%s device=%s compute=%s", settings.whisper_model, settings.whisper_device, settings.whisper_compute_type)
    logger.info("[CONFIG] interval=%ds buffer=%ds speakers=%d", settings.processing_interval_seconds, settings.audio_buffer_duration_seconds, settings.num_speakers)
    logger.info("[CONFIG] hf_token=%s llm_key=%s", "SET" if settings.hf_token else "NO", "SET" if settings.llm_api_key else "NO")

    logger.info("[1/3] Loading Whisper ASR...")
    transcription_service.load_model()
    logger.info("[1/3] ASR ready")

    logger.info("[2/3] Loading pyannote diarization...")
    diarization_service.load_model()
    logger.info("[2/3] Diarization ready")

    logger.info("[3/3] Initializing LLM client...")
    llm_service.initialize()
    if settings.llm_provider == "ollama":
        logger.info("[3/3] Pulling Ollama model %s (first run may take a few minutes)...", settings.llm_model)
        await llm_service.pull_ollama_model()
    logger.info("[3/3] LLM ready")

    logger.info("=" * 60)
    logger.info("Server ready")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down...")
    await llm_service.close()


app = FastAPI(title="Aiston TT v3", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

pipeline = ProcessingPipeline(transcription_service, diarization_service, llm_service)
app.include_router(rest_router)
app.include_router(create_websocket_router(pipeline))
