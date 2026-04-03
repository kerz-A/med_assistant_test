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
from .core.vad_segmenter import VADSegmenter
from .services.llm import LLMService
from .services.speaker_id import SpeakerIDService
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

vad_service = VADSegmenter()
transcription_service = TranscriptionService()
speaker_id_service = SpeakerIDService()
llm_service = LLMService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("AISTON TT v4 — VAD + Speaker Embeddings")
    logger.info("=" * 60)
    logger.info("[CONFIG] llm=%s/%s", settings.llm_provider, settings.llm_model)
    logger.info("[CONFIG] whisper=%s device=%s compute=%s beam=%d",
                settings.whisper_model, settings.whisper_device,
                settings.whisper_compute_type, settings.whisper_beam_size)
    logger.info("[CONFIG] vad_silence=%dms vad_min_speech=%dms concurrent=%d",
                settings.vad_silence_ms, settings.vad_min_speech_ms,
                settings.max_concurrent_segments)

    logger.info("[1/4] Loading Silero VAD...")
    vad_service.load_model()
    logger.info("[1/4] VAD ready")

    logger.info("[2/4] Loading Whisper ASR...")
    transcription_service.load_model()
    logger.info("[2/4] ASR ready")

    logger.info("[3/4] Loading ECAPA-TDNN speaker model...")
    speaker_id_service.load_model()
    logger.info("[3/4] Speaker ID ready")

    logger.info("[4/5] Initializing LLM client...")
    llm_service.initialize()
    if settings.llm_provider == "ollama":
        logger.info("[4/5] Pulling Ollama model %s...", settings.llm_model)
        await llm_service.pull_ollama_model()
    logger.info("[4/5] LLM ready")

    if llm_service._fallback_client:
        logger.info("[5/5] Pulling Ollama fallback model %s...", settings.ollama_model)
        await llm_service.pull_fallback_model()
        logger.info("[5/5] Fallback ready")

    logger.info("=" * 60)
    logger.info("Server ready — all models loaded")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down...")
    await llm_service.close()


app = FastAPI(title="MedScribe", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

pipeline = ProcessingPipeline(transcription_service, speaker_id_service, llm_service)
app.include_router(rest_router)
app.include_router(create_websocket_router(pipeline, vad_service))
