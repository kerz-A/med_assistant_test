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
    logger.info("[CONFIG] vad_silence=%dms vad_min_speech=%dms",
                settings.vad_silence_ms, settings.vad_min_speech_ms)

    logger.info("[1/3] Loading Whisper ASR...")
    transcription_service.load_model()
    logger.info("[1/3] ASR ready")

    logger.info("[2/3] Loading ECAPA-TDNN speaker model...")
    speaker_id_service.load_model()
    logger.info("[2/3] Speaker ID ready")

    logger.info("[3/3] Initializing LLM client...")
    llm_service.initialize()
    if settings.llm_provider == "ollama":
        logger.info("[3/3] Pulling Ollama model %s...", settings.llm_model)
        await llm_service.pull_ollama_model()
    logger.info("[3/3] LLM ready")

    logger.info("=" * 60)
    logger.info("Server ready — VAD + Speaker Embeddings pipeline")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down...")
    await llm_service.close()


app = FastAPI(title="Aiston TT v4", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

pipeline = ProcessingPipeline(transcription_service, speaker_id_service, llm_service)
app.include_router(rest_router)
app.include_router(create_websocket_router(pipeline))
