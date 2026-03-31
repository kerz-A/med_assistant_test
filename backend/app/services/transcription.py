import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
from faster_whisper import WhisperModel

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    language: str = "ru"


class TranscriptionService:
    def __init__(self):
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        device = settings.whisper_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = settings.whisper_compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
            logger.info("[ASR] CPU mode: switching compute_type float16 -> int8")

        logger.info(
            "[ASR] Loading Whisper model=%s device=%s compute_type=%s",
            settings.whisper_model, device, compute_type,
        )
        t0 = time.monotonic()
        self._model = WhisperModel(
            settings.whisper_model,
            device=device,
            compute_type=compute_type,
        )
        load_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[ASR] Whisper model loaded in %dms", load_ms)

    async def transcribe(self, audio: np.ndarray) -> list[TranscriptSegment]:
        if self._model is None:
            raise RuntimeError("Whisper model not loaded. Call load_model() first.")

        logger.info("[ASR] Transcribing: %d samples (%.1fs)", len(audio), len(audio) / 16000)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, partial(self._transcribe_sync, audio))

    def _transcribe_sync(self, audio: np.ndarray) -> list[TranscriptSegment]:
        t0 = time.monotonic()
        segments_gen, info = self._model.transcribe(
            audio,
            language="ru",
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
            },
            beam_size=5,
            best_of=1,
            without_timestamps=False,
        )

        result = []
        for seg in segments_gen:
            text = seg.text.strip()
            if text:
                result.append(TranscriptSegment(
                    text=text,
                    start=seg.start,
                    end=seg.end,
                    language=info.language,
                ))

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "[ASR] Transcription complete: %d segments in %dms | lang=%s lang_prob=%.2f | audio=%.1fs | RTF=%.2f",
            len(result), elapsed_ms,
            info.language, info.language_probability,
            len(audio) / 16000,
            (elapsed_ms / 1000) / (len(audio) / 16000) if len(audio) > 0 else 0,
        )
        for i, seg in enumerate(result):
            logger.debug("[ASR]   seg[%d] [%.1f-%.1f] %s", i, seg.start, seg.end, seg.text[:100])

        return result
