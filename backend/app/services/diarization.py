import asyncio
import logging
import time
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
from pyannote.audio import Pipeline

from ..config import settings

logger = logging.getLogger(__name__)


@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float


class DiarizationService:
    def __init__(self):
        self._pipeline: Pipeline | None = None

    def load_model(self) -> None:
        if not settings.hf_token:
            logger.warning("[DIAR] HF_TOKEN not set — diarization will be unavailable")
            return

        device = settings.whisper_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("[DIAR] Loading pyannote diarization pipeline on %s", device)
        t0 = time.monotonic()
        self._pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_token,
        )
        if device == "cuda":
            self._pipeline.to(torch.device("cuda"))
        load_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[DIAR] Pyannote pipeline loaded in %dms (device=%s)", load_ms, device)

    async def diarize(
        self, audio: np.ndarray, sample_rate: int, num_speakers: int = 2
    ) -> list[DiarizationSegment]:
        if self._pipeline is None:
            logger.warning("[DIAR] Pipeline unavailable — returning single speaker fallback")
            duration = len(audio) / sample_rate
            return [DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=duration)]

        logger.info(
            "[DIAR] Diarizing: %d samples (%.1fs) with num_speakers=%d",
            len(audio), len(audio) / sample_rate, num_speakers,
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self._diarize_sync, audio, sample_rate, num_speakers)
        )

    def _diarize_sync(
        self, audio: np.ndarray, sample_rate: int, num_speakers: int
    ) -> list[DiarizationSegment]:
        t0 = time.monotonic()
        waveform = torch.from_numpy(audio).unsqueeze(0)
        audio_input = {"waveform": waveform, "sample_rate": sample_rate}

        diarization = self._pipeline(audio_input, num_speakers=num_speakers)

        result = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            result.append(DiarizationSegment(
                speaker=speaker,
                start=turn.start,
                end=turn.end,
            ))

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        speakers = {s.speaker for s in result}
        logger.info(
            "[DIAR] Diarization complete: %d segments in %dms | speakers=%s | audio=%.1fs | RTF=%.2f",
            len(result), elapsed_ms, speakers,
            len(audio) / sample_rate,
            (elapsed_ms / 1000) / (len(audio) / sample_rate) if len(audio) > 0 else 0,
        )
        for i, seg in enumerate(result):
            logger.debug("[DIAR]   seg[%d] %s [%.2f-%.2f] (%.1fs)", i, seg.speaker, seg.start, seg.end, seg.end - seg.start)

        return result
