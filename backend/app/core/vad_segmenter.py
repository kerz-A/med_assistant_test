"""VAD-based audio segmenter using Silero VAD.

Segments incoming audio by speech/silence boundaries instead of fixed-time chunks.
When silence >= threshold is detected after speech, emits a complete speech segment.
"""

import asyncio
import logging
from collections.abc import Callable, Awaitable
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# Silero VAD requires chunks of exactly 512 samples at 16kHz (32ms)
VAD_CHUNK_SAMPLES = 512


@dataclass
class SpeechSegment:
    """A complete speech segment detected by VAD."""
    audio: np.ndarray  # float32 normalized audio
    start_time: float  # absolute start time in session (seconds)
    end_time: float    # absolute end time in session (seconds)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class VADSegmenter:
    """Segments incoming audio stream by speech/silence boundaries.

    Usage:
        segmenter = VADSegmenter(on_segment=my_callback)
        segmenter.load_model()

        # Feed audio frames as they arrive from WebSocket
        for pcm_bytes in websocket_stream:
            await segmenter.feed(pcm_bytes)
    """

    def __init__(
        self,
        on_segment: Callable[[SpeechSegment], Awaitable[None]] | None = None,
    ):
        self._model: torch.jit.ScriptModule | None = None
        self._on_segment = on_segment

        # VAD state
        self._speech_buffer: list[np.ndarray] = []
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_start_sample = 0

        # Frame buffer for accumulating PCM until we have VAD_CHUNK_SAMPLES
        self._frame_buffer = np.zeros(0, dtype=np.float32)

        # Timing
        self._total_samples = 0
        self._device = "cpu"

        # Config
        self._threshold = settings.vad_threshold
        self._min_silence_samples = int(settings.vad_silence_ms / 1000 * SAMPLE_RATE)
        self._min_speech_samples = int(settings.vad_min_speech_ms / 1000 * SAMPLE_RATE)
        self._max_speech_samples = int(settings.vad_max_speech_ms / 1000 * SAMPLE_RATE)
        self._speech_pad_samples = int(settings.vad_speech_pad_ms / 1000 * SAMPLE_RATE)

    def load_model(self, device: str = "auto") -> None:
        """Load Silero VAD model. Supports CPU and GPU."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("[VAD] Loading Silero VAD model on %s...", device)
        self._model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        if device == "cuda":
            self._model = self._model.to(torch.device("cuda"))
        self._device = device
        self._model.eval()
        logger.info("[VAD] Silero VAD loaded (%s)", device)

    def reset(self) -> None:
        """Reset VAD state for new session."""
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_samples = 0
        self._speech_start_sample = 0
        self._frame_buffer = np.zeros(0, dtype=np.float32)
        self._total_samples = 0
        if self._model is not None:
            self._model.reset_states()

    async def feed(self, pcm_bytes: bytes) -> list[SpeechSegment]:
        """Feed raw PCM int16 bytes. Returns list of completed segments (usually 0 or 1)."""
        if self._model is None:
            raise RuntimeError("VAD model not loaded")

        # Convert PCM int16 → float32 normalized
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Append to frame buffer
        self._frame_buffer = np.concatenate([self._frame_buffer, samples])

        segments: list[SpeechSegment] = []

        # Process complete VAD chunks (512 samples each)
        while len(self._frame_buffer) >= VAD_CHUNK_SAMPLES:
            chunk = self._frame_buffer[:VAD_CHUNK_SAMPLES]
            self._frame_buffer = self._frame_buffer[VAD_CHUNK_SAMPLES:]

            segment = await self._process_vad_chunk(chunk)
            if segment is not None:
                segments.append(segment)

        return segments

    async def _process_vad_chunk(self, chunk: np.ndarray) -> SpeechSegment | None:
        """Process one VAD chunk (512 samples). Returns a segment if speech ended."""
        # Get speech probability from Silero VAD
        tensor = torch.from_numpy(chunk)
        if self._device == "cuda":
            tensor = tensor.to(torch.device("cuda"))
        speech_prob = self._model(tensor, SAMPLE_RATE).item()

        self._total_samples += VAD_CHUNK_SAMPLES
        segment = None

        if speech_prob >= self._threshold:
            # Speech detected
            if not self._is_speaking:
                # Speech START
                self._is_speaking = True
                self._speech_start_sample = self._total_samples - VAD_CHUNK_SAMPLES
                self._silence_samples = 0
                self._speech_buffer = []
                logger.debug("[VAD] Speech start at %.2fs", self._speech_start_sample / SAMPLE_RATE)

            self._speech_buffer.append(chunk)
            self._silence_samples = 0

        else:
            # Silence detected
            if self._is_speaking:
                self._speech_buffer.append(chunk)  # Include trailing silence for padding
                self._silence_samples += VAD_CHUNK_SAMPLES

                # Check if enough silence to end segment
                if self._silence_samples >= self._min_silence_samples:
                    segment = self._emit_segment()

        # Force-emit if speech too long (protection against non-stop speech)
        if self._is_speaking and len(self._speech_buffer) * VAD_CHUNK_SAMPLES >= self._max_speech_samples:
            logger.warning("[VAD] Max speech duration reached (%.1fs), force-emitting",
                           len(self._speech_buffer) * VAD_CHUNK_SAMPLES / SAMPLE_RATE)
            segment = self._emit_segment()

        # Call callback if segment ready
        if segment is not None and self._on_segment is not None:
            await self._on_segment(segment)

        return segment

    def _emit_segment(self) -> SpeechSegment | None:
        """Finalize and emit current speech segment."""
        if not self._speech_buffer:
            self._is_speaking = False
            return None

        audio = np.concatenate(self._speech_buffer)
        duration_samples = len(audio)

        # Skip too-short segments (noise, clicks)
        if duration_samples < self._min_speech_samples:
            logger.debug("[VAD] Segment too short (%.2fs), skipping",
                         duration_samples / SAMPLE_RATE)
            self._is_speaking = False
            self._speech_buffer = []
            self._silence_samples = 0
            return None

        start_time = self._speech_start_sample / SAMPLE_RATE
        end_time = start_time + duration_samples / SAMPLE_RATE

        segment = SpeechSegment(
            audio=audio,
            start_time=start_time,
            end_time=end_time,
        )

        logger.info("[VAD] Segment: %.2f-%.2fs (%.1fs)",
                     start_time, end_time, segment.duration)

        # Reset state
        self._is_speaking = False
        self._speech_buffer = []
        self._silence_samples = 0

        return segment

    async def flush(self) -> SpeechSegment | None:
        """Force-emit any buffered speech (call at end of recording)."""
        if self._is_speaking and self._speech_buffer:
            segment = self._emit_segment()
            if segment is not None and self._on_segment is not None:
                await self._on_segment(segment)
            return segment
        return None
