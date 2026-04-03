"""Session state for v4 — VAD-driven: calibrate → record → edit → finalize."""

import asyncio
import logging
import time
import uuid
from enum import Enum

from ..config import settings
from ..models.messages import Utterance
from ..models.protocol import MedicalProtocol
from ..core.audio_buffer import AudioRingBuffer
from ..services.speaker_id import SpeakerProfile

logger = logging.getLogger(__name__)


class SessionStage(str, Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    RECORDING = "recording"
    STOPPED = "stopped"
    FINALIZING = "finalizing"
    DONE = "done"


class SessionState:
    def __init__(self, session_id: str | None = None, num_speakers: int = 2):
        self.session_id = session_id or str(uuid.uuid4())
        self.num_speakers = num_speakers
        self.stage = SessionStage.IDLE
        self.protocol = MedicalProtocol()
        self.transcript: list[Utterance] = []
        self.speaker_map: dict[str, str] = {}
        self.audio_buffer = AudioRingBuffer(settings.audio_buffer_duration_seconds)
        self.sequence = 0
        self._tasks: set[asyncio.Task] = set()
        self.calibration_end_time: float = 0.0

        # Speaker ID via embeddings
        self.doctor_profile = SpeakerProfile()
        self.patient_profile = SpeakerProfile()
        self._calibration_first_speaker_set = False

        # Calibration segment buffer (collected during streaming, processed at stop)
        self._calibration_segments: list = []  # list[SpeechSegment] - avoid circular import

        # LLM extraction batch buffer
        self._pending_utterances: list[Utterance] = []
        self._last_extraction_time: float = 0.0

        logger.info("[SESSION] Created: id=%s speakers=%d", self.session_id, num_speakers)

    def add_audio(self, pcm_bytes: bytes) -> None:
        self.audio_buffer.append(pcm_bytes)

    # ---- Calibration segment buffering ----

    def buffer_calibration_segment(self, segment) -> None:
        """Buffer a VAD segment for deferred processing at stop_calibration."""
        self._calibration_segments.append(segment)

    def get_calibration_segments(self) -> list:
        """Return and clear buffered calibration segments."""
        segs = self._calibration_segments
        self._calibration_segments = []
        return segs

    # ---- LLM extraction batching ----

    def add_pending_utterance(self, utterance: Utterance) -> None:
        self._pending_utterances.append(utterance)

    def should_extract_protocol(self) -> bool:
        """Check if we should run LLM extraction (batch threshold reached)."""
        if not self._pending_utterances:
            return False
        if len(self._pending_utterances) >= settings.extraction_batch_size:
            return True
        if time.monotonic() - self._last_extraction_time >= settings.extraction_interval_seconds:
            return True
        return False

    def peek_pending_utterances(self) -> list[Utterance]:
        """Return copy of pending utterances WITHOUT clearing (for safe extraction)."""
        return list(self._pending_utterances)

    def confirm_extraction(self, count: int) -> None:
        """Clear first N pending utterances after successful extraction."""
        self._pending_utterances = self._pending_utterances[count:]
        self._last_extraction_time = time.monotonic()

    def get_pending_utterances(self) -> list[Utterance]:
        """Return and clear pending utterances for extraction."""
        pending = self._pending_utterances
        self._pending_utterances = []
        self._last_extraction_time = time.monotonic()
        return pending

    def has_pending_utterances(self) -> bool:
        return len(self._pending_utterances) > 0

    # ---- Task tracking ----

    def track_task(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def wait_for_processing_complete(self, timeout: float = 30.0) -> bool:
        """Wait until all tracked async tasks complete."""
        if not self._tasks:
            return True
        tasks = list(self._tasks)  # snapshot to avoid mutation during wait
        logger.info("[SESSION] Waiting for %d tasks (timeout=%.1fs)", len(tasks), timeout)
        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout)
            if pending:
                logger.warning("[SESSION] %d tasks still pending after %.1fs", len(pending), timeout)
                return False
            return True
        except Exception as e:
            logger.error("[SESSION] wait error: %s", e)
            return False

    async def cancel_all_tasks(self) -> None:
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    def update_transcript(self, new_utterances: list[Utterance]) -> None:
        self.transcript.extend(new_utterances)
        self.sequence += 1
        self.audio_buffer.mark_processed()

    def get_filled_fields(self) -> list[str]:
        filled = []
        p = self.protocol
        if p.patient_info.full_name: filled.append("full_name")
        if p.patient_info.age is not None: filled.append("age")
        if p.patient_info.gender: filled.append("gender")
        for f in ["complaints", "complaints_details", "anamnesis",
                   "life_anamnesis", "allergies", "medications",
                   "diagnosis", "treatment_plan", "patient_recommendations"]:
            if getattr(p.exam_data, f, None):
                filled.append(f)
        for f in ["height_cm", "weight_kg", "bmi", "pulse", "spo2", "systolic_bp"]:
            if getattr(p.vitals, f, None) is not None:
                filled.append(f)
        return filled

    def format_full_transcript(self) -> str:
        lines = []
        for u in self.transcript:
            role = "Врач" if u.speaker == "doctor" else "Пациент"
            lines.append(f"[{role}]: {u.text}")
        return "\n".join(lines)

    def format_patient_utterances(self) -> str:
        """Extract only patient utterances for summarization."""
        return "\n".join(
            u.text for u in self.transcript if u.speaker == "patient"
        )


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def create_session(self, session_id: str | None = None, num_speakers: int = 2) -> SessionState:
        session = SessionState(session_id, num_speakers)
        self._sessions[session.session_id] = session
        logger.info("[SESSION_MGR] Added: id=%s (active=%d)", session.session_id, len(self._sessions))
        return session

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> None:
        removed = self._sessions.pop(session_id, None)
        if removed:
            logger.info("[SESSION_MGR] Removed: id=%s (remaining=%d)", session_id, len(self._sessions))


session_manager = SessionManager()
