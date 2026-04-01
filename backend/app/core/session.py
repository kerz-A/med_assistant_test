"""Session state for v3 — 4 stages: calibrate → record → edit → finalize."""

import asyncio
import logging
import uuid
from enum import Enum

from ..config import settings
from ..models.messages import Utterance
from ..models.protocol import MedicalProtocol
from ..core.audio_buffer import AudioRingBuffer

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
        self.speaker_map: dict[str, str] = {}  # persists across all cycles
        self.audio_buffer = AudioRingBuffer(settings.audio_buffer_duration_seconds)
        self.sequence = 0
        self.is_processing = False
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()
        # FIX #1: Track calibration zone for stable speaker mapping
        self.calibration_end_time: float = 0.0  # seconds of audio when calibration ended
        self.doctor_raw_label: str | None = None  # pyannote label of doctor from calibration
        logger.info("[SESSION] Created: id=%s speakers=%d", self.session_id, num_speakers)

    def add_audio(self, pcm_bytes: bytes) -> None:
        self.audio_buffer.append(pcm_bytes)

    def should_process(self) -> bool:
        return (
            self.stage == SessionStage.RECORDING  # Calibration is processed as one batch at stop
            and not self.is_processing
            and self.audio_buffer.unprocessed_duration >= settings.processing_interval_seconds
        )

    async def acquire_processing(self) -> bool:
        async with self._lock:
            if self.is_processing:
                return False
            self.is_processing = True
            return True

    def release_processing(self) -> None:
        self.is_processing = False

    async def wait_for_processing_complete(self, timeout: float = 30.0) -> bool:
        """Wait until any in-flight processing completes."""
        for _ in range(int(timeout * 10)):
            if not self.is_processing:
                return True
            await asyncio.sleep(0.1)
        logger.warning("[SESSION] Timed out waiting for processing to complete (%.1fs)", timeout)
        return False

    def track_task(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

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
