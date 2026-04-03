from enum import Enum

from pydantic import BaseModel

from .protocol import MedicalProtocol, PatientInfo


class Utterance(BaseModel):
    speaker: str  # "doctor" | "patient"
    text: str
    start: float
    end: float

    def format_role(self) -> str:
        """Format as '[Врач]: text' or '[Пациент]: text' for LLM context."""
        role = "Врач" if self.speaker == "doctor" else "Пациент"
        return f"[{role}]: {self.text}"


# --- Client → Server ---

class ClientMessageType(str, Enum):
    START_CALIBRATION = "start_calibration"
    STOP_CALIBRATION = "stop_calibration"
    START_RECORDING = "start_recording"
    STOP_RECORDING = "stop_recording"
    FINALIZE = "finalize"
    EDIT_FIELD = "edit_field"


class SessionConfig(BaseModel):
    num_speakers: int = 2


class ClientMessage(BaseModel):
    type: ClientMessageType
    config: SessionConfig = SessionConfig()
    # For edit_field
    field: str | None = None
    value: str | None = None


# --- Server → Client ---

class CalibrationComplete(BaseModel):
    type: str = "calibration_complete"
    patient_info: PatientInfo
    message: str = "Калибровка завершена"


class TranscriptUpdate(BaseModel):
    type: str = "transcript_update"
    utterances: list[Utterance] = []
    processing_time_ms: int = 0


class ProtocolUpdateMessage(BaseModel):
    type: str = "protocol_update"
    protocol: MedicalProtocol
    filled_fields: list[str] = []


class StatusMessage(BaseModel):
    type: str = "status"
    status: str  # calibrating | calibrated | recording | processing | stopped | finalizing | done | error
    message: str = ""
