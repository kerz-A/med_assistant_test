import pytest
from app.core.session import SessionState, SessionStage
from app.models.protocol import MedicalProtocol, PatientInfo, ExamData, Vitals
from app.models.messages import ClientMessage, ClientMessageType, StatusMessage, CalibrationComplete, Utterance


class TestSessionStages:
    def test_initial_stage(self):
        s = SessionState()
        assert s.stage == SessionStage.IDLE

    def test_calibration_flow(self):
        s = SessionState()
        s.stage = SessionStage.CALIBRATING
        assert s.stage == SessionStage.CALIBRATING
        s.stage = SessionStage.CALIBRATED
        assert s.stage == SessionStage.CALIBRATED

    def test_recording_flow(self):
        s = SessionState()
        s.stage = SessionStage.CALIBRATED
        s.stage = SessionStage.RECORDING
        s.stage = SessionStage.STOPPED
        assert s.stage == SessionStage.STOPPED

    def test_speaker_map_persists(self):
        s = SessionState()
        s.speaker_map["SPEAKER_00"] = "doctor"
        s.speaker_map["SPEAKER_01"] = "patient"
        assert s.speaker_map["SPEAKER_00"] == "doctor"

    def test_filled_fields(self):
        s = SessionState()
        s.protocol.patient_info.full_name = "Иванов"
        s.protocol.patient_info.age = 45
        s.protocol.exam_data.complaints = "Головная боль"
        s.protocol.vitals.height_cm = 178
        filled = s.get_filled_fields()
        assert "full_name" in filled
        assert "age" in filled
        assert "complaints" in filled
        assert "height_cm" in filled
        assert "diagnosis" not in filled

    def test_transcript(self):
        s = SessionState()
        s.update_transcript([
            Utterance(speaker="doctor", text="Что беспокоит?", start=0, end=1),
            Utterance(speaker="patient", text="Голова", start=2, end=3),
        ])
        assert len(s.transcript) == 2
        assert s.sequence == 1
        text = s.format_full_transcript()
        assert "[Врач]" in text
        assert "[Пациент]" in text


class TestProtocol:
    def test_patient_info(self):
        p = MedicalProtocol(patient_info=PatientInfo(full_name="Иванов", age=45, gender="М"))
        assert p.patient_info.full_name == "Иванов"
        assert p.patient_info.gender == "М"

    def test_roundtrip(self):
        p = MedicalProtocol(
            patient_info=PatientInfo(full_name="Тест", age=30),
            exam_data=ExamData(complaints="Боль"),
            vitals=Vitals(height_cm=175),
        )
        restored = MedicalProtocol.model_validate_json(p.model_dump_json())
        assert restored.patient_info.full_name == "Тест"
        assert restored.vitals.height_cm == 175


class TestMessages:
    def test_calibration_complete(self):
        msg = CalibrationComplete(patient_info=PatientInfo(full_name="Иванов", age=45))
        data = msg.model_dump()
        assert data["type"] == "calibration_complete"
        assert data["patient_info"]["full_name"] == "Иванов"

    def test_client_message_types(self):
        for t in ClientMessageType:
            msg = ClientMessage(type=t)
            assert msg.type == t

    def test_status_message(self):
        msg = StatusMessage(status="calibrating", message="test")
        assert msg.status == "calibrating"
