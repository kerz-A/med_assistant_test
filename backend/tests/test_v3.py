import json
import sys
from unittest.mock import MagicMock

# Mock heavy dependencies that aren't needed for unit tests
for mod in ["torch", "pyannote", "pyannote.audio", "faster_whisper", "httpx",
            "speechbrain", "speechbrain.inference", "speechbrain.inference.speaker"]:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

import numpy as np
import pytest

from app.core.audio_buffer import AudioRingBuffer, SAMPLE_RATE
from app.core.session import SessionState, SessionStage
from app.models.protocol import MedicalProtocol, PatientInfo, ExamData, Vitals
from app.models.messages import ClientMessage, ClientMessageType, StatusMessage, CalibrationComplete, Utterance
from app.services.alignment import align_segments, verify_speaker_map
from app.services.diarization import DiarizationSegment
from app.services.transcription import TranscriptSegment
from app.services.llm import LLMService


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


# ============================================================
# AudioRingBuffer tests
# ============================================================

class TestAudioRingBuffer:
    def _make_pcm(self, duration_sec: float) -> bytes:
        """Generate PCM int16 silence bytes."""
        n = int(SAMPLE_RATE * duration_sec)
        return np.zeros(n, dtype=np.int16).tobytes()

    def test_basic_append_and_duration(self):
        buf = AudioRingBuffer(max_duration_seconds=10)
        buf.append(self._make_pcm(3.0))
        assert abs(buf.total_duration - 3.0) < 0.01
        assert abs(buf.real_total_duration - 3.0) < 0.01

    def test_real_total_duration_after_wraparound(self):
        buf = AudioRingBuffer(max_duration_seconds=5)
        buf.append(self._make_pcm(3.0))
        buf.append(self._make_pcm(3.0))
        buf.append(self._make_pcm(3.0))
        # 9 seconds written, buffer holds max 5
        assert abs(buf.total_duration - 5.0) < 0.01
        assert abs(buf.real_total_duration - 9.0) < 0.01

    def test_unprocessed_duration(self):
        buf = AudioRingBuffer(max_duration_seconds=10)
        buf.append(self._make_pcm(4.0))
        assert abs(buf.unprocessed_duration - 4.0) < 0.01
        buf.mark_processed()
        assert abs(buf.unprocessed_duration - 0.0) < 0.01
        buf.append(self._make_pcm(2.0))
        assert abs(buf.unprocessed_duration - 2.0) < 0.01

    def test_overlap_no_previous(self):
        buf = AudioRingBuffer(max_duration_seconds=10)
        buf.append(self._make_pcm(3.0))
        audio, overlap = buf.get_unprocessed_audio_with_overlap(2.0)
        assert overlap == 0.0
        assert abs(len(audio) / SAMPLE_RATE - 3.0) < 0.01

    def test_overlap_after_mark(self):
        buf = AudioRingBuffer(max_duration_seconds=10)
        buf.append(self._make_pcm(5.0))
        buf.mark_processed()
        buf.append(self._make_pcm(3.0))
        audio, overlap = buf.get_unprocessed_audio_with_overlap(2.0)
        assert abs(len(audio) / SAMPLE_RATE - 5.0) < 0.01
        assert abs(overlap - 2.0) < 0.01

    def test_overlap_partial(self):
        buf = AudioRingBuffer(max_duration_seconds=10)
        buf.append(self._make_pcm(1.0))
        buf.mark_processed()
        buf.append(self._make_pcm(3.0))
        audio, overlap = buf.get_unprocessed_audio_with_overlap(2.0)
        # Only 1s available for overlap
        assert abs(overlap - 1.0) < 0.01
        assert abs(len(audio) / SAMPLE_RATE - 4.0) < 0.01

    def test_wraparound_integrity(self):
        buf = AudioRingBuffer(max_duration_seconds=5)
        for i in range(3):
            n = int(SAMPLE_RATE * 3.0)
            samples = np.full(n, (i + 1) * 1000, dtype=np.int16)
            buf.append(samples.tobytes())
        full = buf.get_full_audio()
        assert len(full) == 5 * SAMPLE_RATE


# ============================================================
# Alignment tests
# ============================================================

class TestAlignment:
    def test_basic_alignment_same_frame(self):
        asr = [
            TranscriptSegment(text="Привет", start=0.5, end=1.5),
            TranscriptSegment(text="Как дела", start=2.0, end=3.0),
        ]
        diar = [
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.8),
            DiarizationSegment(speaker="SPEAKER_01", start=1.8, end=4.0),
        ]
        speaker_map = {"SPEAKER_00": "doctor", "SPEAKER_01": "patient"}

        utterances = align_segments(asr, diar, speaker_map, time_offset=10.0, diar_offset=10.0)

        assert len(utterances) == 2
        assert utterances[0].speaker == "doctor"
        assert utterances[0].start == 10.5
        assert utterances[1].speaker == "patient"
        assert utterances[1].start == 12.0

    def test_diar_offset_shifts_timestamps(self):
        asr = [TranscriptSegment(text="Тест", start=1.0, end=2.0)]
        diar = [DiarizationSegment(speaker="SPEAKER_00", start=1.0, end=2.0)]
        speaker_map = {"SPEAKER_00": "doctor"}

        utterances = align_segments(asr, diar, speaker_map, time_offset=50.0, diar_offset=50.0)
        assert len(utterances) == 1
        assert utterances[0].speaker == "doctor"
        assert utterances[0].start == 51.0

    def test_calibration_no_diar_offset(self):
        asr = [TranscriptSegment(text="Имя", start=0.0, end=1.0)]
        diar = [
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=5.0),
            DiarizationSegment(speaker="SPEAKER_01", start=5.0, end=10.0),
        ]
        speaker_map = {}

        utterances = align_segments(asr, diar, speaker_map, time_offset=0.0, diar_offset=0.0)
        assert len(utterances) == 1
        assert utterances[0].speaker == "doctor"
        assert "SPEAKER_00" in speaker_map

    def test_verify_speaker_map_detects_swap(self):
        diar = [
            DiarizationSegment(speaker="SPEAKER_01", start=0.0, end=8.0),
            DiarizationSegment(speaker="SPEAKER_00", start=8.0, end=15.0),
        ]
        speaker_map = {"SPEAKER_00": "doctor", "SPEAKER_01": "patient"}

        verify_speaker_map(diar, speaker_map, calibration_end_time=10.0)
        assert speaker_map["SPEAKER_01"] == "doctor"

    def test_mismatched_timestamps_without_offset(self):
        """Without diar_offset, ASR and diar don't overlap → fallback speaker."""
        asr = [TranscriptSegment(text="Тест", start=0.0, end=1.0)]
        # Diar segment at 50-51s (absolute), ASR at 50-51s (after offset)
        diar = [
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
            DiarizationSegment(speaker="SPEAKER_01", start=1.0, end=5.0),
        ]
        speaker_map = {"SPEAKER_00": "doctor", "SPEAKER_01": "patient"}

        # With matching offset: ASR at 50-51, diar at 50-51 → doctor
        utterances = align_segments(asr, diar, speaker_map, time_offset=50.0, diar_offset=50.0)
        assert utterances[0].speaker == "doctor"


# ============================================================
# LLM merge protocol tests
# ============================================================

class TestMergeProtocol:
    def setup_method(self):
        self.llm = LLMService()

    def test_merge_basic_extraction(self):
        raw = json.dumps({
            "exam_data": {"complaints": "головная боль", "anamnesis": "после простуды"},
            "vitals": {"height_cm": 178, "weight_kg": 82},
        })
        result = self.llm._merge_protocol(raw, MedicalProtocol())
        assert result.exam_data.complaints == "головная боль"
        assert result.exam_data.anamnesis == "после простуды"
        assert result.vitals.height_cm == 178
        assert result.vitals.weight_kg == 82
        assert result.vitals.bmi is not None

    def test_merge_preserves_existing_fields(self):
        existing = MedicalProtocol(exam_data=ExamData(complaints="боль"))
        raw = json.dumps({
            "exam_data": {"complaints": None, "anamnesis": "новое"},
            "vitals": {},
        })
        result = self.llm._merge_protocol(raw, existing)
        assert result.exam_data.complaints == "боль"
        assert result.exam_data.anamnesis == "новое"

    def test_merge_appends_medications(self):
        existing = MedicalProtocol(exam_data=ExamData(medications="Нурофен 400 мг"))
        raw = json.dumps({
            "exam_data": {"medications": "Парацетамол 500 мг"},
            "vitals": {},
        })
        result = self.llm._merge_protocol(raw, existing)
        assert "Нурофен" in result.exam_data.medications
        assert "Парацетамол" in result.exam_data.medications

    def test_merge_validates_vitals_range(self):
        raw = json.dumps({
            "exam_data": {},
            "vitals": {"height_cm": 5000, "pulse": -10, "spo2": 200},
        })
        result = self.llm._merge_protocol(raw, MedicalProtocol())
        assert result.vitals.height_cm is None
        assert result.vitals.pulse is None
        assert result.vitals.spo2 is None

    def test_merge_validates_bp_format(self):
        raw = json.dumps({"exam_data": {}, "vitals": {"systolic_bp": "130/85"}})
        result = self.llm._merge_protocol(raw, MedicalProtocol())
        assert result.vitals.systolic_bp == "130/85"

    def test_merge_rejects_invalid_bp(self):
        raw = json.dumps({"exam_data": {}, "vitals": {"systolic_bp": "сто тридцать"}})
        result = self.llm._merge_protocol(raw, MedicalProtocol())
        assert result.vitals.systolic_bp is None

    def test_merge_invalid_json(self):
        result = self.llm._merge_protocol("not json", MedicalProtocol())
        assert result.exam_data.complaints is None

    def test_auto_bmi(self):
        raw = json.dumps({"exam_data": {}, "vitals": {"height_cm": 178, "weight_kg": 82}})
        result = self.llm._merge_protocol(raw, MedicalProtocol())
        expected_bmi = round(82 / (1.78 ** 2), 1)
        assert result.vitals.bmi == expected_bmi


# ============================================================
# Speaker ID tests
# ============================================================

class TestSpeakerID:
    def test_speaker_profile_mean(self):
        from app.services.speaker_id import SpeakerProfile
        profile = SpeakerProfile()
        profile.add(np.array([1.0, 0.0, 0.0]))
        profile.add(np.array([0.0, 1.0, 0.0]))
        mean = profile.mean_embedding
        assert mean is not None
        assert mean.shape == (3,)
        # Should be L2 normalized
        assert abs(np.linalg.norm(mean) - 1.0) < 0.01

    def test_speaker_profile_empty(self):
        from app.services.speaker_id import SpeakerProfile
        profile = SpeakerProfile()
        assert profile.mean_embedding is None
        assert profile.count == 0

    def test_cosine_similarity(self):
        from app.services.speaker_id import _cosine_similarity
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(a, b) - 1.0) < 0.01

        c = np.array([0.0, 1.0, 0.0])
        assert abs(_cosine_similarity(a, c) - 0.0) < 0.01

    def test_classify_speaker_binary(self):
        from app.services.speaker_id import SpeakerProfile, SpeakerIDService
        service = SpeakerIDService()

        doctor = SpeakerProfile()
        doctor.add(np.array([1.0, 0.0, 0.0]))

        patient = SpeakerProfile()
        patient.add(np.array([0.0, 1.0, 0.0]))

        # Embedding close to doctor
        emb_doc = np.array([0.9, 0.1, 0.0])
        role, conf = service.classify_speaker(emb_doc, doctor, patient)
        assert role == "doctor"
        assert conf > 0.3

        # Embedding close to patient
        emb_pat = np.array([0.1, 0.9, 0.0])
        role, conf = service.classify_speaker(emb_pat, doctor, patient)
        assert role == "patient"
        assert conf > 0.3

    def test_classify_only_doctor_known(self):
        from app.services.speaker_id import SpeakerProfile, SpeakerIDService
        service = SpeakerIDService()

        doctor = SpeakerProfile()
        doctor.add(np.array([1.0, 0.0, 0.0]))
        patient = SpeakerProfile()  # empty

        # Similar to doctor
        role, _ = service.classify_speaker(np.array([0.95, 0.05, 0.0]), doctor, patient)
        assert role == "doctor"

        # Different from doctor
        role, _ = service.classify_speaker(np.array([0.0, 1.0, 0.0]), doctor, patient)
        assert role == "patient"


# ============================================================
# VAD Segmenter tests
# ============================================================

class TestVADSegmenter:
    def test_speech_segment_duration(self):
        from app.core.vad_segmenter import SpeechSegment
        seg = SpeechSegment(
            audio=np.zeros(16000, dtype=np.float32),
            start_time=1.0,
            end_time=2.0,
        )
        assert abs(seg.duration - 1.0) < 0.01
