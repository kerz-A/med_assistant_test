"""Processing pipeline v3: ASR + Diarization (parallel) → Alignment → Medical Correction → Protocol Extraction."""

import asyncio
import logging
import time

from ..models.messages import Utterance
from ..models.protocol import MedicalProtocol
from ..services.alignment import align_segments
from ..services.diarization import DiarizationService
from ..services.llm import LLMService
from ..services.transcription import TranscriptionService
from .audio_buffer import SAMPLE_RATE
from .session import SessionState, SessionStage

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(
        self,
        transcription: TranscriptionService,
        diarization: DiarizationService,
        llm: LLMService,
    ):
        self.transcription = transcription
        self.diarization = diarization
        self.llm = llm

    async def process_cycle(self, session: SessionState) -> tuple[list[Utterance], int]:
        """Run one processing cycle. Returns (utterances, processing_time_ms)."""
        start_time = time.monotonic()

        new_audio = session.audio_buffer.get_unprocessed_audio()
        full_audio = session.audio_buffer.get_full_audio()

        if len(new_audio) < SAMPLE_RATE:
            logger.info("[PIPELINE] session=%s | Audio too short, skipping", session.session_id)
            return [], 0

        logger.info(
            "[PIPELINE] session=%s | stage=%s | new=%.1fs full=%.1fs",
            session.session_id, session.stage.value,
            len(new_audio) / SAMPLE_RATE, len(full_audio) / SAMPLE_RATE,
        )

        # 1. ASR + Diarization in PARALLEL
        t0 = time.monotonic()
        asr_task = self.transcription.transcribe(new_audio)
        diar_task = self.diarization.diarize(full_audio, SAMPLE_RATE, session.num_speakers)
        asr_segments, diar_segments = await asyncio.gather(asr_task, diar_task)
        parallel_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "[PIPELINE] ASR+Diar: %d asr, %d diar in %dms | speakers=%s",
            len(asr_segments), len(diar_segments), parallel_ms,
            {s.speaker for s in diar_segments},
        )

        if not asr_segments:
            return [], 0

        # 2. Alignment with persistent speaker_map
        time_offset = max(0, session.audio_buffer.total_duration - len(new_audio) / SAMPLE_RATE)
        utterances = align_segments(asr_segments, diar_segments, session.speaker_map, time_offset)

        if not utterances:
            return [], 0

        # 3. Medical term correction (via Groq)
        t0 = time.monotonic()
        for i, u in enumerate(utterances):
            corrected = await self.llm.correct_medical_terms(u.text)
            utterances[i] = Utterance(
                speaker=u.speaker, text=corrected,
                start=u.start, end=u.end,
            )
        correct_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PIPELINE] Medical correction: %dms", correct_ms)

        # 4. Protocol extraction (via Groq) — only during RECORDING, not calibration
        if session.stage == SessionStage.RECORDING:
            t0 = time.monotonic()
            patient_utterances = [u for u in utterances if u.speaker == "patient"]
            if patient_utterances:
                session.protocol = await self.llm.extract_protocol_data(
                    session.protocol, utterances, session.format_full_transcript(),
                )
            extract_ms = int((time.monotonic() - t0) * 1000)
            logger.info("[PIPELINE] Protocol extraction: %dms", extract_ms)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("[PIPELINE] Cycle complete: %dms total", processing_time_ms)

        return utterances, processing_time_ms

    async def process_calibration(self, session: SessionState) -> tuple[list[Utterance], int]:
        """Process calibration audio: identify speakers + extract patient info."""
        utterances, time_ms = await self.process_cycle(session)

        if not utterances:
            return utterances, time_ms

        # Extract patient info from calibration utterances
        patient_texts = [u.text for u in utterances if u.speaker == "patient"]
        if patient_texts:
            combined = " ".join(patient_texts)
            patient_info = await self.llm.extract_patient_info(combined)
            if patient_info.get("full_name"):
                session.protocol.patient_info.full_name = patient_info["full_name"]
            if patient_info.get("age"):
                session.protocol.patient_info.age = patient_info["age"]
            if patient_info.get("gender"):
                session.protocol.patient_info.gender = patient_info["gender"]
            logger.info(
                "[PIPELINE] Calibration: name=%s age=%s gender=%s | speakers=%s",
                session.protocol.patient_info.full_name,
                session.protocol.patient_info.age,
                session.protocol.patient_info.gender,
                session.speaker_map,
            )

        return utterances, time_ms

    async def finalize(self, session: SessionState) -> int:
        """Run full finalization: generate diagnosis + treatment + recommendations."""
        t0 = time.monotonic()
        logger.info("[PIPELINE] Finalizing session=%s | transcript=%d utterances",
                     session.session_id, len(session.transcript))

        session.protocol = await self.llm.finalize_protocol(
            session.protocol, session.format_full_transcript(),
        )

        # Quality metrics
        filled = session.get_filled_fields()
        cds = session.protocol.clinical_decision_support
        cds.quality_criteria.data_completeness = min(2, len(filled) // 3)
        cds.quality_criteria.complaints_quality = 2 if session.protocol.exam_data.complaints else 0
        cds.quality_criteria.anamnesis_quality = 2 if session.protocol.exam_data.anamnesis else 0
        cds.quality_criteria.vitals_collected = 2 if session.protocol.vitals.height_cm else 0
        cds.quality_criteria.life_history_quality = 2 if session.protocol.exam_data.life_anamnesis else 0
        cds.examination_quality.overall_score = sum([
            cds.quality_criteria.data_completeness,
            cds.quality_criteria.complaints_quality,
            cds.quality_criteria.anamnesis_quality,
            cds.quality_criteria.vitals_collected,
            cds.quality_criteria.life_history_quality,
        ]) / 2.0
        cds.examination_quality.recording_duration_sec = session.audio_buffer.total_duration

        time_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PIPELINE] Finalization complete: %dms | diagnosis=%s",
                     time_ms, bool(session.protocol.exam_data.diagnosis))
        return time_ms
