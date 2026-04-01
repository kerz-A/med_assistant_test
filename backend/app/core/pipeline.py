"""Processing pipeline v3: ASR + Diarization (parallel) → Alignment → Medical Correction → Protocol Extraction."""

import asyncio
import json
import logging
import time

from ..config import settings
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
        """Run one recording processing cycle. Returns (utterances, processing_time_ms)."""
        start_time = time.monotonic()

        # Get audio with overlap for better word boundary handling
        audio_chunk, overlap_duration = session.audio_buffer.get_unprocessed_audio_with_overlap(
            settings.audio_overlap_seconds,
        )

        if len(audio_chunk) < SAMPLE_RATE:
            logger.info("[PIPELINE] session=%s | Audio too short, skipping", session.session_id)
            return [], 0

        real_duration = session.audio_buffer.real_total_duration
        chunk_duration = len(audio_chunk) / SAMPLE_RATE

        logger.info(
            "[PIPELINE] session=%s | stage=%s | chunk=%.1fs (overlap=%.1fs) | session=%.1fs",
            session.session_id, session.stage.value,
            chunk_duration, overlap_duration, real_duration,
        )

        # 1. ASR on chunk, Diarization on longer window (last 30s) for stable speaker ID
        t0 = time.monotonic()

        asr_task = self.transcription.transcribe(audio_chunk)

        # Diarize last 30s of full audio — longer context = more stable speaker separation
        full_audio = session.audio_buffer.get_full_audio()
        max_diar_samples = 30 * SAMPLE_RATE
        diar_audio = full_audio[-max_diar_samples:] if len(full_audio) > max_diar_samples else full_audio
        diar_task = self.diarization.diarize(diar_audio, SAMPLE_RATE, session.num_speakers)

        # Diar timestamps are relative to diar_audio start — need offset to absolute
        diar_window_duration = len(diar_audio) / SAMPLE_RATE
        diar_offset = max(0, real_duration - diar_window_duration)

        asr_segments, diar_segments = await asyncio.gather(asr_task, diar_task)
        parallel_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "[PIPELINE] ASR+Diar: %d asr, %d diar in %dms | speakers=%s | diar_window=%.1fs",
            len(asr_segments), len(diar_segments), parallel_ms,
            {s.speaker for s in diar_segments}, diar_window_duration,
        )

        if not asr_segments:
            return [], 0

        # 2. Filter out overlap segments — keep only segments from the "new" portion
        if overlap_duration > 0:
            asr_segments = [s for s in asr_segments if s.start >= overlap_duration - 0.1]
            if not asr_segments:
                return [], 0

        # 3. time_offset converts ASR timestamps to absolute session time
        # ASR timestamps are relative to audio_chunk start (which includes overlap)
        time_offset = max(0, real_duration - chunk_duration)

        utterances = align_segments(
            asr_segments, diar_segments, session.speaker_map, time_offset,
            calibration_end_time=session.calibration_end_time,
            diar_offset=diar_offset,
        )

        if not utterances:
            return [], 0

        # 4. Update transcript BEFORE extraction so LLM sees full context
        session.update_transcript(utterances)

        # 5. Protocol extraction (1 LLM call per cycle)
        t0 = time.monotonic()
        # Sliding window: send only recent context, not full transcript
        recent = session.transcript[-15:] if len(session.transcript) > 15 else session.transcript
        context = "\n".join(
            f"[{'Врач' if u.speaker == 'doctor' else 'Пациент'}]: {u.text}"
            for u in recent
        )
        session.protocol = await self.llm.extract_protocol_data(
            session.protocol, utterances, context,
        )
        extract_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PIPELINE] Protocol extraction: %dms", extract_ms)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("[PIPELINE] Cycle complete: %dms total", processing_time_ms)

        return utterances, processing_time_ms

    async def process_full_calibration(self, session: SessionState) -> tuple[list[Utterance], int]:
        """Process ALL calibration audio as one batch. Called once at stop_calibration."""
        start_time = time.monotonic()

        full_audio = session.audio_buffer.get_full_audio()
        if len(full_audio) < SAMPLE_RATE:
            logger.warning("[PIPELINE] Calibration: no audio to process")
            return [], 0

        audio_duration = len(full_audio) / SAMPLE_RATE
        logger.info("[PIPELINE] Full calibration: %.1fs audio", audio_duration)

        # 1. ASR + Diarization on FULL calibration audio — timestamps in same frame [0..duration]
        t0 = time.monotonic()
        asr_segments, diar_segments = await asyncio.gather(
            self.transcription.transcribe(full_audio),
            self.diarization.diarize(full_audio, SAMPLE_RATE, session.num_speakers),
        )
        parallel_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "[PIPELINE] Calibration ASR+Diar: %d asr, %d diar in %dms | speakers=%s",
            len(asr_segments), len(diar_segments), parallel_ms,
            {s.speaker for s in diar_segments},
        )

        if not asr_segments:
            return [], 0

        # 2. Alignment — both ASR and diar are in [0..duration], no offset needed
        utterances = align_segments(
            asr_segments, diar_segments, session.speaker_map,
            time_offset=0.0, calibration_end_time=0.0, diar_offset=0.0,
        )

        if not utterances:
            return [], 0

        # 3. Medical term correction
        t0 = time.monotonic()
        all_texts = [u.text for u in utterances]
        corrected_parts = await self.llm.correct_medical_terms_batch(all_texts)
        if corrected_parts and len(corrected_parts) == len(utterances):
            for i, part in enumerate(corrected_parts):
                utterances[i] = Utterance(
                    speaker=utterances[i].speaker, text=part.strip(),
                    start=utterances[i].start, end=utterances[i].end,
                )
        correct_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PIPELINE] Calibration medical correction: %dms", correct_ms)

        # 4. Replace transcript (not append — calibration is processed as one batch)
        session.transcript = []
        session.update_transcript(utterances)

        # 5. Extract patient info from patient utterances
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
                "[PIPELINE] Calibration result: name=%s age=%s gender=%s | speakers=%s",
                session.protocol.patient_info.full_name,
                session.protocol.patient_info.age,
                session.protocol.patient_info.gender,
                session.speaker_map,
            )
        else:
            logger.warning("[PIPELINE] Calibration: no patient utterances found! speakers=%s", session.speaker_map)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("[PIPELINE] Full calibration complete: %dms", processing_time_ms)

        return utterances, processing_time_ms

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
        cds.examination_quality.recording_duration_sec = session.audio_buffer.real_total_duration

        time_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PIPELINE] Finalization complete: %dms | diagnosis=%s",
                     time_ms, bool(session.protocol.exam_data.diagnosis))
        return time_ms
