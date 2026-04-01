"""Processing pipeline v4: VAD-segmented, speaker-embedding-based.

Flow per speech segment:
  1. Speaker ID (ECAPA-TDNN embedding → cosine similarity vs calibration refs)
  2. ASR (faster-whisper on complete speech segment)
  3. Protocol extraction (LLM, during recording only)
"""

import asyncio
import logging
import time

import numpy as np

from ..config import settings
from ..models.messages import Utterance
from ..services.llm import LLMService
from ..services.speaker_id import SpeakerIDService
from ..services.transcription import TranscriptionService
from .audio_buffer import SAMPLE_RATE
from .session import SessionState, SessionStage
from .vad_segmenter import SpeechSegment

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    def __init__(
        self,
        transcription: TranscriptionService,
        speaker_id: SpeakerIDService,
        llm: LLMService,
    ):
        self.transcription = transcription
        self.speaker_id = speaker_id
        self.llm = llm

    async def process_segment(
        self, session: SessionState, segment: SpeechSegment,
    ) -> list[Utterance]:
        """Process a single VAD speech segment during recording.

        1. Speaker ID (embedding) → doctor/patient
        2. ASR (Whisper) → text
        3. Protocol extraction (LLM) → update fields
        """
        start_time = time.monotonic()

        logger.info(
            "[PIPELINE] Processing segment: %.2f-%.2fs (%.1fs) | session=%s",
            segment.start_time, segment.end_time, segment.duration,
            session.session_id,
        )

        # 1. Speaker ID + ASR in parallel
        t0 = time.monotonic()
        speaker_task = self.speaker_id.identify_speaker(
            segment.audio, session.doctor_profile, session.patient_profile,
        )
        asr_task = self.transcription.transcribe(segment.audio)

        (speaker, confidence, embedding), asr_segments = await asyncio.gather(
            speaker_task, asr_task,
        )
        parallel_ms = int((time.monotonic() - t0) * 1000)

        # Combine ASR segments into one text (segment is one speaker turn)
        text = " ".join(seg.text.strip() for seg in asr_segments if seg.text.strip())
        if not text:
            logger.info("[PIPELINE] No text in segment, skipping")
            return []

        logger.info(
            "[PIPELINE] Segment result: speaker=%s (conf=%.3f) text='%s' | %dms",
            speaker, confidence, text[:80], parallel_ms,
        )

        utterance = Utterance(
            speaker=speaker,
            text=text,
            start=round(segment.start_time, 2),
            end=round(segment.end_time, 2),
        )

        # 2. Update transcript
        session.transcript.append(utterance)
        session.sequence += 1

        # 3. Batched protocol extraction — accumulate utterances, extract every 3 or 15s
        if session.stage == SessionStage.RECORDING:
            session.add_pending_utterance(utterance)

            if session.should_extract_protocol():
                pending = session.get_pending_utterances()
                t0 = time.monotonic()
                recent = session.transcript[-15:] if len(session.transcript) > 15 else session.transcript
                context = "\n".join(
                    f"[{'Врач' if u.speaker == 'doctor' else 'Пациент'}]: {u.text}"
                    for u in recent
                )
                session.protocol = await self.llm.extract_protocol_data(
                    session.protocol, pending, context,
                )
                extract_ms = int((time.monotonic() - t0) * 1000)
                logger.info("[PIPELINE] Protocol extraction: %dms (%d utterances batched)",
                            extract_ms, len(pending))

        total_ms = int((time.monotonic() - start_time) * 1000)
        logger.info("[PIPELINE] Segment complete: %dms total", total_ms)

        return [utterance]

    async def process_calibration_segment(
        self, session: SessionState, segment: SpeechSegment,
    ) -> list[Utterance]:
        """Process a speech segment during calibration.

        Identifies speaker role based on order (first = doctor) and
        stores embedding for later classification.
        """
        t0 = time.monotonic()

        # Extract embedding + ASR in parallel
        embedding_task = self.speaker_id.extract_embedding(segment.audio)
        asr_task = self.transcription.transcribe(segment.audio)

        embedding, asr_segments = await asyncio.gather(embedding_task, asr_task)

        text = " ".join(seg.text.strip() for seg in asr_segments if seg.text.strip())
        if not text:
            return []

        # Determine speaker role
        if not session._calibration_first_speaker_set:
            # First speaker in calibration = doctor (convention: doctor asks first)
            role = "doctor"
            session.doctor_profile.add(embedding)
            session._calibration_first_speaker_set = True
            logger.info("[PIPELINE] Calibration: first speaker → doctor")
        else:
            # Subsequent speakers: check if same as doctor or new
            from ..services.speaker_id import _cosine_similarity
            doctor_sim = _cosine_similarity(embedding, session.doctor_profile.mean_embedding)

            if doctor_sim > 0.6:
                # Same speaker as doctor
                role = "doctor"
                session.doctor_profile.add(embedding)
                logger.info("[PIPELINE] Calibration: same speaker → doctor (sim=%.3f)", doctor_sim)
            else:
                # New speaker = patient
                role = "patient"
                session.patient_profile.add(embedding)
                logger.info("[PIPELINE] Calibration: new speaker → patient (doctor_sim=%.3f)", doctor_sim)

        utterance = Utterance(
            speaker=role, text=text,
            start=round(segment.start_time, 2),
            end=round(segment.end_time, 2),
        )

        session.transcript.append(utterance)
        session.sequence += 1

        ms = int((time.monotonic() - t0) * 1000)
        logger.info("[PIPELINE] Calibration segment: %s '%s' in %dms", role, text[:60], ms)

        return [utterance]

    async def finalize_calibration(self, session: SessionState) -> None:
        """Extract patient info from calibration transcript."""
        patient_texts = [u.text for u in session.transcript if u.speaker == "patient"]
        if patient_texts:
            combined = " ".join(patient_texts)
            logger.info("[PIPELINE] Extracting patient info from: %s", combined[:100])
            info = await self.llm.extract_patient_info(combined)
            if info.get("full_name"):
                session.protocol.patient_info.full_name = info["full_name"]
            if info.get("age"):
                session.protocol.patient_info.age = info["age"]
            if info.get("gender"):
                session.protocol.patient_info.gender = info["gender"]

        logger.info(
            "[PIPELINE] Calibration finalized: name=%s age=%s gender=%s | "
            "doctor_embeddings=%d patient_embeddings=%d",
            session.protocol.patient_info.full_name,
            session.protocol.patient_info.age,
            session.protocol.patient_info.gender,
            session.doctor_profile.count,
            session.patient_profile.count,
        )

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
