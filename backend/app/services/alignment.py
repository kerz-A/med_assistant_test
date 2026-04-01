"""Alignment: match ASR segments with diarization segments.

V3 fixes:
- speaker_map persists in session across cycles
- verify_speaker_map: checks diarization against calibration zone to prevent role swaps
- Doctor is ALWAYS the speaker who dominates the first few seconds (calibration zone)
"""

import logging
from collections import defaultdict

from ..models.messages import Utterance
from ..services.diarization import DiarizationSegment
from ..services.transcription import TranscriptSegment

logger = logging.getLogger(__name__)


def _overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    return max(0, min(end1, end2) - max(start1, start2))


def _find_speaker(
    asr_start: float,
    asr_end: float,
    diarization_segments: list[DiarizationSegment],
) -> str:
    best_speaker = "SPEAKER_00"
    best_overlap = 0.0
    for dseg in diarization_segments:
        ov = _overlap(asr_start, asr_end, dseg.start, dseg.end)
        if ov > best_overlap:
            best_overlap = ov
            best_speaker = dseg.speaker
    return best_speaker


def verify_speaker_map(
    diarization_segments: list[DiarizationSegment],
    speaker_map: dict[str, str],
    calibration_end_time: float,
) -> dict[str, str]:
    """Verify and fix speaker_map using calibration zone.

    In calibration, the DOCTOR speaks first (asks "What's your name?").
    So the speaker with the most audio in [0, calibration_end_time] = doctor.
    If pyannote swapped labels between runs, we detect and fix it here.
    """
    if calibration_end_time <= 0 or not diarization_segments:
        return speaker_map

    # Calculate how much each speaker talks in calibration zone
    speaker_time: dict[str, float] = defaultdict(float)
    for seg in diarization_segments:
        ov = _overlap(seg.start, seg.end, 0.0, calibration_end_time)
        if ov > 0:
            speaker_time[seg.speaker] += ov

    if not speaker_time:
        logger.warning("[ALIGN] No diarization segments in calibration zone [0, %.1f]", calibration_end_time)
        return speaker_map

    # Doctor = speaker with most time in calibration zone (they ask the first question)
    doctor_label = max(speaker_time, key=speaker_time.get)

    # Check if current map is correct
    current_doctor = None
    for label, role in speaker_map.items():
        if role == "doctor":
            current_doctor = label
            break

    if current_doctor and current_doctor != doctor_label:
        # Labels swapped! Fix the map
        old_map = dict(speaker_map)
        speaker_map.clear()
        speaker_map[doctor_label] = "doctor"
        for label in speaker_time:
            if label != doctor_label:
                speaker_map[label] = "patient"
        logger.warning(
            "[ALIGN] Speaker labels SWAPPED! Fixed: %s → %s (doctor was %s in calibration zone)",
            old_map, speaker_map, doctor_label,
        )
    elif not current_doctor:
        # First time mapping
        speaker_map[doctor_label] = "doctor"
        for label in speaker_time:
            if label != doctor_label:
                speaker_map[label] = "patient"
        logger.info("[ALIGN] Initial speaker map from calibration: %s", speaker_map)

    return speaker_map


def align_segments(
    asr_segments: list[TranscriptSegment],
    diarization_segments: list[DiarizationSegment],
    speaker_map: dict[str, str],
    time_offset: float = 0.0,
    calibration_end_time: float = 0.0,
    diar_offset: float = 0.0,
) -> list[Utterance]:
    """Align ASR with diarization. speaker_map is MUTATED and persisted in session.

    Args:
        diar_offset: Absolute time offset to add to diarization segment timestamps.
            When ASR and diarization process the same audio chunk, both need the same
            offset to convert to absolute session time.
            When diarization processes full_audio (calibration), offset is 0.
    """
    if not asr_segments:
        return []

    # Shift diarization timestamps to absolute session time
    if diar_offset > 0:
        diarization_segments = [
            DiarizationSegment(speaker=s.speaker, start=s.start + diar_offset, end=s.end + diar_offset)
            for s in diarization_segments
        ]

    # Verify speaker roles haven't swapped
    if calibration_end_time > 0:
        verify_speaker_map(diarization_segments, speaker_map, calibration_end_time)

    logger.info(
        "[ALIGN] %d ASR + %d diar segments, offset=%.1f, map=%s",
        len(asr_segments), len(diarization_segments), time_offset, speaker_map,
    )

    utterances = []
    for seg in asr_segments:
        abs_start = seg.start + time_offset
        abs_end = seg.end + time_offset
        raw_speaker = _find_speaker(abs_start, abs_end, diarization_segments)

        # Map label — if unknown, use calibration-verified map logic
        if raw_speaker not in speaker_map:
            if len(speaker_map) == 0:
                speaker_map[raw_speaker] = "doctor"
            else:
                speaker_map[raw_speaker] = "patient"
            logger.info("[ALIGN] New speaker: %s -> %s", raw_speaker, speaker_map[raw_speaker])

        speaker = speaker_map[raw_speaker]

        utterances.append(Utterance(
            speaker=speaker,
            text=seg.text,
            start=round(abs_start, 2),
            end=round(abs_end, 2),
        ))

    logger.info("[ALIGN] Result: %d utterances, map=%s", len(utterances), speaker_map)
    return utterances
