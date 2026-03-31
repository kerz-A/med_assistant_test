"""Alignment: match ASR segments with diarization segments.

V3 fix: speaker_map is passed in (stored in session) so it persists across cycles.
"""

import logging

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


def _map_speaker_label(raw_label: str, speaker_map: dict[str, str]) -> str:
    """Map raw pyannote label to doctor/patient. First speaker = doctor."""
    if raw_label not in speaker_map:
        if len(speaker_map) == 0:
            speaker_map[raw_label] = "doctor"
        else:
            speaker_map[raw_label] = "patient"
        logger.info("[ALIGN] Speaker mapping: %s -> %s", raw_label, speaker_map[raw_label])
    return speaker_map[raw_label]


def align_segments(
    asr_segments: list[TranscriptSegment],
    diarization_segments: list[DiarizationSegment],
    speaker_map: dict[str, str],
    time_offset: float = 0.0,
) -> list[Utterance]:
    """Align ASR with diarization. speaker_map is MUTATED and persisted in session."""
    if not asr_segments:
        return []

    logger.info(
        "[ALIGN] %d ASR + %d diar segments, offset=%.1f, existing_map=%s",
        len(asr_segments), len(diarization_segments), time_offset, speaker_map,
    )

    utterances = []
    for seg in asr_segments:
        abs_start = seg.start + time_offset
        abs_end = seg.end + time_offset
        raw_speaker = _find_speaker(abs_start, abs_end, diarization_segments)
        speaker = _map_speaker_label(raw_speaker, speaker_map)

        utterances.append(Utterance(
            speaker=speaker,
            text=seg.text,
            start=round(abs_start, 2),
            end=round(abs_end, 2),
        ))

    logger.info("[ALIGN] Result: %d utterances, map=%s", len(utterances), speaker_map)
    return utterances
