"""Speaker identification using ECAPA-TDNN embeddings.

During calibration: extracts reference embeddings for doctor and patient.
During recording: classifies each speech segment as doctor or patient
by cosine similarity to reference embeddings.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from functools import partial

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SpeakerProfile:
    """Accumulated speaker embedding from multiple segments."""
    embeddings: list[np.ndarray] = field(default_factory=list)
    _mean: np.ndarray | None = None

    def add(self, embedding: np.ndarray) -> None:
        self.embeddings.append(embedding)
        self._mean = None  # Invalidate cache

    @property
    def mean_embedding(self) -> np.ndarray | None:
        if not self.embeddings:
            return None
        if self._mean is None:
            self._mean = np.mean(self.embeddings, axis=0)
            # L2 normalize for cosine similarity
            norm = np.linalg.norm(self._mean)
            if norm > 0:
                self._mean = self._mean / norm
        return self._mean

    @property
    def count(self) -> int:
        return len(self.embeddings)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


class SpeakerIDService:
    """Speaker identification using ECAPA-TDNN embeddings from SpeechBrain."""

    def __init__(self):
        self._model = None
        self._device: str = "cpu"

    def load_model(self) -> None:
        """Load ECAPA-TDNN speaker embedding model."""
        from speechbrain.inference.speaker import EncoderClassifier

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("[SPEAKER_ID] Loading ECAPA-TDNN on %s...", self._device)
        t0 = time.monotonic()

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec-ecapa-voxceleb",
            run_opts={"device": self._device},
        )

        load_ms = int((time.monotonic() - t0) * 1000)
        logger.info("[SPEAKER_ID] ECAPA-TDNN loaded in %dms (device=%s)", load_ms, self._device)

    def _extract_embedding_sync(self, audio: np.ndarray) -> np.ndarray:
        """Extract 192-dim speaker embedding from audio (sync, for run_in_executor)."""
        if self._model is None:
            raise RuntimeError("Speaker ID model not loaded")

        # SpeechBrain expects [batch, time] tensor
        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            embedding = self._model.encode_batch(waveform)
            # embedding shape: [1, 1, 192]
            emb = embedding[0, 0, :].cpu().numpy()

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        return emb

    async def extract_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding (async wrapper)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, partial(self._extract_embedding_sync, audio)
        )

    def classify_speaker(
        self,
        embedding: np.ndarray,
        doctor_profile: SpeakerProfile,
        patient_profile: SpeakerProfile,
    ) -> tuple[str, float]:
        """Classify embedding as doctor or patient.

        Returns (role, confidence) where confidence = abs(doctor_sim - patient_sim).
        Higher confidence = more certain classification.
        """
        doctor_ref = doctor_profile.mean_embedding
        patient_ref = patient_profile.mean_embedding

        if doctor_ref is None and patient_ref is None:
            return "unknown", 0.0

        if doctor_ref is None:
            return "patient", 1.0

        if patient_ref is None:
            # Only doctor known — check if this sounds like doctor
            sim = _cosine_similarity(embedding, doctor_ref)
            if sim > 0.5:
                return "doctor", sim
            else:
                return "patient", 1.0 - sim

        # Both profiles known — compare similarities
        doctor_sim = _cosine_similarity(embedding, doctor_ref)
        patient_sim = _cosine_similarity(embedding, patient_ref)
        confidence = abs(doctor_sim - patient_sim)

        role = "doctor" if doctor_sim > patient_sim else "patient"

        logger.info(
            "[SPEAKER_ID] Classification: %s (doctor_sim=%.3f, patient_sim=%.3f, conf=%.3f)",
            role, doctor_sim, patient_sim, confidence,
        )

        return role, confidence

    async def identify_speaker(
        self,
        audio: np.ndarray,
        doctor_profile: SpeakerProfile,
        patient_profile: SpeakerProfile,
    ) -> tuple[str, float, np.ndarray]:
        """Full pipeline: extract embedding + classify.

        Returns (role, confidence, embedding).
        """
        embedding = await self.extract_embedding(audio)
        role, confidence = self.classify_speaker(embedding, doctor_profile, patient_profile)
        return role, confidence, embedding
