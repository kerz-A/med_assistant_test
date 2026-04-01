import numpy as np

SAMPLE_RATE = 16000


class AudioRingBuffer:
    """Circular buffer for PCM int16 audio at 16kHz mono."""

    def __init__(self, max_duration_seconds: int = 120):
        self._max_samples = max_duration_seconds * SAMPLE_RATE
        self._buffer = np.zeros(self._max_samples, dtype=np.float32)
        self._write_pos = 0
        self._total_written = 0
        self._last_processed_pos = 0

    @property
    def total_duration(self) -> float:
        return min(self._total_written, self._max_samples) / SAMPLE_RATE

    @property
    def real_total_duration(self) -> float:
        """Real elapsed session time, not capped by buffer size."""
        return self._total_written / SAMPLE_RATE

    @property
    def unprocessed_duration(self) -> float:
        return (self._total_written - self._last_processed_pos) / SAMPLE_RATE

    def append(self, pcm_bytes: bytes) -> None:
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        n = len(samples)

        if n >= self._max_samples:
            samples = samples[-self._max_samples:]
            self._buffer[:] = samples
            self._write_pos = 0
            self._total_written += n
            return

        end_pos = self._write_pos + n
        if end_pos <= self._max_samples:
            self._buffer[self._write_pos:end_pos] = samples
        else:
            first_chunk = self._max_samples - self._write_pos
            self._buffer[self._write_pos:] = samples[:first_chunk]
            self._buffer[:n - first_chunk] = samples[first_chunk:]

        self._write_pos = end_pos % self._max_samples
        self._total_written += n

    def get_full_audio(self) -> np.ndarray:
        valid_samples = min(self._total_written, self._max_samples)
        if valid_samples == 0:
            return np.zeros(0, dtype=np.float32)

        if self._total_written <= self._max_samples:
            return self._buffer[:valid_samples].copy()

        result = np.empty(self._max_samples, dtype=np.float32)
        first_chunk = self._max_samples - self._write_pos
        result[:first_chunk] = self._buffer[self._write_pos:]
        result[first_chunk:] = self._buffer[:self._write_pos]
        return result

    def get_unprocessed_audio(self) -> np.ndarray:
        unprocessed_samples = self._total_written - self._last_processed_pos
        if unprocessed_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        full = self.get_full_audio()
        n = min(unprocessed_samples, len(full))
        return full[-n:]

    def get_unprocessed_audio_with_overlap(self, overlap_seconds: float = 2.0) -> tuple[np.ndarray, float]:
        """Get unprocessed audio plus context from the end of the previous chunk.

        Returns (audio_with_overlap, overlap_duration_seconds).
        overlap_duration = how many seconds at the start are context from the previous chunk.
        """
        unprocessed_samples = self._total_written - self._last_processed_pos
        if unprocessed_samples <= 0:
            return np.zeros(0, dtype=np.float32), 0.0

        overlap_samples = int(overlap_seconds * SAMPLE_RATE)
        actual_overlap = min(overlap_samples, self._last_processed_pos)
        total_needed = unprocessed_samples + actual_overlap

        full = self.get_full_audio()
        n = min(total_needed, len(full))
        return full[-n:], actual_overlap / SAMPLE_RATE

    def mark_processed(self) -> None:
        self._last_processed_pos = self._total_written

    def reset(self) -> None:
        self._buffer[:] = 0
        self._write_pos = 0
        self._total_written = 0
        self._last_processed_pos = 0
