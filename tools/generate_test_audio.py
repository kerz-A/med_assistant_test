"""Generate test audio with two speakers (doctor + patient) using edge-tts.

No pydub! Uses imageio-ffmpeg for mp3→wav conversion via subprocess.

Usage:
    pip install edge-tts imageio-ffmpeg
    python generate_test_audio.py
"""

import asyncio
import os
import subprocess
import struct
import tempfile
import wave

import edge_tts
import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Russian voices
DOCTOR_VOICE = "ru-RU-DmitryNeural"    # male
PATIENT_VOICE = "ru-RU-SvetlanaNeural"  # female

TARGET_SR = 16000


# ============================================================
# DIALOGUE
# ============================================================

CALIBRATION_DIALOGUE = [
    ("doctor", "Здравствуйте. Как вас зовут? Сколько вам полных лет? И ваш пол?"),
    ("patient", "Здравствуйте. Меня зовут Иванова Мария Петровна. Мне сорок пять лет. Женский."),
]

EXAM_DIALOGUE = [
    ("doctor", "Расскажите, что вас беспокоит."),
    ("patient", "Меня беспокоит головная боль уже около двух недель. Болит в основном в области висков и затылка. Боль давящая, иногда пульсирующая. По шкале от одного до десяти я бы оценила на шесть семь баллов. Усиливается к вечеру и когда долго сижу за компьютером."),
    ("doctor", "Когда впервые появились эти симптомы? Обращались ли вы к врачам?"),
    ("patient", "Головная боль началась примерно две недели назад, после того как я переболела простудой. К врачу пока не обращалась. Принимаю нурофен четыреста миллиграмм, помогает на три четыре часа."),
    ("doctor", "Подскажите ваш рост и вес. Знаете ли своё давление?"),
    ("patient", "Мой рост сто семьдесят восемь сантиметров, вес восемьдесят два килограмма. Давление обычно сто тридцать на восемьдесят пять. Пульс по часам семьдесят два удара в минуту."),
    ("doctor", "Есть ли хронические заболевания, аллергии? Курите?"),
    ("patient", "Из хронических у меня гастрит, стоит на учёте лет пять. Аллергия на амоксициллин, была сыпь. Не курю. Алкоголь по праздникам. У мамы гипертония и сахарный диабет второго типа. У отца был инфаркт в пятьдесят пять лет."),
    ("doctor", "Как со сном? Стресс на работе?"),
    ("patient", "Сплю плохо последнее время, часов пять шесть. На работе аврал, стресс сильный. За компьютером сижу по десять часов в день."),
]


def mp3_to_pcm(mp3_path: str) -> bytes:
    """Convert mp3 to raw PCM16 16kHz mono using ffmpeg."""
    result = subprocess.run(
        [FFMPEG, "-i", mp3_path, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(TARGET_SR), "-ac", "1", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
    return result.stdout


def silence(duration_ms: int) -> bytes:
    """Generate silence as PCM16 bytes."""
    n = int(TARGET_SR * duration_ms / 1000)
    return b'\x00\x00' * n


def save_wav(path: str, pcm: bytes):
    """Save raw PCM16 mono 16kHz as WAV."""
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm)
    sec = len(pcm) / 2 / TARGET_SR
    print(f"  Saved: {path} ({sec:.1f}s)")


async def tts_to_pcm(text: str, voice: str) -> bytes:
    """Text → mp3 (edge-tts) → PCM16 (ffmpeg)."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(tmp_fd)
    try:
        comm = edge_tts.Communicate(text, voice, rate="-5%")
        await comm.save(tmp_path)
        return mp3_to_pcm(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except (PermissionError, OSError):
            pass


async def build_dialogue(dialogue: list, pause_ms: int = 800) -> bytes:
    """Build PCM audio from dialogue."""
    pcm = silence(500)
    gap = silence(pause_ms)

    for i, (speaker, text) in enumerate(dialogue):
        voice = DOCTOR_VOICE if speaker == "doctor" else PATIENT_VOICE
        icon = "👨‍⚕️" if speaker == "doctor" else "🤒"
        print(f"  [{i+1}/{len(dialogue)}] {icon} {text[:55]}...")
        pcm += await tts_to_pcm(text, voice) + gap

    return pcm


async def main():
    print("=" * 60)
    print("Generating test audio for MedScribe")
    print(f"ffmpeg: {FFMPEG}")
    print("=" * 60)

    print("\n[1/3] Calibration dialogue...")
    cal = await build_dialogue(CALIBRATION_DIALOGUE, 1500)

    print("\n[2/3] Exam dialogue...")
    exam = await build_dialogue(EXAM_DIALOGUE, 800)

    print("\n[3/3] Saving...")
    save_wav("test_calibration.wav", cal)
    save_wav("test_exam.wav", exam)
    save_wav("test_dialogue.wav", cal + silence(2000) + exam)

    print(f"\n{'='*60}")
    print("Done! Run:")
    print("  python ws_streamer.py test_calibration.wav test_exam.wav --auto-finalize")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
