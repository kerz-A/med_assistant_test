"""Generate test audio for all medical dialogue scenarios.

Extends generate_test_audio.py with support for multiple scenarios,
pitch/rate adjustments, and per-scenario output directories.

Usage:
    pip install edge-tts imageio-ffmpeg

    python generate_test_scenarios.py              # all scenarios
    python generate_test_scenarios.py --scenario 01_cardiology  # single
    python generate_test_scenarios.py --list        # list available
"""

import argparse
import asyncio
import os
import subprocess
import tempfile
import wave

import edge_tts
import imageio_ffmpeg

from test_scenarios import SCENARIOS

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
TARGET_SR = 16000


def mp3_to_pcm(mp3_path: str) -> bytes:
    result = subprocess.run(
        [FFMPEG, "-i", mp3_path, "-f", "s16le", "-acodec", "pcm_s16le",
         "-ar", str(TARGET_SR), "-ac", "1", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr.decode()[:200]}")
    return result.stdout


def silence(duration_ms: int) -> bytes:
    n = int(TARGET_SR * duration_ms / 1000)
    return b'\x00\x00' * n


def save_wav(path: str, pcm: bytes):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm)
    sec = len(pcm) / 2 / TARGET_SR
    print(f"  Saved: {path} ({sec:.1f}s)")


async def tts_to_pcm(text: str, voice: str, rate: str = "-5%", pitch: str = "+0Hz") -> bytes:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(tmp_fd)
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        await comm.save(tmp_path)
        return mp3_to_pcm(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except (PermissionError, OSError):
            pass


async def build_dialogue(dialogue: list, scenario: dict, pause_ms: int = 800) -> bytes:
    pcm = silence(500)
    gap = silence(pause_ms)

    for i, (speaker, text) in enumerate(dialogue):
        if speaker == "doctor":
            voice = scenario["doctor_voice"]
            rate = scenario["doctor_rate"]
            pitch = scenario["doctor_pitch"]
        else:
            voice = scenario["patient_voice"]
            rate = scenario["patient_rate"]
            pitch = scenario["patient_pitch"]

        icon = "👨‍⚕️" if speaker == "doctor" else "🤒"
        print(f"  [{i+1}/{len(dialogue)}] {icon} {text[:55]}...")
        pcm += await tts_to_pcm(text, voice, rate, pitch) + gap

    return pcm


async def generate_scenario(scenario: dict, output_dir: str):
    sid = scenario["id"]
    name = scenario["name"]
    out_dir = os.path.join(output_dir, sid)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scenario: {sid} — {name}")
    print(f"Doctor:  {scenario['doctor_voice']} rate={scenario['doctor_rate']} pitch={scenario['doctor_pitch']}")
    print(f"Patient: {scenario['patient_voice']} rate={scenario['patient_rate']} pitch={scenario['patient_pitch']}")
    print(f"{'='*60}")

    print("\n  [Calibration]")
    cal = await build_dialogue(scenario["calibration_dialogue"], scenario, 1500)

    print("\n  [Exam]")
    exam = await build_dialogue(scenario["exam_dialogue"], scenario, 800)

    print("\n  [Saving]")
    save_wav(os.path.join(out_dir, "calibration.wav"), cal)
    save_wav(os.path.join(out_dir, "exam.wav"), exam)
    save_wav(os.path.join(out_dir, "dialogue.wav"), cal + silence(2000) + exam)

    print(f"\n  ✅ {sid} done")


async def main():
    parser = argparse.ArgumentParser(description="Generate test audio for medical scenarios")
    parser.add_argument("--scenario", type=str, help="Scenario ID (e.g. 01_cardiology)")
    parser.add_argument("--list", action="store_true", help="List all scenarios")
    parser.add_argument("--output-dir", default="test_scenarios_audio", help="Output directory")
    args = parser.parse_args()

    if args.list:
        print(f"\n{'='*60}")
        print(f"Available scenarios ({len(SCENARIOS)}):")
        print(f"{'='*60}")
        for s in SCENARIOS:
            voices = f"{s['doctor_voice'].split('-')[-1]} → {s['patient_voice'].split('-')[-1]}"
            hard = " ⚠️ HARD (same gender)" if s["doctor_voice"] == s["patient_voice"] else ""
            print(f"  {s['id']}: {s['name']} [{voices}]{hard}")
        return

    print(f"{'='*60}")
    print(f"Aiston TT — Test Scenario Audio Generator")
    print(f"ffmpeg: {FFMPEG}")
    print(f"{'='*60}")

    scenarios = SCENARIOS
    if args.scenario:
        scenarios = [s for s in SCENARIOS if s["id"] == args.scenario]
        if not scenarios:
            print(f"ERROR: Scenario '{args.scenario}' not found")
            print(f"Available: {', '.join(s['id'] for s in SCENARIOS)}")
            return

    for scenario in scenarios:
        await generate_scenario(scenario, args.output_dir)

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(scenarios)} scenario(s)")
    print(f"Run test:")
    sid = scenarios[0]["id"]
    print(f"  python ws_streamer.py {args.output_dir}/{sid}/calibration.wav "
          f"{args.output_dir}/{sid}/exam.wav --auto-finalize")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
