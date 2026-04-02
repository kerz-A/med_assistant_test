"""Stream WAV files through WebSocket to Aiston TT backend.

Simulates the full 4-stage flow:
  1. Calibration: stream calibration WAV → stop calibration
  2. Recording: stream exam WAV → stop recording
  3. Wait for user to press Enter for finalization
  4. Finalize

Usage:
    pip install websockets

    # Full flow with two files:
    python ws_streamer.py test_calibration.wav test_exam.wav

    # Stream single file as recording (skip calibration):
    python ws_streamer.py --no-calibration test_dialogue.wav

    # Custom server:
    python ws_streamer.py --url ws://192.168.1.100:8000/ws/session test_calibration.wav test_exam.wav
"""

import argparse
import asyncio
import json
import os
import struct
import sys
import wave

import websockets

EXPECTED_SAMPLE_RATE = 16000


def validate_wav(wav_path: str) -> int:
    """Validate WAV file format. Returns sample rate."""
    with wave.open(wav_path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        if channels != 1:
            raise ValueError(f"{wav_path}: Expected mono, got {channels} channels")
        if sampwidth != 2:
            raise ValueError(f"{wav_path}: Expected 16-bit, got {sampwidth*8}-bit")
        if sr != EXPECTED_SAMPLE_RATE:
            print(f"    ⚠ WARNING: {wav_path} sample rate is {sr}Hz, expected {EXPECTED_SAMPLE_RATE}Hz")
            print(f"    ⚠ Backend expects 16kHz. Timestamps may be incorrect!")
        return sr


async def stream_wav(ws, wav_path: str, chunk_duration_ms: int = 100) -> float:
    """Stream WAV file through WebSocket in real-time chunks. Returns duration in seconds."""
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        total_frames = wf.getnframes()
        duration = total_frames / sr

        chunk_frames = int(sr * chunk_duration_ms / 1000)
        sent = 0

        print(f"    Streaming: {wav_path} ({duration:.1f}s, {sr}Hz)")

        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break

            await ws.send(frames)
            sent += len(frames) // 2  # 2 bytes per sample

            # Real-time pacing
            await asyncio.sleep(chunk_duration_ms / 1000)

        print(f"    Sent {sent} samples ({sent/sr:.1f}s)")
        return duration


async def wait_for_messages(ws, timeout: float = 30.0, stop_on: str | None = None,
                           transcript_out: list | None = None,
                           protocol_out: dict | None = None) -> list[dict]:
    """Collect server messages until timeout or specific status."""
    messages = []
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            messages.append(msg)

            msg_type = msg.get("type", "?")
            status = msg.get("status", "")

            if msg_type == "status":
                print(f"    ← Status: {status} {msg.get('message', '')}")
            elif msg_type == "transcript_update":
                for u in msg.get("utterances", []):
                    icon = "👨‍⚕️" if u["speaker"] == "doctor" else "🤒"
                    print(f"    ← {icon} {u['speaker']}: {u['text'][:120]}")
                    if transcript_out is not None:
                        transcript_out.append({"speaker": u["speaker"], "text": u["text"]})
            elif msg_type == "protocol_update":
                if protocol_out is not None:
                    protocol_out.update(msg.get("protocol", {}))
                filled = msg.get("filled_fields", [])
                if filled:
                    print(f"    ← Protocol: {', '.join(filled)}")
            elif msg_type == "calibration_complete":
                pi = msg.get("patient_info", {})
                print(f"    ← Calibration OK: {pi.get('full_name', '?')}, {pi.get('age', '?')} лет")
            else:
                print(f"    ← {msg_type}: {str(msg)[:80]}")

            if stop_on and status == stop_on:
                break

    except asyncio.TimeoutError:
        pass
    except websockets.ConnectionClosed:
        print("    ← Connection closed")

    return messages


async def run(args):
    url = args.url
    print(f"\n{'='*60}")
    print(f"Aiston TT v4 — WebSocket Test Streamer (VAD + Speaker Embeddings)")
    print(f"Server: {url}")
    print(f"{'='*60}")

    # Validate WAV files upfront
    if args.calibration_wav:
        validate_wav(args.calibration_wav)
    if args.exam_wav:
        validate_wav(args.exam_wav)

    # Collectors for evaluation output
    collected_transcript = []
    final_protocol = {}

    async with websockets.connect(url, ping_interval=30, ping_timeout=300, max_size=10 * 1024 * 1024) as ws:

        # ---- STAGE 1: CALIBRATION ----
        if not args.no_calibration and args.calibration_wav:
            print(f"\n[STAGE 1] Calibration: {args.calibration_wav}")
            await ws.send(json.dumps({"type": "start_calibration", "config": {"num_speakers": 2}}))
            await asyncio.sleep(0.5)

            await stream_wav(ws, args.calibration_wav)

            # Wait for processing messages (NOT for "calibrated" — that comes after stop)
            print("    Waiting for calibration processing...")
            await wait_for_messages(ws, timeout=10, transcript_out=collected_transcript, protocol_out=final_protocol)

            # Stop calibration — server will process remaining audio and respond with "calibrated"
            print("    Sending stop_calibration...")
            await ws.send(json.dumps({"type": "stop_calibration"}))
            await wait_for_messages(ws, timeout=45, stop_on="calibrated", transcript_out=collected_transcript, protocol_out=final_protocol)
            print("    ✅ Calibration done")

        else:
            print("\n[STAGE 1] Skipping calibration (--no-calibration)")
            await ws.send(json.dumps({"type": "start_calibration", "config": {"num_speakers": 2}}))
            await asyncio.sleep(1)
            await ws.send(json.dumps({"type": "stop_calibration"}))
            await wait_for_messages(ws, timeout=10, stop_on="calibrated", transcript_out=collected_transcript, protocol_out=final_protocol)

        # ---- STAGE 2: RECORDING ----
        exam_wav = args.exam_wav or args.calibration_wav
        print(f"\n[STAGE 2] Recording: {exam_wav}")
        await ws.send(json.dumps({"type": "start_recording"}))
        await asyncio.sleep(0.5)

        await stream_wav(ws, exam_wav)

        # Wait for processing to catch up
        print("    Waiting for processing to catch up...")
        await wait_for_messages(ws, timeout=45, transcript_out=collected_transcript, protocol_out=final_protocol)

        # ---- STAGE 3: STOP ----
        print(f"\n[STAGE 3] Stopping recording...")
        await ws.send(json.dumps({"type": "stop_recording"}))
        await wait_for_messages(ws, timeout=45, stop_on="stopped", transcript_out=collected_transcript, protocol_out=final_protocol)
        print("    ✅ Recording stopped")

        # ---- STAGE 4: FINALIZE ----
        if args.auto_finalize:
            print(f"\n[STAGE 4] Finalizing (auto)...")
        else:
            input(f"\n[STAGE 4] Press Enter to finalize...")

        await ws.send(json.dumps({"type": "finalize"}))
        print("    Generating diagnosis...")
        msgs = await wait_for_messages(ws, timeout=90, stop_on="done", transcript_out=collected_transcript, protocol_out=final_protocol)

        # Print final protocol
        for m in msgs:
            if m.get("type") == "protocol_update":
                proto = m.get("protocol", {})
                print(f"\n{'='*60}")
                print("FINAL PROTOCOL:")
                print(f"{'='*60}")
                pi = proto.get("patient_info", {})
                print(f"Patient: {pi.get('full_name', '?')}, {pi.get('age', '?')} лет, {pi.get('gender', '?')}")
                exam = proto.get("exam_data", {})
                for k, v in exam.items():
                    if v:
                        print(f"  {k}: {v[:80]}{'...' if v and len(str(v)) > 80 else ''}")
                vitals = proto.get("vitals", {})
                for k, v in vitals.items():
                    if v is not None:
                        print(f"  {k}: {v}")
                print(f"{'='*60}")

        print("\n✅ Test complete!")

        # Save result JSON for evaluation
        if hasattr(args, 'output') and args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            result = {
                "scenario_id": getattr(args, 'scenario_id', None) or "unknown",
                "transcript": collected_transcript,
                "protocol": final_protocol,
            }
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"\n📊 Result saved: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Stream WAV to Aiston TT backend")
    parser.add_argument("calibration_wav", help="WAV file for calibration phase")
    parser.add_argument("exam_wav", nargs="?", help="WAV file for exam phase (uses calibration_wav if not specified)")
    parser.add_argument("--url", default="ws://localhost:8000/ws/session", help="WebSocket URL")
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration phase")
    parser.add_argument("--auto-finalize", action="store_true", help="Auto-finalize without waiting for Enter")
    parser.add_argument("--scenario-id", type=str, help="Scenario ID for evaluation (e.g. 01_cardiology)")
    parser.add_argument("--output", type=str, help="Save result JSON for evaluation (e.g. results/01.json)")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
