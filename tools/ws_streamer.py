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
import struct
import sys
import wave

import websockets


async def stream_wav(ws, wav_path: str, chunk_duration_ms: int = 100) -> float:
    """Stream WAV file through WebSocket in real-time chunks. Returns duration in seconds."""
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono, got {wf.getnchannels()} channels"
        assert wf.getsampwidth() == 2, f"Expected 16-bit, got {wf.getsampwidth()*8}-bit"
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

            # If sample rate != 16000, we need to resample
            # For simplicity, just send raw — backend expects 16kHz int16 mono
            await ws.send(frames)
            sent += len(frames) // 2  # 2 bytes per sample

            # Real-time pacing
            await asyncio.sleep(chunk_duration_ms / 1000)

        print(f"    Sent {sent} samples ({sent/sr:.1f}s)")
        return duration


async def wait_for_messages(ws, timeout: float = 30.0, stop_on: str | None = None) -> list[dict]:
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
                    print(f"    ← {icon} {u['speaker']}: {u['text'][:60]}")
            elif msg_type == "protocol_update":
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
    print(f"Aiston TT v3 — WebSocket Test Streamer")
    print(f"Server: {url}")
    print(f"{'='*60}")

    async with websockets.connect(url) as ws:

        # ---- STAGE 1: CALIBRATION ----
        if not args.no_calibration and args.calibration_wav:
            print(f"\n[STAGE 1] Calibration: {args.calibration_wav}")
            await ws.send(json.dumps({"type": "start_calibration", "config": {"num_speakers": 2}}))
            await asyncio.sleep(0.5)

            await stream_wav(ws, args.calibration_wav)

            # Wait a bit for processing
            print("    Waiting for calibration processing...")
            await wait_for_messages(ws, timeout=20, stop_on="calibrated")

            # Stop calibration
            await ws.send(json.dumps({"type": "stop_calibration"}))
            await wait_for_messages(ws, timeout=15, stop_on="calibrated")
            print("    ✅ Calibration done")

        else:
            print("\n[STAGE 1] Skipping calibration (--no-calibration)")
            await ws.send(json.dumps({"type": "start_calibration", "config": {"num_speakers": 2}}))
            await asyncio.sleep(1)
            await ws.send(json.dumps({"type": "stop_calibration"}))
            await wait_for_messages(ws, timeout=5, stop_on="calibrated")

        # ---- STAGE 2: RECORDING ----
        exam_wav = args.exam_wav or args.calibration_wav
        print(f"\n[STAGE 2] Recording: {exam_wav}")
        await ws.send(json.dumps({"type": "start_recording"}))
        await asyncio.sleep(0.5)

        await stream_wav(ws, exam_wav)

        # Wait for processing to catch up
        print("    Waiting for processing to catch up...")
        await wait_for_messages(ws, timeout=30)

        # ---- STAGE 3: STOP ----
        print(f"\n[STAGE 3] Stopping recording...")
        await ws.send(json.dumps({"type": "stop_recording"}))
        await wait_for_messages(ws, timeout=20, stop_on="stopped")
        print("    ✅ Recording stopped")

        # ---- STAGE 4: FINALIZE ----
        if args.auto_finalize:
            print(f"\n[STAGE 4] Finalizing (auto)...")
        else:
            input(f"\n[STAGE 4] Press Enter to finalize...")

        await ws.send(json.dumps({"type": "finalize"}))
        print("    Generating diagnosis...")
        msgs = await wait_for_messages(ws, timeout=60, stop_on="done")

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


def main():
    parser = argparse.ArgumentParser(description="Stream WAV to Aiston TT backend")
    parser.add_argument("calibration_wav", help="WAV file for calibration phase")
    parser.add_argument("exam_wav", nargs="?", help="WAV file for exam phase (uses calibration_wav if not specified)")
    parser.add_argument("--url", default="ws://localhost:8000/ws/session", help="WebSocket URL")
    parser.add_argument("--no-calibration", action="store_true", help="Skip calibration phase")
    parser.add_argument("--auto-finalize", action="store_true", help="Auto-finalize without waiting for Enter")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
