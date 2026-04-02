"""Run all test scenarios sequentially and evaluate results.

Orchestrates the full test pipeline:
  1. For each scenario: stream calibration + exam audio via WebSocket
  2. Capture transcript and protocol as JSON
  3. After all scenarios: run evaluation metrics and print report

Usage:
    pip install websockets

    # Run all 10 scenarios
    python run_all_scenarios.py

    # Run specific scenarios
    python run_all_scenarios.py --scenarios 01_cardiology 02_gastroenterology

    # Run only "hard" same-gender scenarios
    python run_all_scenarios.py --hard-only

    # Custom server and audio directory
    python run_all_scenarios.py --url ws://192.168.1.100:8000/ws/session --audio-dir test_scenarios_audio

    # Skip evaluation (just run tests)
    python run_all_scenarios.py --no-eval
"""

import argparse
import asyncio
import json
import os
import sys
import time

import websockets

# Reuse existing tools
from test_scenarios import SCENARIOS

EXPECTED_SAMPLE_RATE = 16000
SAME_GENDER_IDS = {"03_neurology", "04_pulmonology", "08_urology", "09_pediatrics"}


# ============================================================
# Audio streaming (adapted from ws_streamer.py)
# ============================================================

async def stream_wav(ws, wav_path: str, chunk_duration_ms: int = 100) -> float:
    import wave
    with wave.open(wav_path, "rb") as wf:
        sr = wf.getframerate()
        total_frames = wf.getnframes()
        duration = total_frames / sr
        chunk_frames = int(sr * chunk_duration_ms / 1000)
        sent = 0
        while True:
            frames = wf.readframes(chunk_frames)
            if not frames:
                break
            await ws.send(frames)
            sent += len(frames) // 2
            await asyncio.sleep(chunk_duration_ms / 1000)
        return duration


async def collect_messages(ws, timeout: float = 30.0, stop_on: str = None,
                           transcript: list = None, protocol: dict = None) -> list[dict]:
    messages = []
    try:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            msg = json.loads(raw)
            messages.append(msg)
            msg_type = msg.get("type", "?")
            status = msg.get("status", "")

            if msg_type == "transcript_update":
                for u in msg.get("utterances", []):
                    if transcript is not None:
                        transcript.append({"speaker": u["speaker"], "text": u["text"]})
            elif msg_type == "protocol_update":
                if protocol is not None:
                    protocol.update(msg.get("protocol", {}))
            elif msg_type == "calibration_complete":
                pi = msg.get("patient_info", {})
                print(f"      Calibration: {pi.get('full_name', '?')}, {pi.get('age', '?')} лет")

            if stop_on and status == stop_on:
                break
    except asyncio.TimeoutError:
        pass
    except websockets.ConnectionClosed:
        print("      ⚠ Connection closed")
    return messages


# ============================================================
# Run single scenario
# ============================================================

async def run_scenario(scenario: dict, audio_dir: str, url: str) -> dict:
    """Run a single test scenario. Returns result dict for evaluation."""
    sid = scenario["id"]
    cal_wav = os.path.join(audio_dir, sid, "calibration.wav")
    exam_wav = os.path.join(audio_dir, sid, "exam.wav")

    if not os.path.exists(cal_wav):
        print(f"    ⚠ Missing: {cal_wav}")
        return None
    if not os.path.exists(exam_wav):
        print(f"    ⚠ Missing: {exam_wav}")
        return None

    transcript = []
    protocol = {}
    t0 = time.monotonic()

    try:
        async with websockets.connect(url, ping_interval=60, ping_timeout=120) as ws:
            # Stage 1: Calibration
            await ws.send(json.dumps({"type": "start_calibration", "config": {"num_speakers": 2}}))
            await asyncio.sleep(0.3)
            await stream_wav(ws, cal_wav)
            await collect_messages(ws, timeout=10, transcript=transcript, protocol=protocol)
            await ws.send(json.dumps({"type": "stop_calibration"}))
            await collect_messages(ws, timeout=300, stop_on="calibrated",
                                   transcript=transcript, protocol=protocol)

            # Stage 2: Recording
            await ws.send(json.dumps({"type": "start_recording"}))
            await asyncio.sleep(0.3)
            await stream_wav(ws, exam_wav)
            await collect_messages(ws, timeout=300, transcript=transcript, protocol=protocol)

            # Stage 3: Stop
            await ws.send(json.dumps({"type": "stop_recording"}))
            await collect_messages(ws, timeout=300, stop_on="stopped",
                                   transcript=transcript, protocol=protocol)

            # Stage 4: Finalize
            await ws.send(json.dumps({"type": "finalize"}))
            await collect_messages(ws, timeout=120, stop_on="done",
                                   transcript=transcript, protocol=protocol)

        elapsed = time.monotonic() - t0
        print(f"      ✅ Done in {elapsed:.0f}s | {len(transcript)} utterances")

        return {
            "scenario_id": sid,
            "transcript": transcript,
            "protocol": protocol,
            "elapsed_seconds": round(elapsed, 1),
        }

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"      ❌ Failed after {elapsed:.0f}s: {e}")
        return {
            "scenario_id": sid,
            "transcript": transcript,
            "protocol": protocol,
            "elapsed_seconds": round(elapsed, 1),
            "error": str(e),
        }


# ============================================================
# Main
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="Run all test scenarios and evaluate")
    parser.add_argument("--scenarios", nargs="+", help="Scenario IDs to run (default: all)")
    parser.add_argument("--hard-only", action="store_true", help="Run only same-gender (hard) scenarios")
    parser.add_argument("--url", default="ws://localhost:8000/ws/session", help="WebSocket URL")
    parser.add_argument("--audio-dir", default="test_scenarios_audio", help="Audio directory")
    parser.add_argument("--results-dir", default="results", help="Results output directory")
    parser.add_argument("--no-eval", action="store_true", help="Skip evaluation after tests")
    args = parser.parse_args()

    # Select scenarios
    scenarios = SCENARIOS
    if args.scenarios:
        scenarios = [s for s in SCENARIOS if s["id"] in args.scenarios]
    elif args.hard_only:
        scenarios = [s for s in SCENARIOS if s["id"] in SAME_GENDER_IDS]

    if not scenarios:
        print("No scenarios selected")
        return

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Aiston TT — Automated Test Suite")
    print(f"Server: {args.url}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Results dir: {args.results_dir}")
    print(f"{'='*70}")

    results = []
    for i, scenario in enumerate(scenarios):
        sid = scenario["id"]
        hard = " ⚠️HARD" if sid in SAME_GENDER_IDS else ""
        print(f"\n[{i+1}/{len(scenarios)}] {sid}{hard}")

        result = await run_scenario(scenario, args.audio_dir, args.url)
        if result:
            # Save individual result
            result_path = os.path.join(args.results_dir, f"{sid}.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            results.append(result)
            print(f"      📊 Saved: {result_path}")

        # Brief pause between scenarios to let backend cleanup
        if i < len(scenarios) - 1:
            print("      ⏳ Waiting 3s before next scenario...")
            await asyncio.sleep(3)

    print(f"\n{'='*70}")
    print(f"Completed: {len(results)}/{len(scenarios)} scenarios")
    print(f"{'='*70}")

    # Run evaluation
    if not args.no_eval and results:
        print("\n📊 Running evaluation...")
        try:
            from evaluate_test import evaluate_result, print_report
            metrics = []
            for r in results:
                try:
                    m = evaluate_result(r)
                    metrics.append(m)
                except Exception as e:
                    print(f"  ⚠ Eval failed for {r['scenario_id']}: {e}")
            if metrics:
                print_report(metrics)
        except ImportError:
            print("  ⚠ evaluate_test.py not found, skipping evaluation")
            print(f"  Run manually: python evaluate_test.py {args.results_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
