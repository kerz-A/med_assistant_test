"""WebSocket handler v4.1: VAD-driven segmentation + speaker embeddings.

Calibration: collect VAD segments during streaming, process sequentially at stop.
Recording: fire-and-forget tasks per VAD segment, batched LLM extraction.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..core.pipeline import ProcessingPipeline
from ..core.session import SessionState, SessionStage, session_manager
from ..core.vad_segmenter import VADSegmenter, SpeechSegment
from ..models.messages import (
    CalibrationComplete,
    ClientMessage,
    ClientMessageType,
    ProtocolUpdateMessage,
    StatusMessage,
    TranscriptUpdate,
)

logger = logging.getLogger(__name__)


async def _safe_send(ws: WebSocket, text: str) -> bool:
    try:
        await ws.send_text(text)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False


async def _send_protocol(ws: WebSocket, session: SessionState) -> None:
    msg = ProtocolUpdateMessage(
        protocol=session.protocol,
        filled_fields=session.get_filled_fields(),
    )
    await _safe_send(ws, msg.model_dump_json())


def create_websocket_router(pipeline: ProcessingPipeline) -> APIRouter:
    ws_router = APIRouter()

    @ws_router.websocket("/ws/session")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        client = ws.client.host if ws.client else "?"
        logger.info("[WS] Connected: %s", client)
        session: SessionState | None = None
        vad: VADSegmenter | None = None

        async def on_recording_segment(segment: SpeechSegment) -> None:
            """Process a VAD speech segment during recording (runs as async task)."""
            if session is None or session.stage != SessionStage.RECORDING:
                return

            await _safe_send(ws, StatusMessage(status="processing").model_dump_json())

            utterances = await pipeline.process_segment(session, segment)
            if utterances:
                await _safe_send(ws, TranscriptUpdate(
                    utterances=utterances, processing_time_ms=0,
                ).model_dump_json())
                await _send_protocol(ws, session)

            await _safe_send(ws, StatusMessage(status="recording").model_dump_json())

        try:
            while True:
                message = await ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                # Binary audio — feed to VAD
                if "bytes" in message and message["bytes"] and session and vad:
                    if session.stage in (SessionStage.CALIBRATING, SessionStage.RECORDING):
                        session.add_audio(message["bytes"])
                        segments = await vad.feed(message["bytes"])

                        for seg in segments:
                            if session.stage == SessionStage.CALIBRATING:
                                # Buffer for deferred processing at stop_calibration
                                session.buffer_calibration_segment(seg)
                            elif session.stage == SessionStage.RECORDING:
                                # Fire-and-forget for realtime results
                                task = asyncio.create_task(on_recording_segment(seg))
                                session.track_task(task)
                    continue

                # Text messages
                if "text" not in message or not message["text"]:
                    continue

                try:
                    data = json.loads(message["text"])
                    msg = ClientMessage.model_validate(data)
                except (json.JSONDecodeError, ValueError) as e:
                    await _safe_send(ws, StatusMessage(status="error", message=str(e)).model_dump_json())
                    continue

                logger.info("[WS] Message: %s (session=%s)", msg.type.value,
                            session.session_id if session else "none")

                # ---- STAGE 1: START CALIBRATION ----
                if msg.type == ClientMessageType.START_CALIBRATION:
                    session = session_manager.create_session(num_speakers=msg.config.num_speakers)
                    session.stage = SessionStage.CALIBRATING

                    vad = VADSegmenter()
                    vad.load_model()

                    await _safe_send(ws, StatusMessage(
                        status="calibrating",
                        message="Спросите пациента: ФИО, полный возраст, пол",
                    ).model_dump_json())
                    await _send_protocol(ws, session)

                # ---- STOP CALIBRATION ----
                elif msg.type == ClientMessageType.STOP_CALIBRATION and session and vad:
                    if session.stage == SessionStage.CALIBRATING:
                        await _safe_send(ws, StatusMessage(
                            status="processing", message="Обработка калибровки...",
                        ).model_dump_json())

                        # Flush remaining speech from VAD
                        final_seg = await vad.flush()
                        if final_seg:
                            session.buffer_calibration_segment(final_seg)

                        # Process ALL buffered segments SEQUENTIALLY
                        cal_segments = session.get_calibration_segments()
                        logger.info("[WS] Processing %d calibration segments sequentially",
                                    len(cal_segments))

                        for seg in cal_segments:
                            utterances = await pipeline.process_calibration_segment(session, seg)
                            if utterances:
                                await _safe_send(ws, TranscriptUpdate(
                                    utterances=utterances, processing_time_ms=0,
                                ).model_dump_json())

                        # Finalize: extract patient info from completed transcript
                        await pipeline.finalize_calibration(session)

                        # Validate calibration results
                        if session.doctor_profile.count == 0 and session.patient_profile.count == 0:
                            logger.error("[WS] Calibration failed: no speech detected")
                            await _safe_send(ws, StatusMessage(
                                status="error",
                                message="Калибровка не удалась: речь не обнаружена. Попробуйте снова.",
                            ).model_dump_json())
                            session.stage = SessionStage.IDLE
                            continue

                        if session.doctor_profile.count == 0:
                            logger.warning("[WS] No doctor segments during calibration")
                        if session.patient_profile.count == 0:
                            logger.warning("[WS] No patient segments during calibration")

                        session.calibration_end_time = session.audio_buffer.real_total_duration
                        session.stage = SessionStage.CALIBRATED

                        pi = session.protocol.patient_info
                        await _safe_send(ws, CalibrationComplete(
                            patient_info=pi,
                            message=f"Калибровка завершена. Пациент: {pi.full_name or '?'}, {pi.age or '?'} лет, {pi.gender or '?'}",
                        ).model_dump_json())
                        await _send_protocol(ws, session)
                        await _safe_send(ws, StatusMessage(
                            status="calibrated",
                            message="Нажмите «Запись» для начала приёма.",
                        ).model_dump_json())

                        logger.info(
                            "[WS] Calibration done: patient=%s, doctor_embs=%d, patient_embs=%d",
                            session.protocol.patient_info.full_name,
                            session.doctor_profile.count,
                            session.patient_profile.count,
                        )

                # ---- STAGE 2: START RECORDING ----
                elif msg.type == ClientMessageType.START_RECORDING and session and vad:
                    if session.stage == SessionStage.CALIBRATED:
                        session.stage = SessionStage.RECORDING
                        # Clear any leftover calibration data, reset VAD
                        session.get_calibration_segments()
                        vad.reset()
                        await _safe_send(ws, StatusMessage(status="recording").model_dump_json())
                        logger.info("[WS] Recording started: session=%s", session.session_id)

                # ---- STAGE 3: STOP RECORDING ----
                elif msg.type == ClientMessageType.STOP_RECORDING and session and vad:
                    if session.stage == SessionStage.RECORDING:
                        # Flush remaining speech
                        final_seg = await vad.flush()
                        if final_seg:
                            await on_recording_segment(final_seg)

                        # Wait for all recording tasks
                        await session.wait_for_processing_complete(timeout=60.0)

                        # Flush remaining pending utterances through LLM
                        if session.has_pending_utterances():
                            pending = session.get_pending_utterances()
                            recent = session.transcript[-15:]
                            context = "\n".join(
                                f"[{'Врач' if u.speaker == 'doctor' else 'Пациент'}]: {u.text}"
                                for u in recent
                            )
                            session.protocol = await pipeline.llm.extract_protocol_data(
                                session.protocol, pending, context,
                            )

                        session.stage = SessionStage.STOPPED
                        await _safe_send(ws, StatusMessage(
                            status="stopped",
                            message="Запись остановлена. Проверьте и отредактируйте поля протокола.",
                        ).model_dump_json())
                        await _send_protocol(ws, session)
                        logger.info("[WS] Recording stopped: session=%s duration=%.1fs",
                                    session.session_id, session.audio_buffer.real_total_duration)

                # ---- STAGE 4: FINALIZE ----
                elif msg.type == ClientMessageType.FINALIZE and session:
                    if session.stage == SessionStage.STOPPED:
                        session.stage = SessionStage.FINALIZING
                        await _safe_send(ws, StatusMessage(
                            status="finalizing", message="Формирование заключения...",
                        ).model_dump_json())

                        time_ms = await pipeline.finalize(session)

                        session.stage = SessionStage.DONE
                        await _send_protocol(ws, session)
                        await _safe_send(ws, StatusMessage(
                            status="done",
                            message=f"Заключение готово ({time_ms}мс). Проверьте результат.",
                        ).model_dump_json())
                        logger.info("[WS] Finalized: session=%s in %dms", session.session_id, time_ms)

                # ---- EDIT FIELD (after stop) ----
                elif msg.type == ClientMessageType.EDIT_FIELD and session:
                    if session.stage in (SessionStage.STOPPED, SessionStage.DONE) and msg.field and msg.value is not None:
                        proto = session.protocol
                        if hasattr(proto.exam_data, msg.field):
                            setattr(proto.exam_data, msg.field, msg.value)
                        elif hasattr(proto.vitals, msg.field):
                            setattr(proto.vitals, msg.field, msg.value)
                        elif hasattr(proto.patient_info, msg.field):
                            setattr(proto.patient_info, msg.field, msg.value)
                        await _send_protocol(ws, session)
                        logger.info("[WS] Field edited: %s", msg.field)

        except WebSocketDisconnect:
            logger.info("[WS] Disconnected: session=%s", session.session_id if session else "none")
        except Exception as e:
            logger.exception("[WS] Error: %s", e)
        finally:
            if session:
                logger.info("[WS] Cleanup: session=%s stage=%s", session.session_id, session.stage.value)
                await session.cancel_all_tasks()
                session_manager.remove_session(session.session_id)

    return ws_router
