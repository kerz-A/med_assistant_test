"""WebSocket handler v3: calibrate → record → edit → finalize."""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import settings
from ..core.pipeline import ProcessingPipeline
from ..core.session import SessionState, SessionStage, session_manager
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


async def _run_processing_cycle(
    ws: WebSocket, session: SessionState, pipeline: ProcessingPipeline,
) -> None:
    """Triggered every N seconds during calibration/recording."""
    if not await session.acquire_processing():
        return

    try:
        await _safe_send(ws, StatusMessage(status="processing").model_dump_json())

        utterances, time_ms = await pipeline.process_cycle(session)

        if utterances:
            await _safe_send(ws, TranscriptUpdate(
                utterances=utterances, processing_time_ms=time_ms,
            ).model_dump_json())
            await _send_protocol(ws, session)

        await _safe_send(ws, StatusMessage(status="recording").model_dump_json())

    except WebSocketDisconnect:
        logger.warning("[WS] Disconnected during processing")
    except Exception as e:
        logger.exception("[WS] Processing error: %s", e)
        await _safe_send(ws, StatusMessage(status="error", message=str(e)).model_dump_json())
    finally:
        session.release_processing()

        # Catch-up — if more audio accumulated during processing, run again
        if (session.stage == SessionStage.RECORDING
                and session.audio_buffer.unprocessed_duration >= settings.processing_interval_seconds):
            logger.info("[WS] Catch-up: %.1fs unprocessed, starting next cycle",
                        session.audio_buffer.unprocessed_duration)
            task = asyncio.create_task(_run_processing_cycle(ws, session, pipeline))
            session.track_task(task)


def create_websocket_router(pipeline: ProcessingPipeline) -> APIRouter:
    ws_router = APIRouter()

    @ws_router.websocket("/ws/session")
    async def websocket_endpoint(ws: WebSocket):
        await ws.accept()
        client = ws.client.host if ws.client else "?"
        logger.info("[WS] Connected: %s", client)
        session: SessionState | None = None

        try:
            while True:
                message = await ws.receive()

                if message.get("type") == "websocket.disconnect":
                    break

                # Binary audio
                if "bytes" in message and message["bytes"] and session:
                    if session.stage in (SessionStage.CALIBRATING, SessionStage.RECORDING):
                        session.add_audio(message["bytes"])
                        if session.should_process():
                            task = asyncio.create_task(_run_processing_cycle(ws, session, pipeline))
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
                    await _safe_send(ws, StatusMessage(
                        status="calibrating",
                        message="Спросите пациента: ФИО, полный возраст, пол",
                    ).model_dump_json())
                    await _send_protocol(ws, session)

                # ---- STOP CALIBRATION ----
                elif msg.type == ClientMessageType.STOP_CALIBRATION and session:
                    if session.stage == SessionStage.CALIBRATING:
                        await _safe_send(ws, StatusMessage(
                            status="processing", message="Обработка калибровки...",
                        ).model_dump_json())

                        # Process ALL calibration audio as one batch
                        utterances, time_ms = await pipeline.process_full_calibration(session)

                        if utterances:
                            await _safe_send(ws, TranscriptUpdate(
                                utterances=utterances, processing_time_ms=time_ms,
                            ).model_dump_json())

                        if not session.speaker_map:
                            session.speaker_map = {"SPEAKER_00": "doctor", "SPEAKER_01": "patient"}
                            logger.warning("[WS] Calibration: no speakers detected, using defaults")

                        session.calibration_end_time = session.audio_buffer.real_total_duration
                        logger.info("[WS] Calibration end: %.1fs, patient=%s", session.calibration_end_time, session.protocol.patient_info)
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

                # ---- STAGE 2: START RECORDING ----
                elif msg.type == ClientMessageType.START_RECORDING and session:
                    if session.stage == SessionStage.CALIBRATED:
                        session.stage = SessionStage.RECORDING
                        session.audio_buffer.mark_processed()  # reset unprocessed counter
                        await _safe_send(ws, StatusMessage(status="recording").model_dump_json())
                        logger.info("[WS] Recording started: session=%s", session.session_id)

                # ---- STAGE 3: STOP RECORDING ----
                elif msg.type == ClientMessageType.STOP_RECORDING and session:
                    if session.stage == SessionStage.RECORDING:
                        # Wait for any in-flight processing to finish
                        await session.wait_for_processing_complete(timeout=15.0)

                        # Process remaining audio
                        if session.audio_buffer.unprocessed_duration > 0.5:
                            await _run_processing_cycle(ws, session, pipeline)

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
