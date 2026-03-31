import logging
from fastapi import APIRouter, HTTPException
from ..core.session import session_manager
from ..models.protocol import MedicalProtocol

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


@router.get("/health")
async def health():
    import torch
    from ..config import settings
    gpu = torch.cuda.is_available()
    return {
        "status": "ok", "version": "3.1",
        "gpu_available": gpu,
        "gpu_name": torch.cuda.get_device_name(0) if gpu else None,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
    }


@router.get("/session/{session_id}/protocol", response_model=MedicalProtocol)
async def get_protocol(session_id: str):
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(404, "Session not found")
    return s.protocol
