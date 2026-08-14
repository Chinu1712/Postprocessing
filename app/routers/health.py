"""Liveness and service-description endpoints (unauthenticated)."""

from __future__ import annotations

import platform
import time
from typing import Any

from fastapi import APIRouter

from ..config import get_settings

router = APIRouter(tags=["meta"])

_STARTED_AT = time.time()


@router.get("/health", summary="Liveness probe")
@router.get("/healthz", include_in_schema=False)
async def health() -> dict[str, Any]:
    settings = get_settings()
    return {
        "status": "ok",
        "service": settings.app_name,
        "version": settings.version,
        "uptime_seconds": round(time.time() - _STARTED_AT, 1),
        "python": platform.python_version(),
        "auth_required": bool(settings.api_keys),
    }


@router.get("/", summary="Service description")
async def root() -> dict[str, Any]:
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.version,
        "description": "REST postprocessing for segmentation and pose model output.",
        "endpoints": {
            "POST /postprocess/segmentation": "Postprocess instance-segmentation results",
            "POST /postprocess/pose": "Postprocess pose / keypoint results",
            "POST /postprocess": "Route on task or model id",
            "POST /postprocess/batch": "Several frames in one call",
            "GET /postprocess/params": "Every tuning parameter and its default",
            "GET /postprocess/models": "Known model ids and their tasks",
            "GET /health": "Liveness probe",
            "GET /docs": "Interactive OpenAPI docs",
        },
        "defaults": {
            "confidence_threshold": settings.confidence_threshold,
            "iou_threshold": settings.iou_threshold,
            "keypoint_threshold": settings.keypoint_threshold,
            "oks_threshold": settings.oks_threshold,
        },
    }
